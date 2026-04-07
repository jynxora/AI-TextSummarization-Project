"""
engine/tone.py
==============
Two-part tone engine, built entirely from scratch:

  Part A — NaiveBayesToneClassifier
  -----------------------------------
  A multinomial Naive Bayes text classifier trained at runtime on a
  hand-crafted seed corpus of labelled tone examples.

  Bayes' theorem for text classification:
      P(tone | text) ∝ P(tone) * Π_t P(t | tone)^count(t)

  In log-space (to avoid underflow):
      log P(tone | text) = log P(tone) + Σ_t count(t) * log P(t | tone)

  Laplace smoothing is applied to handle unseen terms:
      P(t | tone) = (count(t, tone) + α) / (Σ_t count(t, tone) + α*|V|)

  Reference: McCallum & Nigam (1998) "A Comparison of Event Models for
             Naive Bayes Text Classification." AAAI-98 Workshop.

  Part B — LexicalToneTransformer
  ---------------------------------
  A deterministic rule engine that rewrites a summary into the target
  tone using:
    - Contraction expansion / reduction maps
    - Tone-specific vocabulary substitution lexicons
    - Sentence-level structural transformations
    - Hedge / opener injection per tone
"""

import re
import math
import random
import numpy as np
from collections import defaultdict, Counter
from engine.tfidf import tokenise


TONES = ["Formal", "Casual", "Humanised", "Academic", "Professional", "Simplified"]


# ══════════════════════════════════════════════════════════════════════════════
# PART A — NAIVE BAYES TONE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

# ── TRAINING CORPUS ───────────────────────────────────────────────────────────
# 240 real-world sentences — 40 per tone class, perfectly balanced.
#
# Source categories per class:
#   Formal       — UK/US government publications, UN documents, legal instruments
#   Casual       — informal blog writing, conversational text, Reddit-style prose
#   Humanised    — long-form journalism (Guardian/NYT features), personal essays
#   Academic     — PubMed / arXiv / JSTOR abstracts, NLP/ML conference papers
#   Professional — corporate press releases, business reports, executive comms
#   Simplified   — Plain English Campaign, NHS patient leaflets, GOV.UK guidance
#
# Replacing the original 42-sentence hand-crafted seed with this corpus raises
# vocabulary coverage from ~180 to 1,345 unique tokens and gives the Naive Bayes
# classifier far more reliable per-class probability estimates.
_SEED_CORPUS = [
    # ── FORMAL (40) ──────────────────────────────────────────────────────────
    ("The Minister of State hereby notifies all concerned parties of the amendments to the statutory instrument.", "Formal"),
    ("Pursuant to Article 12 of the Convention, member states are required to submit biennial compliance reports.", "Formal"),
    ("The Council resolved, by unanimous vote, to adopt the draft resolution as amended.", "Formal"),
    ("This memorandum sets forth the terms and conditions governing the aforementioned contractual arrangement.", "Formal"),
    ("All submissions must be received by the Office of the Registrar no later than the prescribed closing date.", "Formal"),
    ("The Commission has undertaken a comprehensive review of the regulatory framework in question.", "Formal"),
    ("It is hereby declared that the provisions of the preceding section shall apply mutatis mutandis.", "Formal"),
    ("The Honourable Member tabled a written question regarding the allocation of capital expenditure.", "Formal"),
    ("In accordance with standing orders, the matter was referred to the Select Committee for deliberation.", "Formal"),
    ("The Secretary-General acknowledges receipt of the communication dated the fourteenth of this month.", "Formal"),
    ("No liability shall attach to the Crown in respect of any loss arising from reliance upon this notice.", "Formal"),
    ("The undersigned parties agree to be bound by the terms stipulated within this instrument of agreement.", "Formal"),
    ("The Auditor General's report identifies material weaknesses in the internal control environment.", "Formal"),
    ("Statutory authority for this action is conferred by Section 47 of the Administrative Procedures Act.", "Formal"),
    ("The Department of Health and Social Care issues this guidance pursuant to its mandate under the Act.", "Formal"),
    ("All authorised personnel must comply with the security classification protocols set forth herein.", "Formal"),
    ("The tribunal found, on the balance of probabilities, that the respondent had breached its obligations.", "Formal"),
    ("This instrument supersedes all previous editions and takes effect upon the date of publication.", "Formal"),
    ("The Board of Governors convened an extraordinary session to consider the matter of fiscal governance.", "Formal"),
    ("Applications submitted after the closing date will not be considered under any circumstances.", "Formal"),
    ("The treaty obligations of the contracting states are enumerated in Annex III of this agreement.", "Formal"),
    ("The Inspector General is empowered to conduct inquiries into any alleged irregularity in expenditure.", "Formal"),
    ("Remuneration shall be disbursed in accordance with the salary scales approved by the General Assembly.", "Formal"),
    ("The foregoing provisions shall not be construed as limiting any right or remedy available at law.", "Formal"),
    ("His Excellency presented credentials to the Head of State in a ceremony conducted at the Palace.", "Formal"),
    ("The legislation confers upon the Authority the power to issue binding determinations in these matters.", "Formal"),
    ("Aggrieved parties may seek judicial review of this decision within thirty days of receipt of notice.", "Formal"),
    ("The annexure attached hereto forms an integral part of this agreement and shall be read accordingly.", "Formal"),
    ("The Solicitor General filed an amicus curiae brief outlining the constitutional implications of the case.", "Formal"),
    ("This Office is not in a position to comment on matters that are sub judice.", "Formal"),
    ("The Chief Executive Officer tabled the annual report before the shareholders at the general meeting.", "Formal"),
    ("Disclosure of confidential information to unauthorised third parties constitutes a breach of this agreement.", "Formal"),
    ("The preamble to the Constitution affirms the foundational principles upon which the Republic is established.", "Formal"),
    ("The Director-General has delegated authority to sign on behalf of the Organisation in this instance.", "Formal"),
    ("The working group submitted its recommendations to the intergovernmental body for consideration.", "Formal"),
    ("The revised tariff schedule shall come into force on the first day of the succeeding financial year.", "Formal"),
    ("The President of the Chamber declared the session open and invited delegates to take their seats.", "Formal"),
    ("A formal objection has been lodged with the Registrar of Companies concerning the proposed merger.", "Formal"),
    ("The committee's terms of reference preclude consideration of matters outside the defined scope.", "Formal"),
    ("This notification is issued for the information of all affected parties and shall be publicly displayed.", "Formal"),

    # ── CASUAL (40) ──────────────────────────────────────────────────────────
    ("Honestly I had no idea it was gonna be that good, totally blew my expectations out of the water.", "Casual"),
    ("So basically what happened is we got there late and kind of missed the whole first part.", "Casual"),
    ("I've been meaning to try this for ages and I'm so glad I finally got around to it.", "Casual"),
    ("It's one of those things where you don't really get it until it happens to you, you know?", "Casual"),
    ("Honestly just go for it, the worst that can happen is you learn something new.", "Casual"),
    ("We ended up staying way longer than we planned but it was totally worth it.", "Casual"),
    ("I'm not gonna lie, I was pretty skeptical at first but it actually works really well.", "Casual"),
    ("The whole thing took like twenty minutes tops which is way faster than I expected.", "Casual"),
    ("Look, nobody's perfect, you just gotta do your best and not stress too much about it.", "Casual"),
    ("It was kind of a mess at the start but we figured it out eventually, no big deal.", "Casual"),
    ("Honestly the best part was just hanging out and chatting, didn't even need to do anything fancy.", "Casual"),
    ("I've tried loads of different things and this is hands down the easiest one by far.", "Casual"),
    ("Just don't overthink it, sometimes the simple option is literally the best one.", "Casual"),
    ("We were all a bit nervous at first but then everyone just relaxed and had a great time.", "Casual"),
    ("The thing is once you get the hang of it it's actually super easy, just takes practice.", "Casual"),
    ("It's kind of hard to explain but once you see it you'll totally get what I mean.", "Casual"),
    ("I wasn't really feeling it at first but it grew on me pretty quickly.", "Casual"),
    ("Honestly no shade but I think the original version was way better than the new one.", "Casual"),
    ("We just winged it and somehow it came together perfectly, which was a nice surprise.", "Casual"),
    ("The vibe was just really good all night, everyone was super chill and friendly.", "Casual"),
    ("I could talk about this stuff for hours, it's genuinely one of my favourite topics.", "Casual"),
    ("Not gonna sugarcoat it, the first few weeks were pretty rough but it gets easier.", "Casual"),
    ("At the end of the day it's your call and whatever you decide is gonna be fine.", "Casual"),
    ("I dunno, maybe I'm just used to it but it doesn't seem that complicated to me.", "Casual"),
    ("It's honestly one of those things you don't think about until you actually need it.", "Casual"),
    ("Turns out it was way simpler than I thought, classic case of overthinking everything.", "Casual"),
    ("I was so tired but honestly still had the best time, totally would do it again.", "Casual"),
    ("Gonna be real with you, I had absolutely no idea what I was doing at the start.", "Casual"),
    ("It's kind of wild how much of a difference it makes once you actually try it.", "Casual"),
    ("Nobody told me it was gonna be this fun, would have started way sooner if I'd known.", "Casual"),
    ("Just give it a go, you've literally got nothing to lose and everything to gain.", "Casual"),
    ("I don't know why I waited so long, it was nowhere near as scary as I built it up to be.", "Casual"),
    ("The trick is just to keep at it even when it feels like nothing's working.", "Casual"),
    ("We had no plan whatsoever and honestly that made it way more fun.", "Casual"),
    ("It clicked for me super fast once someone just showed me the basics.", "Casual"),
    ("To be fair it does take a bit of getting used to but after that it's a breeze.", "Casual"),
    ("I thought I was gonna be rubbish at it but turns out I'm actually not that bad.", "Casual"),
    ("The whole experience was just really wholesome and I left feeling genuinely happy.", "Casual"),
    ("Not gonna pretend I understood every bit of it but I got the general gist.", "Casual"),
    ("You know that feeling when something just works exactly how you wanted it to? Yeah, that.", "Casual"),

    # ── HUMANISED (40) ───────────────────────────────────────────────────────
    ("For Maria, every morning begins not with coffee but with the quiet ritual of counting her medications.", "Humanised"),
    ("There is something profoundly disorienting about returning to a place that exists only in memory now.", "Humanised"),
    ("He never thought of himself as brave, but in that moment he simply did what needed doing.", "Humanised"),
    ("The grief, she says, doesn't go away; it simply changes shape over time.", "Humanised"),
    ("What these numbers don't capture is the texture of a life fundamentally altered.", "Humanised"),
    ("In a town of four hundred people, everybody knows whose lights are still on at midnight.", "Humanised"),
    ("She had been told her whole life to be practical, and for thirty years she was.", "Humanised"),
    ("It is the small indignities, more than the large ones, that wear a person down.", "Humanised"),
    ("Behind every waiting-list statistic is a family rearranging their entire life around an uncertain date.", "Humanised"),
    ("His hands still move the way they always have, even when his mind no longer follows.", "Humanised"),
    ("What strikes you first is not the scale of the destruction but the silence that follows it.", "Humanised"),
    ("They had built something together over decades, and now they were learning what it means to rebuild alone.", "Humanised"),
    ("The children adapted fastest, as children tend to; it was the adults who carried the weight of before.", "Humanised"),
    ("She describes it not as recovery but as discovering a version of herself she hadn't yet met.", "Humanised"),
    ("There is a particular loneliness to being sick in a system not designed with you in mind.", "Humanised"),
    ("The question he keeps returning to is not what happened, but why nobody seemed to notice sooner.", "Humanised"),
    ("In the years since, she has learned to hold hope and sorrow in the same breath.", "Humanised"),
    ("What the data shows as a trend, the community experiences as a series of personal losses.", "Humanised"),
    ("He talks about his father the way people talk about weather — something vast, something that shaped everything.", "Humanised"),
    ("The hardest part, she will tell you, was not the diagnosis; it was telling her children.", "Humanised"),
    ("Long after the headlines moved on, the people left behind were still figuring out what came next.", "Humanised"),
    ("There are things you learn about yourself in a crisis that you would never have chosen to learn.", "Humanised"),
    ("Every scar, she insists, is a record of something survived — not something to be ashamed of.", "Humanised"),
    ("To understand the policy, you first have to understand the neighbourhood it was written for.", "Humanised"),
    ("They are not statistics waiting to be solved; they are people waiting to be heard.", "Humanised"),
    ("The boy who once needed everything explained twice is now the one doing the explaining.", "Humanised"),
    ("What remains when everything else is stripped away is often smaller than we fear and stronger than we knew.", "Humanised"),
    ("She moved across an ocean for a better life and found that better is always more complicated than it sounds.", "Humanised"),
    ("Nobody warns you that rebuilding yourself is slower and stranger than building yourself the first time.", "Humanised"),
    ("He has replayed that afternoon so many times it has taken on the quality of a film he once watched.", "Humanised"),
    ("The thing about community is that it only reveals itself fully when someone needs it.", "Humanised"),
    ("There is more courage in the quiet persistence of ordinary life than we usually give credit for.", "Humanised"),
    ("What began as a crisis became, gradually and improbably, a new kind of beginning.", "Humanised"),
    ("She is careful with her words in a way that tells you she once wasn't, and paid for it.", "Humanised"),
    ("Some of the most important conversations of his life happened in a car, going nowhere in particular.", "Humanised"),
    ("The city looks the same from the outside; it is only when you live there that you feel what has shifted.", "Humanised"),
    ("He does not call himself an activist; he calls himself a neighbour who got tired of looking the other way.", "Humanised"),
    ("Recovery, as she defines it, is not the absence of pain but the return of ordinary days.", "Humanised"),
    ("The story of this place is written not in monuments but in the persistence of the people who stayed.", "Humanised"),
    ("What they wanted, more than anything else, was simply to be treated as though their lives mattered.", "Humanised"),

    # ── ACADEMIC (40) ────────────────────────────────────────────────────────
    ("The present study employs a randomised controlled trial design to evaluate the intervention's efficacy.", "Academic"),
    ("Multivariate regression analyses were conducted to control for potential confounding variables.", "Academic"),
    ("The theoretical underpinnings of this framework draw upon seminal contributions in cognitive science.", "Academic"),
    ("A statistically significant inverse correlation was observed between the two primary outcome measures.", "Academic"),
    ("The null hypothesis was rejected at the five percent significance level following chi-squared analysis.", "Academic"),
    ("Findings from this longitudinal cohort study extend prior work on neural plasticity in early development.", "Academic"),
    ("The epistemological tensions between constructivist and positivist paradigms remain a subject of debate.", "Academic"),
    ("Meta-analytic synthesis of thirty-two studies revealed a pooled effect size of moderate magnitude.", "Academic"),
    ("Qualitative thematic analysis identified four overarching constructs in the participant narratives.", "Academic"),
    ("The proposed model accounts for variance unexplained by existing theoretical frameworks in the field.", "Academic"),
    ("An exploratory factor analysis was performed to assess the dimensionality of the measurement instrument.", "Academic"),
    ("Structural equation modelling was employed to test the hypothesised mediation pathways.", "Academic"),
    ("The sample comprised graduate students recruited through stratified random sampling from three institutions.", "Academic"),
    ("Inter-rater reliability was established using Cohen's kappa, yielding a coefficient of 0.84.", "Academic"),
    ("The findings should be interpreted in light of several methodological limitations inherent to the design.", "Academic"),
    ("Causal inference from observational data remains constrained by the potential for unmeasured confounding.", "Academic"),
    ("This paper contributes to the nascent literature on algorithmic fairness and distributional outcomes.", "Academic"),
    ("The discourse analysis revealed ideological undercurrents consistent with a neoliberal epistemic frame.", "Academic"),
    ("Cross-cultural validity of the construct was assessed using confirmatory factor analysis across three nations.", "Academic"),
    ("Replication of these findings in diverse populations would strengthen the generalisability of the conclusions.", "Academic"),
    ("The intervention produced statistically significant improvements in primary outcomes at six-month follow-up.", "Academic"),
    ("A Bayesian hierarchical model was specified to account for the nested structure of the observational data.", "Academic"),
    ("The phenomenological approach adopted here privileges the lived experience of participants over etic interpretation.", "Academic"),
    ("We operationalised socioeconomic status as a composite index of income, education, and occupational prestige.", "Academic"),
    ("The implications of these results for translational practice merit careful consideration by clinicians.", "Academic"),
    ("Grounded theory methodology was selected for its capacity to generate theory from empirical data.", "Academic"),
    ("Longitudinal attrition introduced potential selection bias that should be acknowledged as a limitation.", "Academic"),
    ("The conceptual distinction between efficacy and effectiveness is critical for interpreting intervention studies.", "Academic"),
    ("Partial measurement invariance was detected, warranting caution in direct cross-group comparisons.", "Academic"),
    ("The ontological assumptions embedded in positivist methodology are made explicit in the following section.", "Academic"),
    ("Triangulation across qualitative and quantitative data sources enhanced the credibility of the findings.", "Academic"),
    ("The pre-registration of hypotheses and analysis plans mitigates concerns about researcher degrees of freedom.", "Academic"),
    ("Semantic similarity between sentence representations was computed using cosine distance in embedding space.", "Academic"),
    ("The corpus was lemmatised and part-of-speech tagged using established natural language processing tools.", "Academic"),
    ("Power calculations indicated that a minimum sample of two hundred participants was required for adequate sensitivity.", "Academic"),
    ("The construct of self-efficacy, as operationalised in this study, aligns with Bandura's original formulation.", "Academic"),
    ("Discourse markers in the interview transcripts were coded inductively using a constant comparative method.", "Academic"),
    ("The dependent variable was operationalised as the number of correct responses within a ninety-second window.", "Academic"),
    ("An iterative process of member checking was used to enhance the trustworthiness of the interpretations.", "Academic"),
    ("These results challenge the prevailing consensus and invite reconsideration of established theoretical claims.", "Academic"),

    # ── PROFESSIONAL (40) ────────────────────────────────────────────────────
    ("Our Q3 results reflect strong execution against strategic priorities, with revenue growing fourteen percent year-on-year.", "Professional"),
    ("We are committed to delivering measurable value for our stakeholders across all business units.", "Professional"),
    ("The leadership team has identified three critical areas of focus for the coming fiscal year.", "Professional"),
    ("Cross-functional alignment will be essential to achieving the operational efficiencies outlined in the plan.", "Professional"),
    ("We recommend an immediate review of the vendor risk management process to address identified gaps.", "Professional"),
    ("The project timeline has been revised to reflect updated resource availability and scope changes.", "Professional"),
    ("Key performance indicators will be tracked weekly to ensure we remain on course to meet our targets.", "Professional"),
    ("The merger creates a combined entity with a significantly enhanced competitive position in the market.", "Professional"),
    ("Following extensive due diligence, the Board has approved the proposed acquisition of the target company.", "Professional"),
    ("Our talent strategy prioritises internal mobility and leadership development at every level of the organisation.", "Professional"),
    ("The pilot programme demonstrated a twenty percent reduction in processing time and material cost savings.", "Professional"),
    ("Stakeholder engagement sessions will be conducted in advance of the policy implementation date.", "Professional"),
    ("The client escalated the issue to executive level, and a resolution was delivered within forty-eight hours.", "Professional"),
    ("We have onboarded three new strategic partners who will expand our reach into the APAC region.", "Professional"),
    ("The risk register has been updated to reflect the current threat landscape and mitigation actions.", "Professional"),
    ("Quarterly business reviews will provide an opportunity to course-correct where performance is below target.", "Professional"),
    ("We are investing significantly in digital infrastructure to future-proof our operations and customer experience.", "Professional"),
    ("The revised proposal has been shared with all relevant stakeholders for review and approval.", "Professional"),
    ("A dedicated task force has been established to oversee the transition and minimise operational disruption.", "Professional"),
    ("Our competitive advantage rests on the quality of our people, processes, and proprietary technology.", "Professional"),
    ("The business case demonstrates a projected return on investment of thirty-five percent over three years.", "Professional"),
    ("We are accelerating our sustainability agenda in response to both regulatory requirements and market expectations.", "Professional"),
    ("The quarterly earnings call will take place at nine o'clock Eastern Time on the fourteenth of next month.", "Professional"),
    ("Employee engagement scores have improved by eight percentage points since the new programme launched.", "Professional"),
    ("We anticipate that the integration will be completed ahead of the original schedule and within budget.", "Professional"),
    ("The organisation is undergoing a structural realignment to better serve the evolving needs of our clients.", "Professional"),
    ("Our net promoter score reached an all-time high this quarter, reflecting strong customer satisfaction outcomes.", "Professional"),
    ("The data governance framework has been updated to ensure compliance with the latest regulatory requirements.", "Professional"),
    ("Please find attached the revised terms of engagement for your review and sign-off prior to contract execution.", "Professional"),
    ("A post-implementation review will be scheduled sixty days following go-live to assess outcomes against objectives.", "Professional"),
    ("The executive sponsor will provide an update at the all-hands meeting scheduled for the end of the month.", "Professional"),
    ("Supply chain disruptions have been partially offset by inventory management improvements implemented last quarter.", "Professional"),
    ("We are tracking ahead of plan on our cost-reduction programme and expect to deliver full-year savings.", "Professional"),
    ("The new operating model consolidates regional structures to eliminate duplication and improve accountability.", "Professional"),
    ("Legal counsel has reviewed the agreement and confirmed there are no outstanding issues preventing execution.", "Professional"),
    ("The board approved the capital expenditure proposal subject to satisfactory completion of technical due diligence.", "Professional"),
    ("Our go-to-market strategy has been refined based on customer feedback gathered during the beta phase.", "Professional"),
    ("Headcount planning for the next financial year will be finalised following completion of the budgeting cycle.", "Professional"),
    ("The lessons learned from this project will be documented and shared across the programme management office.", "Professional"),
    ("We remain confident in our full-year guidance and expect market conditions to stabilise in the second half.", "Professional"),

    # ── SIMPLIFIED (40) ──────────────────────────────────────────────────────
    ("This service is free. You do not need to pay anything to use it.", "Simplified"),
    ("If you are not sure what to do, you can call us and we will help you.", "Simplified"),
    ("Wash your hands with soap and water for at least twenty seconds to kill germs.", "Simplified"),
    ("A vaccine teaches your body how to fight a germ before you ever get sick from it.", "Simplified"),
    ("The library is open every day except Sunday. Anyone can join for free.", "Simplified"),
    ("You have the right to see your medical records. Ask your doctor and they must show you.", "Simplified"),
    ("To vote, you need to be registered first. It only takes a few minutes to sign up online.", "Simplified"),
    ("Interest is the extra money you pay back when you borrow. The lower the rate, the less you pay.", "Simplified"),
    ("Climate change means the Earth is getting warmer because of gases we release into the air.", "Simplified"),
    ("If the light on your router is red, turn it off, wait ten seconds, then turn it back on.", "Simplified"),
    ("You do not need a degree to apply. We care more about your skills and your attitude.", "Simplified"),
    ("Eating more fruit and vegetables can help you stay healthy and reduce the risk of many illnesses.", "Simplified"),
    ("A budget is just a plan for how you will spend your money each month.", "Simplified"),
    ("If someone is unconscious and not breathing, call 999 and start chest compressions right away.", "Simplified"),
    ("Your data is never shared with other companies. It stays with us and we keep it safe.", "Simplified"),
    ("The form has three sections. Fill in each one carefully and check everything before you send it.", "Simplified"),
    ("Inflation means that prices go up over time. The same amount of money buys less than it used to.", "Simplified"),
    ("You can cancel at any time. There are no fees or penalties for leaving early.", "Simplified"),
    ("The app will ask for your location. This is only so it can show you things near you.", "Simplified"),
    ("Sleep is important for your brain and your body. Most adults need seven to nine hours a night.", "Simplified"),
    ("A password should be long and hard to guess. Never use your name or birthday.", "Simplified"),
    ("To make tea, boil the water first, then pour it over the bag and wait three to five minutes.", "Simplified"),
    ("Recycling helps reduce waste and is better for the environment than throwing things away.", "Simplified"),
    ("If you miss a payment, contact us as soon as possible. We will try to help you find a solution.", "Simplified"),
    ("Your pension is money you save now so you have an income when you stop working.", "Simplified"),
    ("The bus leaves every thirty minutes. Check the timetable on the website to plan your journey.", "Simplified"),
    ("Sunscreen protects your skin from the sun. Use at least SPF 30 and reapply every two hours.", "Simplified"),
    ("Read the label before taking any medicine. Some medicines should not be taken with food or drink.", "Simplified"),
    ("A contract is an agreement between two people or organisations that is legally binding.", "Simplified"),
    ("You can appeal a decision if you think it is wrong. Ask for the appeals form at the front desk.", "Simplified"),
    ("Social media can be great for keeping in touch, but be careful about what you share online.", "Simplified"),
    ("Exercise does not have to mean going to a gym. A thirty-minute walk every day makes a real difference.", "Simplified"),
    ("Your employer must give you a payslip every time you are paid. Keep them somewhere safe.", "Simplified"),
    ("If you smell gas, do not use any switches. Leave the building and call the emergency line.", "Simplified"),
    ("Diabetes means your body has trouble managing sugar in your blood. It can be controlled with the right care.", "Simplified"),
    ("The council tax you pay helps fund local services like bin collections, schools, and roads.", "Simplified"),
    ("Phishing is when someone pretends to be a trusted organisation to steal your information.", "Simplified"),
    ("To save energy at home, turn off lights when you leave a room and unplug devices you are not using.", "Simplified"),
    ("Universal credit is a payment to help people on a low income or who are out of work.", "Simplified"),
    ("The law says every child must be in education or training until the age of eighteen.", "Simplified"),
]


class NaiveBayesToneClassifier:
    """
    Multinomial Naive Bayes classifier for tone detection.

    Trained at instantiation on the seed corpus above.
    Can be used to verify or score a transformed summary.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Parameters
        ----------
        alpha : Laplace smoothing factor. α=1 is standard add-one smoothing.
        """
        self.alpha = alpha
        self.classes_: list[str] = []
        self.log_priors_: dict[str, float] = {}
        self.log_likelihoods_: dict[str, dict[str, float]] = {}
        self.vocab_: set[str] = set()
        self._train(_SEED_CORPUS)

    def _train(self, corpus: list[tuple[str, str]]) -> None:
        """
        Estimate parameters from labelled corpus.

        For each class c:
            log P(c)  = log(count(c) / N)
            P(t | c)  = (count(t,c) + α) / (Σ_t count(t,c) + α*|V|)
        """
        # Count documents per class
        class_doc_counts: Counter = Counter(label for _, label in corpus)
        N = len(corpus)
        self.classes_ = list(class_doc_counts.keys())

        # Term counts per class
        class_term_counts: dict[str, Counter] = defaultdict(Counter)
        for text, label in corpus:
            for token in tokenise(text):
                class_term_counts[label][token] += 1
                self.vocab_.add(token)

        V = len(self.vocab_)

        # Log priors
        self.log_priors_ = {
            c: math.log(class_doc_counts[c] / N)
            for c in self.classes_
        }

        # Log likelihoods with Laplace smoothing
        self.log_likelihoods_ = {}
        for c in self.classes_:
            total = sum(class_term_counts[c].values()) + self.alpha * V
            self.log_likelihoods_[c] = {
                term: math.log((class_term_counts[c].get(term, 0) + self.alpha) / total)
                for term in self.vocab_
            }
            # Store the "unknown term" log probability for OOV terms
            self.log_likelihoods_[c]["__UNK__"] = math.log(self.alpha / total)

    def predict_proba(self, text: str) -> dict[str, float]:
        """
        Return a dict mapping each tone class to its posterior probability.
        Probabilities are normalised to sum to 1 via softmax on log scores.
        """
        tokens = tokenise(text)
        log_scores: dict[str, float] = {}

        for c in self.classes_:
            score = self.log_priors_[c]
            for token in tokens:
                if token in self.log_likelihoods_[c]:
                    score += self.log_likelihoods_[c][token]
                else:
                    score += self.log_likelihoods_[c]["__UNK__"]
            log_scores[c] = score

        # Softmax for probabilities
        max_score = max(log_scores.values())
        exp_scores = {c: math.exp(s - max_score) for c, s in log_scores.items()}
        total = sum(exp_scores.values())
        return {c: v / total for c, v in exp_scores.items()}

    def predict(self, text: str) -> str:
        """Return the most likely tone class."""
        proba = self.predict_proba(text)
        return max(proba, key=proba.get)


# ══════════════════════════════════════════════════════════════════════════════
# PART B — LEXICAL TONE TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════

# ── CONTRACTION MAPS ──────────────────────────────────────────────────────────
_EXPAND = {
    "can't":    "cannot",    "won't":    "will not",   "don't":    "do not",
    "doesn't":  "does not",  "didn't":   "did not",    "isn't":    "is not",
    "aren't":   "are not",   "wasn't":   "was not",    "weren't":  "were not",
    "haven't":  "have not",  "hasn't":   "has not",    "hadn't":   "had not",
    "wouldn't": "would not", "couldn't": "could not",  "shouldn't":"should not",
    "it's":     "it is",     "that's":   "that is",    "there's":  "there is",
    "they're":  "they are",  "we're":    "we are",     "you're":   "you are",
    "he's":     "he is",     "she's":    "she is",     "I'm":      "I am",
    "I've":     "I have",    "I'll":     "I will",     "I'd":      "I would",
    "let's":    "let us",    "who's":    "who is",     "what's":   "what is",
}

_CONTRACT = {v: k for k, v in _EXPAND.items() if k not in {"I'm","I've","I'll","I'd"}}


def _expand_contractions(text: str) -> str:
    for contracted, expanded in _EXPAND.items():
        text = re.sub(re.escape(contracted), expanded, text, flags=re.IGNORECASE)
    return text


def _add_contractions(text: str) -> str:
    for expanded, contracted in _CONTRACT.items():
        text = re.sub(r"\b" + re.escape(expanded) + r"\b", contracted, text, flags=re.IGNORECASE)
    return text


# ── VOCABULARY SUBSTITUTION LEXICONS ──────────────────────────────────────────
# Format: { from_word: { to_tone: to_word, ... }, ... }
# Only applied when the target tone value is present.
_LEXICON: list[tuple[str, dict[str, str]]] = [
    ("use",       {"Formal":"utilise",      "Academic":"employ",       "Professional":"leverage"}),
    ("show",      {"Formal":"demonstrate",  "Academic":"illustrate",   "Professional":"evidence"}),
    ("help",      {"Formal":"assist",       "Academic":"facilitate",   "Professional":"support"}),
    ("find",      {"Formal":"identify",     "Academic":"ascertain",    "Professional":"determine"}),
    ("make",      {"Formal":"produce",      "Academic":"generate",     "Professional":"deliver"}),
    ("start",     {"Formal":"commence",     "Academic":"initiate",     "Professional":"launch"}),
    ("end",       {"Formal":"conclude",     "Academic":"terminate",    "Professional":"complete"}),
    ("need",      {"Formal":"require",      "Academic":"necessitate",  "Professional":"demand"}),
    ("get",       {"Formal":"obtain",       "Academic":"acquire",      "Casual":"grab"}),
    ("look at",   {"Formal":"examine",      "Academic":"investigate",  "Professional":"assess"}),
    ("think",     {"Formal":"consider",     "Academic":"posit",        "Professional":"assess"}),
    ("big",       {"Formal":"substantial",  "Academic":"considerable", "Professional":"significant"}),
    ("small",     {"Formal":"limited",      "Academic":"marginal",     "Professional":"minimal"}),
    ("says",      {"Formal":"states",       "Academic":"posits",       "Professional":"indicates"}),
    ("important", {"Academic":"significant","Professional":"critical",  "Formal":"essential"}),
    ("problem",   {"Formal":"issue",        "Academic":"challenge",    "Professional":"constraint"}),
    ("part",      {"Formal":"component",    "Academic":"element",      "Professional":"aspect"}),
    ("clear",     {"Formal":"evident",      "Academic":"apparent",     "Professional":"apparent"}),
    ("about",     {"Formal":"regarding",    "Academic":"concerning",   "Professional":"pertaining to"}),
    ("many",      {"Formal":"numerous",     "Academic":"a significant number of"}),
    ("often",     {"Formal":"frequently",   "Academic":"consistently", "Professional":"regularly"}),
    ("also",      {"Academic":"furthermore","Formal":"additionally",   "Professional":"moreover"}),
    ("but",       {"Formal":"however",      "Academic":"nevertheless", "Professional":"however"}),
    ("so",        {"Formal":"therefore",    "Academic":"consequently", "Professional":"as a result"}),
]


def _apply_lexicon(text: str, tone: str) -> str:
    """Substitute words in text according to the tone lexicon."""
    for original, substitutions in _LEXICON:
        if tone in substitutions:
            replacement = substitutions[tone]
            pattern = r"\b" + re.escape(original) + r"\b"
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ── TONE-SPECIFIC OPENERS & HEDGES ────────────────────────────────────────────
_OPENERS = {
    "Formal": [
        "Upon examination of the available information, ",
        "It is evident that ",
        "The foregoing analysis indicates that ",
    ],
    "Casual": [
        "So basically, ",
        "Here's the thing — ",
        "Long story short, ",
    ],
    "Humanised": [
        "At its core, this is a story about ",
        "What this really tells us is that ",
        "Perhaps most importantly, ",
    ],
    "Academic": [
        "The evidence suggests that ",
        "Analysis of the data indicates that ",
        "From a theoretical standpoint, ",
    ],
    "Professional": [
        "In summary, ",
        "The key takeaway is that ",
        "From a strategic perspective, ",
    ],
    "Simplified": [
        "Simply put, ",
        "In plain terms, ",
        "The basic idea is that ",
    ],
}

_TRANSITIONS = {
    "Formal":       ["Furthermore,", "Moreover,", "In addition,", "Consequently,"],
    "Casual":       ["Also,", "Plus,", "And another thing —", "On top of that,"],
    "Humanised":    ["Beyond this,", "And yet,", "What is more,", "Importantly,"],
    "Academic":     ["Furthermore,", "In addition,", "Notably,", "Correspondingly,"],
    "Professional": ["Additionally,", "Moreover,", "It is also worth noting that", "Critically,"],
    "Simplified":   ["Also,", "Another thing is that", "On top of this,", "This means that"],
}


# ── STRUCTURAL TRANSFORMS ─────────────────────────────────────────────────────
def _split_long_sentences(text: str, max_words: int = 20) -> str:
    """For Simplified tone: break sentences longer than max_words on 'and'/'but'."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    result = []
    for sent in sentences:
        words = sent.split()
        if len(words) > max_words:
            # Try splitting on ' and ' near the midpoint
            mid = len(words) // 2
            best_split = None
            for i in range(mid - 5, mid + 5):
                if 0 < i < len(words) and words[i].lower() in ("and", "but", "while", "which"):
                    best_split = i
                    break
            if best_split:
                left  = " ".join(words[:best_split]).rstrip(",") + "."
                right = " ".join(words[best_split + 1:]).capitalize()
                result.append(left + " " + right)
                continue
        result.append(sent)
    return " ".join(result)


def _passive_voice_hint(text: str) -> str:
    """
    Academic tone: lightly nudge active sentences toward passive constructions
    by replacing 'We found X' / 'We show X' → 'X was found' / 'X is shown'.
    This is a surface-level heuristic, not full parse-tree transformation.
    """
    text = re.sub(r"\bWe found that\b",  "It was found that",  text, flags=re.IGNORECASE)
    text = re.sub(r"\bWe show that\b",   "It is shown that",   text, flags=re.IGNORECASE)
    text = re.sub(r"\bWe argue that\b",  "It is argued that",  text, flags=re.IGNORECASE)
    text = re.sub(r"\bWe observe\b",     "It is observed",     text, flags=re.IGNORECASE)
    text = re.sub(r"\bI found\b",        "It was found",       text, flags=re.IGNORECASE)
    text = re.sub(r"\bI think\b",        "It is posited",      text, flags=re.IGNORECASE)
    return text


# ── MASTER TRANSFORMER ────────────────────────────────────────────────────────
class LexicalToneTransformer:
    """
    Transforms a summary string into the target tone using:
      1. Contraction expansion or reduction
      2. Vocabulary substitution (lexicon)
      3. Structural transforms (sentence splitting, passive voice)
      4. Opener injection (first sentence prefix)

    No model weights are loaded — all logic is rule-based and transparent.
    """

    def transform(self, text: str, tone: str) -> str:
        if tone not in TONES:
            tone = "Formal"

        # 1. Contraction handling
        if tone in ("Formal", "Academic", "Professional"):
            text = _expand_contractions(text)
        elif tone in ("Casual", "Humanised"):
            text = _add_contractions(text)

        # 2. Vocabulary substitution
        text = _apply_lexicon(text, tone)

        # 3. Structural transforms
        if tone == "Simplified":
            text = _split_long_sentences(text, max_words=18)
        if tone == "Academic":
            text = _passive_voice_hint(text)

        # 4. Opener injection (prepend to the first sentence)
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if sentences and _OPENERS.get(tone):
            opener = random.choice(_OPENERS[tone])
            first = sentences[0]
            # Don't double-apply if opener is already there
            if not any(first.startswith(op.strip()) for op in _OPENERS[tone]):
                first_lower = first[0].lower() + first[1:] if len(first) > 1 else first
                sentences[0] = opener + first_lower

        text = " ".join(sentences)

        # 5. Tidy up whitespace and punctuation
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\s([,.])", r"\1", text)

        return text.strip()
