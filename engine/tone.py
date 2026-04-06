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

# ── SEED TRAINING CORPUS ──────────────────────────────────────────────────────
# Each entry: (text, tone_label)
# Designed to exhibit strong lexical signals per tone class.
_SEED_CORPUS = [
    # Formal
    ("The committee has determined that the aforementioned policy shall be implemented forthwith.", "Formal"),
    ("It is hereby established that all parties must adhere to the stipulated regulations.", "Formal"),
    ("The proceedings were conducted in accordance with the prescribed legislative framework.", "Formal"),
    ("Pursuant to the agreement, the obligations of each party are delineated herein.", "Formal"),
    ("The official documentation requires the signatures of all authorised signatories.", "Formal"),
    ("The department has issued a formal notice regarding the revised compliance requirements.", "Formal"),
    ("In accordance with established protocol, the matter has been referred to the appropriate authority.", "Formal"),

    # Casual
    ("Hey, so basically what happened is the team totally nailed it last week.", "Casual"),
    ("It's pretty simple really, you just need to give it a go and see what happens.", "Casual"),
    ("Honestly, I think we're overthinking this — let's just get it done.", "Casual"),
    ("So yeah, the whole thing ended up being way easier than we thought.", "Casual"),
    ("Don't worry too much about it, things usually work themselves out.", "Casual"),
    ("It's kind of a big deal but nothing we can't handle together.", "Casual"),
    ("Anyway, the point is we should probably just chill and take it one step at a time.", "Casual"),

    # Humanised
    ("What makes this particularly meaningful is the way ordinary people are affected every day.", "Humanised"),
    ("Behind every statistic there is a real person with a story worth hearing.", "Humanised"),
    ("It can feel overwhelming at times, and that is completely understandable.", "Humanised"),
    ("The human element here cannot be overstated — these are lives we are talking about.", "Humanised"),
    ("Listening to these experiences reminds us why this work genuinely matters.", "Humanised"),
    ("There is something quietly profound about the resilience people show in difficult circumstances.", "Humanised"),
    ("We often forget that every decision we make has a ripple effect on real families.", "Humanised"),

    # Academic
    ("The empirical evidence suggests a statistically significant correlation between the variables.", "Academic"),
    ("This study employs a mixed-methods approach to triangulate the qualitative and quantitative data.", "Academic"),
    ("The theoretical framework draws upon seminal works in the field of cognitive linguistics.", "Academic"),
    ("Subsequent analysis revealed a number of methodological limitations inherent to the dataset.", "Academic"),
    ("The findings are consistent with the existing literature on neural plasticity and behaviour.", "Academic"),
    ("Further longitudinal research is warranted to establish causality rather than mere correlation.", "Academic"),
    ("The epistemological implications of these results merit careful consideration by scholars.", "Academic"),

    # Professional
    ("Our Q3 results exceeded targets by 12%, driven primarily by improved operational efficiency.", "Professional"),
    ("The strategic roadmap outlines three key initiatives to drive growth in the next fiscal year.", "Professional"),
    ("We recommend immediate action to mitigate risk and protect stakeholder value.", "Professional"),
    ("Cross-functional collaboration will be essential to delivering this project on schedule.", "Professional"),
    ("The client has approved the revised proposal and expects delivery by end of quarter.", "Professional"),
    ("Action items from today's meeting have been assigned with clear owners and deadlines.", "Professional"),
    ("Performance metrics indicate that resource allocation should be reviewed in Q4.", "Professional"),

    # Simplified
    ("This means that everyone gets a fair chance to join in.", "Simplified"),
    ("The idea is simple. Do one thing at a time and do it well.", "Simplified"),
    ("You do not need any special skills to get started.", "Simplified"),
    ("Think of it like a recipe — follow the steps and you will be fine.", "Simplified"),
    ("The main point is that clean water is something everyone needs to stay healthy.", "Simplified"),
    ("When the light turns green, you can go. When it turns red, you stop.", "Simplified"),
    ("It helps people find jobs. That is the main thing it does.", "Simplified"),
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
