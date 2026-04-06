"""
app.py
======
SummarisAI — Flask API Server

Wires together the three-stage from-scratch ML pipeline:

    Stage 1 — TextRank (engine/textrank.py)
                Extractive summarisation via graph-based sentence ranking
                and PageRank power iteration.

    Stage 2 — LexicalToneTransformer (engine/tone.py)
                Rule-based tone rewriting: contraction maps, vocabulary
                substitution lexicons, structural transforms.

    Stage 3 — NaiveBayesToneClassifier (engine/tone.py)
                Multinomial Naive Bayes classifier trained at startup
                on a seed corpus. Scores how well the output matches
                the requested tone.

    Stage 4 — KeyPoint Extractor (engine/keypoints.py)
                TF-IDF + positional + length scoring for extractive
                key point selection.

Endpoints
---------
    GET  /health      — liveness check + model info
    GET  /info        — full algorithm description
    POST /summarize   — main summarisation endpoint
    POST /classify    — tone classification only (for debugging/demo)

Run
---
    python app.py
    # -> http://localhost:5000
"""

import logging
import re
import sys
import time

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Bootstrap engine ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

log.info("Initialising SummarisAI ML engine (all models built from scratch)...")
t0 = time.time()

from engine.textrank  import TextRankSummariser
from engine.tone      import LexicalToneTransformer, NaiveBayesToneClassifier, TONES
from engine.keypoints import extract_key_points

summariser  = TextRankSummariser(damping=0.85, min_sim=0.0)
transformer = LexicalToneTransformer()
classifier  = NaiveBayesToneClassifier(alpha=1.0)   # trains on seed corpus here

log.info(f"Engine ready in {time.time() - t0:.2f}s  |  No external models downloaded.")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


# ── Helpers ───────────────────────────────────────────────────────────────────
def clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def compression_pct(in_words: int, out_words: int) -> int:
    if in_words == 0:
        return 0
    return max(0, round((1 - out_words / in_words) * 100))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "engine": "from-scratch (no pretrained models)",
        "algorithms": {
            "summarisation": "TextRank -- PageRank power iteration on TF-IDF cosine similarity graph",
            "tone":          "Lexical transformer + Multinomial Naive Bayes classifier",
            "key_points":    "TF-IDF + positional + length composite scoring",
        },
        "available_tones":  TONES,
        "length_levels":    [1, 2, 3, 4, 5],
        "external_deps":    "none (NumPy only for maths)",
    })


@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "pipeline": [
            {
                "stage": 1,
                "name":  "TextRank Extractive Summariser",
                "file":  "engine/textrank.py",
                "description": (
                    "Segments the document into sentences, vectorises each with TF-IDF, "
                    "builds a fully-connected weighted graph (edge weight = cosine similarity), "
                    "then runs PageRank power iteration (d=0.85) to score each sentence. "
                    "Top-k sentences are returned in document order."
                ),
                "reference": "Mihalcea & Tarau (2004). TextRank: Bringing order into texts. EMNLP.",
            },
            {
                "stage": 2,
                "name":  "Lexical Tone Transformer",
                "file":  "engine/tone.py -- LexicalToneTransformer",
                "description": (
                    "Applies contraction expansion/reduction, a vocabulary substitution "
                    "lexicon (50+ term mappings per tone), structural transforms "
                    "(passive voice nudging for Academic, sentence splitting for Simplified), "
                    "and tone-appropriate opener injection."
                ),
            },
            {
                "stage": 3,
                "name":  "Naive Bayes Tone Classifier (verifier)",
                "file":  "engine/tone.py -- NaiveBayesToneClassifier",
                "description": (
                    "Multinomial Naive Bayes with Laplace smoothing (alpha=1), trained at "
                    "startup on a 42-sentence seed corpus (7 examples x 6 tones). "
                    "Returns posterior probabilities for each tone class. Used to score "
                    "how well the transformed output matches the requested tone."
                ),
                "reference": "McCallum & Nigam (1998). A Comparison of Event Models for Naive Bayes. AAAI-98.",
            },
            {
                "stage": 4,
                "name":  "Key Point Extractor",
                "file":  "engine/keypoints.py",
                "description": (
                    "Scores each sentence by a weighted combination of: "
                    "(a) mean TF-IDF weight (0.60), "
                    "(b) Gaussian positional decay / lead bias (0.25), "
                    "(c) sentence length normalisation (0.15). "
                    "Top-n sentences are returned in document order."
                ),
            },
        ]
    })


@app.route("/summarize", methods=["POST"])
def summarize():
    """
    POST /summarize
    ---------------
    Body (JSON)
        text   : str  -- input document (min 20 words)
        tone   : str  -- one of TONES list (default "Formal")
        length : int  -- 1-5 (default 3)

    Response (JSON)
        summary            : str
        key_points         : list[str]
        tone               : str
        tone_confidence    : float   -- Naive Bayes posterior for requested tone
        tone_probabilities : dict    -- full posterior distribution
        length_level       : int
        input_words        : int
        output_words       : int
        compression_pct    : int
        sentences_selected : int
        sentences_total    : int
        processing_ms      : int
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    text   = clean(data.get("text", ""))
    tone   = data.get("tone", "Formal")
    try:
        length = int(data.get("length", 3))
        if length not in range(1, 6):
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({"error": "'length' must be an integer 1-5."}), 400

    if not text:
        return jsonify({"error": "'text' field is required."}), 400
    if word_count(text) < 20:
        return jsonify({"error": "Text is too short -- please provide at least 20 words."}), 400
    if tone not in TONES:
        tone = "Formal"

    t_start = time.perf_counter()

    try:
        # Stage 1: TextRank summarisation
        tr_result    = summariser.summarise(text, length=length)
        base_summary = tr_result["summary"]

        # Stage 2: Tone transformation
        toned_summary = transformer.transform(base_summary, tone)

        # Stage 3: Naive Bayes tone verification
        tone_proba = classifier.predict_proba(toned_summary)
        tone_conf  = tone_proba.get(tone, 0.0)

        # Stage 4: Key point extraction
        key_points = extract_key_points(text, n=4)

        elapsed_ms = round((time.perf_counter() - t_start) * 1000)
        in_w  = word_count(text)
        out_w = word_count(toned_summary)

        log.info(
            f"OK  tone={tone}  length={length}  "
            f"in={in_w}w  out={out_w}w  "
            f"confidence={tone_conf:.2f}  {elapsed_ms}ms"
        )

        return jsonify({
            "summary":            toned_summary,
            "key_points":         key_points,
            "tone":               tone,
            "tone_confidence":    round(tone_conf, 4),
            "tone_probabilities": {k: round(v, 4) for k, v in tone_proba.items()},
            "length_level":       length,
            "input_words":        in_w,
            "output_words":       out_w,
            "compression_pct":    compression_pct(in_w, out_w),
            "sentences_selected": tr_result["sentence_count"],
            "sentences_total":    tr_result["input_sents"],
            "processing_ms":      elapsed_ms,
        })

    except Exception as exc:
        log.error(f"Pipeline error: {exc}", exc_info=True)
        return jsonify({"error": f"Pipeline failed: {str(exc)}"}), 500


@app.route("/classify", methods=["POST"])
def classify():
    """
    POST /classify
    Classify the tone of any text using the Naive Bayes model.
    Useful for demos and dissertation evaluation sections.
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "'text' field required."}), 400

    text  = clean(data["text"])
    proba = classifier.predict_proba(text)
    pred  = classifier.predict(text)

    return jsonify({
        "predicted_tone":   pred,
        "probabilities":    {k: round(v, 4) for k, v in proba.items()},
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
