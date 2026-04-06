"""
engine/keypoints.py
====================
Extractive key-point extraction using a combined scoring function:

    score(s) = w_tfidf * tfidf_score(s)
             + w_pos   * position_score(s)
             + w_len   * length_score(s)

Where:
    tfidf_score  : average TF-IDF weight of non-stopword tokens in sentence s,
                   relative to the full document corpus.
    position_score: Gaussian-shaped decay rewarding sentences near the start
                    (lead bias) — common in news and report summarisation.
    length_score : mild penalty for very short sentences (< 8 words) and
                   very long ones (> 40 words).

Returns top-n sentences in their original document order.
"""

import math
import numpy as np
from engine.tfidf import TFIDFVectoriser, tokenise
from engine.textrank import split_sentences


def _position_score(idx: int, total: int) -> float:
    """
    Gaussian centred at position 0 (first sentence).
    Sentences early in the document get a higher position score.

        pos_score(i) = exp( -0.5 * (i / (total * 0.4))^2 )

    This gives ~1.0 at the start and decays toward 0 for later sentences.
    """
    if total <= 1:
        return 1.0
    sigma = total * 0.4
    return math.exp(-0.5 * ((idx / sigma) ** 2))


def _length_score(sentence: str) -> float:
    """
    Soft penalty for overly short or overly long sentences.
    Ideal range: 10–30 words → score 1.0.
    """
    n = len(sentence.split())
    if n < 5:
        return 0.2
    if n < 10:
        return 0.6 + 0.04 * (n - 5)     # ramp 0.6 → 0.8
    if n <= 30:
        return 1.0
    if n <= 45:
        return 1.0 - 0.02 * (n - 30)    # decay 1.0 → 0.7
    return 0.7


def extract_key_points(
    text: str,
    n: int = 4,
    w_tfidf: float = 0.6,
    w_pos: float = 0.25,
    w_len: float = 0.15,
) -> list[str]:
    """
    Extract the top-n most informative sentences from `text`.

    Parameters
    ----------
    text    : full input document.
    n       : number of key-point sentences to return.
    w_tfidf : weight for TF-IDF importance score.
    w_pos   : weight for positional (lead-bias) score.
    w_len   : weight for sentence-length score.

    Returns
    -------
    List of key-point sentences in document order.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) <= n:
        return sentences

    total = len(sentences)

    # ── TF-IDF scoring ────────────────────────────────────────────────────────
    vec = TFIDFVectoriser()
    vec.fit(sentences)
    matrix = vec.transform(sentences)   # (n_sents × V)

    # Per-sentence TF-IDF score = mean of non-zero elements in its row
    tfidf_scores = np.zeros(total)
    for i, row in enumerate(matrix):
        nonzero = row[row > 0]
        tfidf_scores[i] = float(nonzero.mean()) if len(nonzero) else 0.0

    # Normalise to [0, 1]
    t_max = tfidf_scores.max()
    if t_max > 0:
        tfidf_scores /= t_max

    # ── Combined score ────────────────────────────────────────────────────────
    combined = np.zeros(total)
    for i, sent in enumerate(sentences):
        combined[i] = (
            w_tfidf * tfidf_scores[i]
            + w_pos  * _position_score(i, total)
            + w_len  * _length_score(sent)
        )

    # Select top-n indices, then sort back to document order
    top_indices = np.argsort(combined)[::-1][:n]
    top_indices = sorted(top_indices.tolist())

    return [sentences[i] for i in top_indices]
