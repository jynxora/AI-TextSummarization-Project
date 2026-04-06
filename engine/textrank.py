"""
engine/textrank.py
==================
TextRank extractive summariser — implemented from scratch.

Algorithm (Mihalcea & Tarau, 2004)
-----------------------------------
1. Segment the document into sentences.
2. Represent each sentence as a TF-IDF vector (see tfidf.py).
3. Build an undirected weighted graph:
      nodes  = sentences
      edges  = cosine similarity between every sentence pair
4. Run PageRank (power iteration) on the similarity graph.
5. Rank sentences by their PageRank score.
6. Return the top-k sentences in their original document order.

PageRank formula:
    PR(i) = (1 - d) + d * Σ_j [ sim(i,j) / Σ_k sim(j,k) * PR(j) ]

where d = damping factor (classically 0.85).

Reference: Mihalcea, R. & Tarau, P. (2004). "TextRank: Bringing order into
           texts." EMNLP 2004.
"""

import re
import numpy as np
from engine.tfidf import TFIDFVectoriser, cosine_similarity_matrix


# ── SENTENCE SPLITTER ─────────────────────────────────────────────────────────
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\'])")

def split_sentences(text: str) -> list[str]:
    """
    Split a document into sentences.
    Handles common abbreviations by requiring the next token to start with
    an uppercase letter (or quote), which filters out 'Dr.', 'U.S.' etc.
    """
    text = re.sub(r"\s+", " ", text).strip()
    sentences = _SENT_RE.split(text)
    # Filter fragments that are too short to be meaningful
    return [s.strip() for s in sentences if len(s.split()) >= 5]


# ── PAGERANK (POWER ITERATION) ────────────────────────────────────────────────
def _pagerank(
    similarity_matrix: np.ndarray,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Power-iteration PageRank on a weighted adjacency matrix.

    Parameters
    ----------
    similarity_matrix : (n × n) symmetric matrix of edge weights (cosine sim).
    damping           : probability of following an edge vs. teleporting (0.85).
    max_iter          : maximum number of iterations before forced stop.
    tol               : convergence threshold on L1 norm of score delta.

    Returns
    -------
    scores : (n,) array of PageRank scores, summing to 1.
    """
    n = similarity_matrix.shape[0]

    # Remove self-loops (diagonal = 0)
    np.fill_diagonal(similarity_matrix, 0.0)

    # Row-normalise: each row sums to 1 (stochastic matrix)
    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)   # avoid /0
    transition = similarity_matrix / row_sums

    # Initialise uniform scores
    scores = np.ones(n, dtype=np.float64) / n
    teleport = (1.0 - damping) / n                        # uniform teleport

    for _ in range(max_iter):
        prev = scores.copy()
        # PR(i) = (1-d)/n  +  d * Σ_j [ T[j,i] * PR(j) ]
        scores = teleport + damping * (transition.T @ scores)
        # Normalise to prevent drift
        scores /= scores.sum()

        if np.abs(scores - prev).sum() < tol:
            break

    return scores


# ── TEXTRANK SUMMARISER ───────────────────────────────────────────────────────
class TextRankSummariser:
    """
    Full TextRank pipeline.

    Parameters
    ----------
    damping   : PageRank damping factor d (default 0.85).
    min_sim   : edges below this cosine similarity are pruned (default 0.0).
    """

    # Number of sentences to return per length level
    LENGTH_TO_K = {
        1: 1,    # One sentence
        2: 2,    # Short
        3: 4,    # Medium
        4: 7,    # Detailed
        5: 11,   # Comprehensive
    }

    def __init__(self, damping: float = 0.85, min_sim: float = 0.0):
        self.damping  = damping
        self.min_sim  = min_sim
        self._vec     = TFIDFVectoriser()

    def summarise(self, text: str, length: int = 3) -> dict:
        """
        Summarise `text` and return a result dict.

        Parameters
        ----------
        text   : raw input document.
        length : 1-5 (maps to number of sentences via LENGTH_TO_K).

        Returns
        -------
        {
            "summary"        : str   — summary text,
            "ranked_sents"   : list  — (sentence, score) pairs, best-first,
            "sentence_count" : int   — sentences in summary,
            "input_sents"    : int   — total sentences parsed,
        }
        """
        k = self.LENGTH_TO_K.get(length, 4)

        # ── Step 1: Sentence segmentation ─────────────────────────────────────
        sentences = split_sentences(text)

        if len(sentences) == 0:
            return {"summary": text, "ranked_sents": [], "sentence_count": 1, "input_sents": 1}

        if len(sentences) <= k:
            # Document is already short — return as-is
            return {
                "summary":        " ".join(sentences),
                "ranked_sents":   [(s, 1.0) for s in sentences],
                "sentence_count": len(sentences),
                "input_sents":    len(sentences),
            }

        # ── Step 2: TF-IDF vectorisation ──────────────────────────────────────
        self._vec.fit(sentences)
        matrix = self._vec.transform(sentences)       # (n × V)

        # ── Step 3: Build similarity graph ────────────────────────────────────
        sim_matrix = cosine_similarity_matrix(matrix) # (n × n)

        # Prune weak edges (keeps graph sparse for large docs)
        sim_matrix[sim_matrix < self.min_sim] = 0.0

        # ── Step 4: PageRank ──────────────────────────────────────────────────
        scores = _pagerank(sim_matrix, damping=self.damping)

        # ── Step 5: Rank and select ───────────────────────────────────────────
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_indices = sorted([idx for idx, _ in ranked[:k]])  # restore order

        selected = [sentences[i] for i in top_indices]
        ranked_pairs = [(sentences[i], float(scores[i])) for i in range(len(sentences))]
        ranked_pairs.sort(key=lambda x: x[1], reverse=True)

        return {
            "summary":        " ".join(selected),
            "ranked_sents":   ranked_pairs,
            "sentence_count": len(selected),
            "input_sents":    len(sentences),
        }
