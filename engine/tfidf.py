"""
engine/tfidf.py
================
TF-IDF (Term Frequency – Inverse Document Frequency) vectoriser
implemented entirely from scratch using NumPy.

Theory
------
Given a corpus of D documents (here: sentences) and a vocabulary V:

    TF(t, d)  = count(t in d) / |d|
    IDF(t)    = log( (D + 1) / (df(t) + 1) ) + 1     ← smoothed
    TF-IDF(t, d) = TF(t, d) * IDF(t)

Vectors are L2-normalised so cosine similarity reduces to a dot product.

Reference: Salton & Buckley (1988) "Term-weighting approaches in automatic
           text retrieval", Information Processing & Management 24(5).
"""

import re
import math
import numpy as np
from collections import Counter


# ── STOPWORDS ─────────────────────────────────────────────────────────────────
STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","aren't","as","at","be","because","been","before","being","below",
    "between","both","but","by","can't","cannot","could","couldn't","did",
    "didn't","do","does","doesn't","doing","don't","down","during","each",
    "few","for","from","further","get","got","had","hadn't","has","hasn't",
    "have","haven't","having","he","he'd","he'll","he's","her","here","here's",
    "hers","herself","him","himself","his","how","how's","i","i'd","i'll",
    "i'm","i've","if","in","into","is","isn't","it","it's","its","itself",
    "let's","me","more","most","mustn't","my","myself","no","nor","not","of",
    "off","on","once","only","or","other","ought","our","ours","ourselves",
    "out","over","own","same","shan't","she","she'd","she'll","she's","should",
    "shouldn't","so","some","such","than","that","that's","the","their",
    "theirs","them","themselves","then","there","there's","these","they",
    "they'd","they'll","they're","they've","this","those","through","to","too",
    "under","until","up","very","was","wasn't","we","we'd","we'll","we're",
    "we've","were","weren't","what","what's","when","when's","where","where's",
    "which","while","who","who's","whom","why","why's","will","with","won't",
    "would","wouldn't","you","you'd","you'll","you're","you've","your","yours",
    "yourself","yourselves",
}


# ── PORTER-LITE STEMMER ───────────────────────────────────────────────────────
def _stem(word: str) -> str:
    """
    A lightweight suffix-stripping stemmer (subset of Porter's rules).
    Keeps implementation transparent and free of external dependencies.
    """
    if len(word) <= 3:
        return word
    for suffix, replacement in [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("ising", "ise"),
        ("izing", "ize"), ("nesses", ""), ("ational", "ate"),
        ("fulness", "ful"), ("ousness", "ous"), ("alism", "al"),
        ("ation", "ate"), ("ness", ""), ("ment", ""), ("ings", ""),
        ("ing", ""), ("tion", "te"), ("ies", "y"), ("ess", ""),
        ("ers", "er"), ("ated", "ate"), ("edly", ""), ("ely", ""),
        ("ed", ""), ("ly", ""), ("er", ""), ("al", ""),
        ("ic", ""), ("ful", ""), ("ous", ""), ("ive", ""),
    ]:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: len(word) - len(suffix)] + replacement
    return word


def tokenise(text: str) -> list[str]:
    """
    Lowercase → strip non-alpha → remove stopwords → stem.
    Returns a list of normalised tokens.
    """
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    return [_stem(t) for t in tokens if t not in STOPWORDS and len(t) > 2]


# ── TFIDF VECTORISER ──────────────────────────────────────────────────────────
class TFIDFVectoriser:
    """
    Fit on a corpus of strings (sentences), then transform any string
    into a dense TF-IDF vector in R^|V|.

    Usage
    -----
        vec = TFIDFVectoriser()
        vec.fit(sentences)
        matrix = vec.transform(sentences)   # shape (n, |V|)
        v      = vec.transform_one(text)    # shape (|V|,)
    """

    def __init__(self):
        self.vocab_: dict[str, int] = {}   # term → column index
        self.idf_: np.ndarray = np.array([])  # IDF weight per term
        self._fitted = False

    # ── FIT ───────────────────────────────────────────────────────────────────
    def fit(self, corpus: list[str]) -> "TFIDFVectoriser":
        """
        Build vocabulary and compute IDF weights from the corpus.

        IDF formula (smooth):
            idf(t) = log( (N + 1) / (df(t) + 1) ) + 1
        where N = number of documents and df(t) = documents containing t.
        """
        N = len(corpus)
        df: Counter = Counter()
        tokenised_corpus = []

        for doc in corpus:
            tokens = tokenise(doc)
            tokenised_corpus.append(tokens)
            for term in set(tokens):       # set → count each term once per doc
                df[term] += 1

        # Build sorted vocabulary (deterministic column ordering)
        self.vocab_ = {term: idx for idx, term in enumerate(sorted(df.keys()))}
        V = len(self.vocab_)

        # Compute IDF vector
        self.idf_ = np.zeros(V, dtype=np.float64)
        for term, idx in self.vocab_.items():
            self.idf_[idx] = math.log((N + 1) / (df[term] + 1)) + 1.0

        self._fitted = True
        return self

    # ── TRANSFORM ─────────────────────────────────────────────────────────────
    def transform(self, corpus: list[str]) -> np.ndarray:
        """Return an (n × V) TF-IDF matrix, L2-normalised row-wise."""
        assert self._fitted, "Call fit() first."
        matrix = np.zeros((len(corpus), len(self.vocab_)), dtype=np.float64)
        for row, doc in enumerate(corpus):
            matrix[row] = self._tfidf_vector(doc)
        return matrix

    def transform_one(self, text: str) -> np.ndarray:
        """Return a single L2-normalised TF-IDF vector of shape (V,)."""
        assert self._fitted, "Call fit() first."
        return self._tfidf_vector(text)

    # ── INTERNAL ──────────────────────────────────────────────────────────────
    def _tfidf_vector(self, text: str) -> np.ndarray:
        tokens = tokenise(text)
        if not tokens:
            return np.zeros(len(self.vocab_), dtype=np.float64)

        tf = Counter(tokens)
        n_tokens = len(tokens)
        vec = np.zeros(len(self.vocab_), dtype=np.float64)

        for term, count in tf.items():
            if term in self.vocab_:
                idx = self.vocab_[term]
                vec[idx] = (count / n_tokens) * self.idf_[idx]

        # L2 normalisation
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


# ── COSINE SIMILARITY ─────────────────────────────────────────────────────────
def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity for an (n × V) L2-normalised matrix.
    Because rows are already unit vectors, cosine(i, j) = dot(row_i, row_j).
    Returns an (n × n) symmetric similarity matrix.
    """
    # Renormalise defensively (float precision drift)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = matrix / norms
    return normed @ normed.T   # (n × n)
