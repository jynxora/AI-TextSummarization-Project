import pytest
from engine.textrank import TextRankSummariser, split_sentences
from engine.tfidf import TFIDFVectoriser, cosine_similarity_matrix
from engine.keypoints import extract_key_points
from engine.tone import LexicalToneTransformer, NaiveBayesToneClassifier, TONES

SAMPLE = """
Artificial intelligence is transforming industries across the globe.
Machine learning models can now perform tasks that once required human expertise.
Natural language processing enables computers to understand and generate human text.
Deep learning has revolutionized image recognition and speech synthesis.
These advances raise important questions about the future of work and society.
The benefits of AI must be weighed against ethical concerns and bias risks.
"""

# --- TextRank ---
def test_split_sentences():
    sents = split_sentences(SAMPLE)
    assert len(sents) >= 4

def test_textrank_summary():
    summariser = TextRankSummariser()
    result = summariser.summarise(SAMPLE, length=2)
    assert "summary" in result
    assert len(result["summary"]) > 0

def test_textrank_length_levels():
    summariser = TextRankSummariser()
    for level in [1, 2, 3, 4, 5]:
        result = summariser.summarise(SAMPLE, length=level)
        assert result["sentence_count"] >= 1

# --- Tone Transformer ---
def test_all_tones():
    transformer = LexicalToneTransformer()
    for tone in TONES:
        output = transformer.transform(SAMPLE, tone)
        assert isinstance(output, str) and len(output) > 0

# --- Naive Bayes Classifier ---
def test_classifier_returns_all_tones():
    clf = NaiveBayesToneClassifier()
    proba = clf.predict_proba(SAMPLE)
    assert set(proba.keys()) == set(TONES)
    assert abs(sum(proba.values()) - 1.0) < 0.01  # probabilities sum to 1

def test_classifier_predict():
    clf = NaiveBayesToneClassifier()
    pred = clf.predict(SAMPLE)
    assert pred in TONES

# --- Key Points ---
def test_key_points():
    kps = extract_key_points(SAMPLE, n=3)
    assert len(kps) <= 3
    assert all(isinstance(k, str) for k in kps)