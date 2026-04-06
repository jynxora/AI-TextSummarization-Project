import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

# --- /health ---
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert "available_tones" in data

# --- /info ---
def test_info(client):
    r = client.get("/info")
    assert r.status_code == 200
    assert len(r.get_json()["pipeline"]) == 4

# --- /summarize happy path ---
def test_summarize_basic(client):
    r = client.post("/summarize", json={
        "text": "AI is changing the world rapidly. " * 10,
        "tone": "Formal",
        "length": 3
    })
    assert r.status_code == 200
    data = r.get_json()
    assert "summary" in data
    assert "key_points" in data
    assert "tone_confidence" in data

# --- /summarize edge cases ---
def test_summarize_too_short(client):
    r = client.post("/summarize", json={"text": "Too short."})
    assert r.status_code == 400

def test_summarize_missing_text(client):
    r = client.post("/summarize", json={"tone": "Formal"})
    assert r.status_code == 400

def test_summarize_invalid_length(client):
    r = client.post("/summarize", json={
        "text": "AI is changing the world rapidly. " * 10,
        "length": 99
    })
    assert r.status_code == 400

def test_summarize_all_tones(client):
    from engine.tone import TONES
    for tone in TONES:
        r = client.post("/summarize", json={
            "text": "AI is changing the world rapidly. " * 10,
            "tone": tone, "length": 2
        })
        assert r.status_code == 200

# --- /classify ---
def test_classify(client):
    r = client.post("/classify", json={"text": "This research demonstrates significant findings."})
    assert r.status_code == 200
    data = r.get_json()
    assert "predicted_tone" in data
    assert "probabilities" in data