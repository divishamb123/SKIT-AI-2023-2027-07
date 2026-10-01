"""
test_predict.py — Unit tests for text-service /detect endpoint

Verifies the interface contract against the TextDetectionResponse schema
for both human-written and AI-generated samples.
"""

from pathlib import Path
import sys

# Ensure services/text-service is importable
TEXT_SERVICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TEXT_SERVICE_DIR))

# Clear any cached 'app' modules from other services
for mod in list(sys.modules.keys()):
    if mod == "app" or mod.startswith("app."):
        del sys.modules[mod]

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.schemas import TextDetectionResponse

# Clearly human-written sample: organic phrasing, personal experience, narrative flow
HUMAN_TEXT_SAMPLE = (
    "Yesterday afternoon, I went down to the local farmers market to pick up some fresh Honeycrisp apples, "
    "a loaf of sourdough cinnamon bread, and raw wildflower honey. The weather was unusually crisp for late September, "
    "and the vendor mentioned that this year's harvest was one of the best their family orchard had seen in a decade."
)

# Clearly AI-generated sample: formal, generic academic tone, standard LLM transition markers
AI_TEXT_SAMPLE = (
    "In conclusion, artificial intelligence continues to transform modern society across numerous sectors and domains. "
    "Furthermore, it is crucial to recognize that large language models utilize probabilistic token prediction mechanisms "
    "to generate coherent, contextually aligned output. Consequently, comprehensive verification frameworks and robust "
    "evaluative standards must be systematically deployed to uphold digital authenticity and safeguard against automated deception."
)


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_predict_human_sample_contract(client: TestClient):
    """Hits /detect with a clearly human-written sample and asserts response matches TextDetectionResponse schema."""
    response = client.post("/detect", json={"text": HUMAN_TEXT_SAMPLE})
    assert response.status_code == 200

    data = response.json()

    # Validate dictionary fields and types
    assert "label" in data
    assert data["label"] in ["ai", "human"]
    assert "confidence" in data
    assert isinstance(data["confidence"], (float, int))
    assert 0.0 <= data["confidence"] <= 1.0
    assert "model_version" in data
    assert isinstance(data["model_version"], str)
    assert len(data["model_version"]) > 0

    # Validate against Pydantic schema contract
    parsed = TextDetectionResponse(**data)
    assert parsed.label in ("ai", "human")
    assert 0.0 <= parsed.confidence <= 1.0
    assert parsed.model_version == data["model_version"]


def test_predict_ai_sample_contract(client: TestClient):
    """Hits /detect with a clearly AI-generated sample and asserts response matches TextDetectionResponse schema."""
    response = client.post("/detect", json={"text": AI_TEXT_SAMPLE})
    assert response.status_code == 200

    data = response.json()

    # Validate dictionary fields and types
    assert "label" in data
    assert data["label"] in ["ai", "human"]
    assert "confidence" in data
    assert isinstance(data["confidence"], (float, int))
    assert 0.0 <= data["confidence"] <= 1.0
    assert "model_version" in data
    assert isinstance(data["model_version"], str)
    assert len(data["model_version"]) > 0

    # Validate against Pydantic schema contract
    parsed = TextDetectionResponse(**data)
    assert parsed.label in ("ai", "human")
    assert 0.0 <= parsed.confidence <= 1.0
    assert parsed.model_version == data["model_version"]


def test_predict_empty_text_validation(client: TestClient):
    """Verify that an empty or whitespace-only text request is rejected with 400 Bad Request."""
    response = client.post("/detect", json={"text": "   "})
    assert response.status_code == 400
    assert "Text cannot be empty" in response.json().get("detail", "")


def test_health_endpoints(client: TestClient):
    """Verify that both /health and /internal/health endpoints respond successfully."""
    for path in ["/health", "/internal/health"]:
        response = client.get(path)
        assert response.status_code == 200
        assert response.json().get("status") == "ok"
