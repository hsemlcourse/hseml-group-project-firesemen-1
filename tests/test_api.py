from app.main import app
from fastapi.testclient import TestClient


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info() -> None:
    client = TestClient(app)
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"]
    assert body["feature_count"] > 0
