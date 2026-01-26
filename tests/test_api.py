import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    """
    TestClient fixture.
    Startup events (model loading) run once per session.
    """
    with TestClient(app) as c:
        yield c


def test_healthcheck(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_infer_with_valid_image(client):
    img_path = Path(__file__).parent / "assets" / "image.JPG"
    assert img_path.exists(), "Test image missing"

    with open(img_path, "rb") as f:
        resp = client.post(
            "/infer",
            files={"file": ("image.JPG", f, "image/jpeg")},
        )

    assert resp.status_code == 200
    data = resp.json()

    assert "final_ok" in data
    assert "per_concept" in data
    assert len(data["per_concept"]) == 4

    for concept, result in data["per_concept"].items():
        assert "p_ok" in result
        assert 0.0 <= result["p_ok"] <= 1.0
        assert "threshold" in result
        assert "pred_ok" in result


def test_infer_without_file(client):
    resp = client.post("/infer")
    assert resp.status_code == 422  # FastAPI validation error


def test_infer_with_empty_file(client):
    empty = io.BytesIO(b"")

    resp = client.post(
        "/infer",
        files={"file": ("empty.jpg", empty, "image/jpeg")},
    )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Empty file"


def test_infer_with_unsupported_content_type(client):
    fake_text = io.BytesIO(b"this is not an image")

    resp = client.post(
        "/infer",
        files={"file": ("test.txt", fake_text, "text/plain")},
    )

    assert resp.status_code == 415
    assert resp.json()["detail"] == "Unsupported image type"


def test_infer_with_corrupted_image_bytes(client):
    corrupted = io.BytesIO(b"\x00\x01\x02\x03\x04")

    resp = client.post(
        "/infer",
        files={"file": ("corrupt.jpg", corrupted, "image/jpeg")},
    )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Could not decode image"
