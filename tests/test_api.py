"""API tests for POST /classify and GET /predictions/{id}.

Uses the real ResNet18Classifier backed by the local model artifact, so no
weights are downloaded at test time. Each test module gets its own temp
SQLite database.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from galaxeye.api import create_app
from galaxeye.classifier import ResNet18Classifier
from galaxeye.storage import SQLiteResultRepository


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (90, 120, 60)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client_and_repo(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("data") / "test.db"
    classifier = ResNet18Classifier()
    repository = SQLiteResultRepository(db_path)
    app = create_app(classifier, repository)
    with TestClient(app) as client:
        yield client, repository


@pytest.fixture
def client(client_and_repo):
    return client_and_repo[0]


@pytest.fixture
def repo(client_and_repo):
    return client_and_repo[1]


def test_successful_classification(client):
    resp = client.post(
        "/classify",
        files={"file": ("tile.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    for key in ("id", "tile_id", "prediction", "confidence", "status",
                "created_at", "model_version"):
        assert key in body
    assert body["status"] in ("CONFIDENT", "LOW_CONFIDENCE")
    assert 0.0 <= body["confidence"] <= 1.0


def test_generated_tile_id(client):
    resp = client.post(
        "/classify",
        files={"file": ("tile.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["tile_id"].startswith("tile_")


def test_caller_provided_tile_id(client):
    resp = client.post(
        "/classify",
        files={"file": ("tile.png", _png_bytes(), "image/png")},
        data={"tile_id": "my-custom-tile"},
    )
    assert resp.status_code == 200
    assert resp.json()["tile_id"] == "my-custom-tile"


def test_result_is_persisted(client, repo):
    resp = client.post(
        "/classify",
        files={"file": ("tile.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    stored = repo.get(resp.json()["id"])
    assert stored is not None
    assert stored.id == resp.json()["id"]


def test_get_retrieves_persisted_result(client):
    post = client.post(
        "/classify",
        files={"file": ("tile.png", _png_bytes(), "image/png")},
    )
    result_id = post.json()["id"]
    get = client.get(f"/predictions/{result_id}")
    assert get.status_code == 200
    assert get.json()["id"] == result_id


def test_get_returns_404_for_unknown_id(client):
    resp = client.get("/predictions/does-not-exist")
    assert resp.status_code == 404


def test_invalid_image_returns_400(client):
    resp = client.post(
        "/classify",
        files={"file": ("bad.png", b"not an image", "image/png")},
    )
    assert resp.status_code == 400


def test_missing_file_returns_422(client):
    resp = client.post("/classify")
    assert resp.status_code == 422
