import pytest
from app import create_app

SAMPLE = dict(Cells=1, QValue=1, Fat=1, Protein=1, Lactose=1, Solids=1,
              FFA=1, Citrate=1, FrzPoint=-0.5, SNF=1, MUN=1, Casein=1)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "t.db"
    app = create_app({"TESTING": True, "ENABLE_HISTORY": True, "DB_PATH": str(db)})
    return app.test_client()


def test_predict(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    assert "disclaimer" in r.get_json()


def test_missing_feature(client):
    bad = {k: v for k, v in SAMPLE.items() if k != "Fat"}
    assert client.post("/predict", json=bad).status_code == 422


def test_extra_feature(client):
    assert client.post("/predict", json={**SAMPLE, "Extra": 1}).status_code == 422


def test_non_numeric(client):
    assert client.post("/predict", json={**SAMPLE, "Fat": "abc"}).status_code == 422


def test_batch_and_history(client):
    r = client.post("/predict/batch", json={"samples": [SAMPLE, SAMPLE]})
    assert r.status_code == 200 and len(r.get_json()["predictions"]) == 2
    assert len(client.get("/history").get_json()) >= 2
