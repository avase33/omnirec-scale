import pytest

from omnirec.cli import main
from omnirec.data.mockdata import generate_dataset
from omnirec.pipeline.streaming import EventQueue, StreamProcessor
from omnirec.feature_store.store import FeatureStore


def test_cli_demo_runs(capsys):
    rc = main(["--items", "250", "--users", "80", "--epochs", "3", "demo", "--requests", "100"])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "recommendations" in out
    assert "latency" in out


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_mockdata_deterministic():
    a = generate_dataset(n_items=100, n_users=30, seed=5)
    b = generate_dataset(n_items=100, n_users=30, seed=5)
    assert [i.title for i in a.items] == [i.title for i in b.items]
    assert len(a.interactions) == len(b.interactions)


def test_stream_processor_applies_clicks():
    store = FeatureStore()
    vecs = {"item_1": [1.0, 0.0], "item_2": [0.0, 1.0]}
    proc = StreamProcessor(store, item_vector_fn=lambda i: vecs.get(i), click_lr=0.5)
    q = EventQueue()
    q.publish({"user_id": "u1", "item_id": "item_1", "clicked": 1})
    q.publish({"user_id": "u1", "item_id": "item_2", "clicked": 0})   # not a click -> skipped
    proc.run(q)
    assert proc.stats.clicks_applied == 1
    assert proc.stats.skipped == 1
    assert store.get_online("u1") is not None


def test_api_endpoints():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from omnirec.serving.api import create_app

    client = TestClient(create_app(items=250, users=80, seed=7))
    assert client.get("/healthz").json()["status"] == "ok"

    r = client.get("/recommend/user_1?k=5")
    assert r.status_code == 200
    body = r.json()
    assert len(body["recommendations"]) == 5

    assert client.post("/click", json={"user_id": "user_1", "item_id": "item_1"}).json()["applied"]

    metrics = client.get("/metrics").text
    assert "omnirec_requests_total" in metrics
    assert client.get("/recommend/does_not_exist").status_code in (200, 404)
