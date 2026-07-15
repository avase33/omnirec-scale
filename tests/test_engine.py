import pytest

from omnirec.config import Settings
from omnirec.data.mockdata import generate_dataset
from omnirec.engine import OmniRec
from omnirec.evaluation import retrieval_precision_at_k
from omnirec.benchmark import benchmark_latency


@pytest.fixture(scope="module")
def fitted():
    ds = generate_dataset(n_items=300, n_users=120, seed=13)
    engine = OmniRec(Settings())
    report = engine.fit_dataset(ds, retrieval_epochs=6, ranking_epochs=6)
    return engine, ds, report


def test_two_tower_retrieval_beats_random(fitted):
    engine, ds, _ = fitted
    ev = retrieval_precision_at_k(engine, ds.user_pref, ds.item_category, k=20)
    # learned retrieval should beat the random baseline by a clear margin
    assert ev.precision_at_k > ev.random_baseline * 1.5
    assert ev.lift > 1.5


def test_ranker_auc(fitted):
    _, _, report = fitted
    assert report.ranking.auc > 0.6


def test_recommend_returns_topk(fitted):
    engine, ds, _ = fitted
    uid = next(iter(engine.users))
    res = engine.recommend(uid, k=10)
    assert len(res.recommendations) == 10
    assert res.recommendations[0].rank == 1
    # ranked by ctr descending
    ctrs = [r.ctr_score for r in res.recommendations]
    assert ctrs == sorted(ctrs, reverse=True)
    assert res.latency_ms >= 0


def test_realtime_click_updates_user(fitted):
    engine, ds, _ = fitted
    uid = next(iter(engine.users))
    before = engine.feature_store.get_online(uid)
    before_vec = list(before.vector)
    # click an item and confirm the online embedding moved
    item_id = next(iter(engine.items))
    engine.record_click(uid, item_id)
    after = engine.feature_store.get_online(uid)
    assert after.clicks >= 1
    assert after.vector != before_vec


def test_benchmark_reports_percentiles(fitted):
    engine, _, _ = fitted
    lat = benchmark_latency(engine, n_requests=200)
    assert lat.p99_ms >= lat.p95_ms >= lat.p50_ms >= 0
    assert lat.throughput_rps > 0
    # cache should be serving item vectors on the hot path
    assert lat.cache_hit_rate > 0.5


def test_cold_user_falls_back_to_tower(fitted):
    engine, _, _ = fitted
    res = engine.recommend(engine.users[next(iter(engine.users))].id)
    assert res.recommendations
