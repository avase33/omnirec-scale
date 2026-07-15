import random

from omnirec.indexing.ann import HNSWIndex
from omnirec.linalg import normalize


def _rand_vec(r, d):
    return normalize([r.gauss(0, 1) for _ in range(d)])


def test_hnsw_recall_vs_exact():
    r = random.Random(0)
    dim = 32
    idx = HNSWIndex(dim, m=16, ef=64, seed=1)
    vecs = {f"i{i}": _rand_vec(r, dim) for i in range(400)}
    idx.add_many(vecs)

    hits, total = 0, 0
    for _ in range(30):
        q = _rand_vec(r, dim)
        approx = {iid for iid, _ in idx.search(q, k=10)}
        exact = {iid for iid, _ in idx.exact_search(q, k=10)}
        hits += len(approx & exact)
        total += len(exact)
    recall = hits / total
    assert recall >= 0.8, f"recall too low: {recall:.2f}"


def test_hnsw_returns_k():
    r = random.Random(2)
    idx = HNSWIndex(16, seed=2)
    idx.add_many({f"i{i}": _rand_vec(r, 16) for i in range(50)})
    res = idx.search(_rand_vec(r, 16), k=5)
    assert len(res) == 5
    # scores descending
    assert all(res[i][1] >= res[i + 1][1] for i in range(len(res) - 1))
