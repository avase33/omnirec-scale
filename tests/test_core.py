import math

from omnirec.linalg import cosine, matvec, normalize, vecmat_t
from omnirec.multimodal import MultimodalEncoder
from omnirec.config import Settings
from omnirec.data.mockdata import generate_dataset


def test_normalize_unit_length():
    v = normalize([3.0, 4.0])
    assert abs(math.hypot(*v) - 1.0) < 1e-9


def test_vecmat_t_shapes():
    m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]   # 2x3
    out = vecmat_t([1.0, 1.0], m)            # len 3
    assert out == [5.0, 7.0, 9.0]


def test_multimodal_encoder_clusters_by_category():
    ds = generate_dataset(n_items=120, n_users=10, seed=1)
    enc = MultimodalEncoder(Settings())
    embs = enc.encode_items(ds.items)
    # two items of the same category should be more similar on average than cross-category
    same, diff, ns, nd = 0.0, 0.0, 0, 0
    items = ds.items
    for i in range(0, 40, 2):
        a, b = items[i], items[i + 1]
        c = cosine(embs[a.id], embs[b.id])
        if a.category == b.category:
            same += c; ns += 1
        else:
            diff += c; nd += 1
    if ns and nd:
        assert same / ns > diff / nd


def test_encode_query_text_dim():
    enc = MultimodalEncoder(Settings())
    q = enc.encode_query_text("wireless headphones")
    assert len(q) == Settings().item_dim
