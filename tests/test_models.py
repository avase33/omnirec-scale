import random

from omnirec.ranking.dcn import DCNv2Ranker


def test_dcn_learns_separable_signal():
    r = random.Random(0)
    X, y = [], []
    for _ in range(600):
        x = [r.gauss(0, 1) for _ in range(6)]
        # label depends on a nonlinear cross of features 0 and 1
        logit = 2.5 * x[0] * x[1] + 1.5 * x[2] - 1.0
        p = 1.0 / (1.0 + pow(2.718281828, -logit))
        y.append(1 if r.random() < p else 0)
        X.append(x)

    ranker = DCNv2Ranker(in_dim=6, cross_layers=2, deep_dims=(16, 8), seed=1)
    report = ranker.fit(X, y, epochs=25, lr=0.05)
    assert report.auc > 0.75, f"DCN failed to learn: AUC {report.auc:.3f}"
    # loss decreased over training
    assert report.history[-1] < report.history[0]


def test_dcn_predict_range():
    r = random.Random(1)
    X = [[r.gauss(0, 1) for _ in range(5)] for _ in range(100)]
    y = [r.randint(0, 1) for _ in range(100)]
    ranker = DCNv2Ranker(in_dim=5, seed=2)
    ranker.fit(X, y, epochs=3)
    p = ranker.predict_features(X[0])
    assert 0.0 <= p <= 1.0
