"""Deep & Cross Network v2 (DCN-v2) — CTR ranker, trained from scratch.

DCN-v2 learns explicit bounded-degree feature crosses cheaply. Two parallel
branches consume the same input ``x0``:

* **Cross network** — each layer computes
  ``x_{l+1} = x0 ⊙ (W_l · x_l + b_l) + x_l`` with a full weight matrix ``W_l``
  (the "v2" upgrade over DCN-v1's vector). This models multiplicative feature
  interactions of increasing order without a combinatorial blow-up.
* **Deep network** — a ReLU MLP capturing implicit interactions.

Their outputs are concatenated and passed through a logit + sigmoid to predict
click-through rate. Trained with mini-batch SGD on binary cross-entropy; all
forward/backward passes are hand-derived (no autograd framework).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from ..errors import NotTrainedError
from ..linalg import axpy, dot, matvec, relu, relu_grad, rng, sigmoid, vecmat_t, xavier, zeros


def _outer_accum(acc: list[list[float]], a: Sequence[float], b: Sequence[float]) -> None:
    for i, ai in enumerate(a):
        if ai != 0.0:
            row = acc[i]
            for j, bj in enumerate(b):
                row[j] += ai * bj


@dataclass
class RankReport:
    epochs: int
    final_logloss: float
    auc: float
    history: list[float] = field(default_factory=list)


class DCNv2Ranker:
    def __init__(self, in_dim: int, cross_layers: int = 2, deep_dims: Sequence[int] = (32, 16),
                 seed: int = 5) -> None:
        r = rng(seed)
        self.in_dim = in_dim
        # cross layers
        self.cW = [xavier(in_dim, in_dim, r) for _ in range(cross_layers)]
        self.cb = [zeros(in_dim) for _ in range(cross_layers)]
        # deep layers
        self.dW: list[list[list[float]]] = []
        self.db: list[list[float]] = []
        prev = in_dim
        for h in deep_dims:
            self.dW.append(xavier(h, prev, r))
            self.db.append(zeros(h))
            prev = h
        self.deep_out_dim = prev
        # final combiner
        self.out_dim = in_dim + self.deep_out_dim
        self.w = [r.uniform(-0.1, 0.1) for _ in range(self.out_dim)]
        self.b0 = 0.0
        self._trained = False

    # ---- forward --------------------------------------------------------

    def _forward(self, x0: list[float]):
        # cross branch, caching pre-activations and layer inputs
        cross_inputs = [x0]
        cross_pre = []
        x = x0
        for W, b in zip(self.cW, self.cb):
            pre = matvec(W, x)
            axpy(pre, 1.0, b)
            nxt = [x0[i] * pre[i] + x[i] for i in range(self.in_dim)]
            cross_pre.append(pre)
            cross_inputs.append(nxt)
            x = nxt
        cross_out = x

        # deep branch
        deep_inputs = [x0]
        deep_pre = []
        h = x0
        for W, b in zip(self.dW, self.db):
            pre = matvec(W, h)
            axpy(pre, 1.0, b)
            deep_pre.append(pre)
            h = relu(pre)
            deep_inputs.append(h)
        deep_out = h

        combined = cross_out + deep_out
        logit = dot(self.w, combined) + self.b0
        prob = sigmoid(logit)
        cache = (x0, cross_inputs, cross_pre, cross_out, deep_inputs, deep_pre, deep_out, combined, prob)
        return prob, cache

    def predict_features(self, x0: Sequence[float]) -> float:
        if not self._trained:
            raise NotTrainedError("ranker not trained")
        return self._forward(list(x0))[0]

    # ---- backward + update ---------------------------------------------

    def _backward(self, cache, y: float, lr: float, l2: float) -> float:
        (x0, cross_inputs, cross_pre, cross_out, deep_inputs, deep_pre, deep_out, combined, prob) = cache
        loss = -(y * math.log(max(prob, 1e-12)) + (1 - y) * math.log(max(1 - prob, 1e-12)))
        d_logit = prob - y

        # final combiner grads
        for i in range(self.out_dim):
            self.w[i] -= lr * (d_logit * combined[i] + l2 * self.w[i])
        self.b0 -= lr * d_logit
        d_combined = [d_logit * wi for wi in self.w]
        d_cross_out = d_combined[: self.in_dim]
        d_deep_out = d_combined[self.in_dim:]

        # ---- deep branch backward ----
        d_h = d_deep_out
        for li in reversed(range(len(self.dW))):
            pre = deep_pre[li]
            g = relu_grad(pre)
            d_pre = [d_h[i] * g[i] for i in range(len(pre))]
            inp = deep_inputs[li]
            W = self.dW[li]
            # grads
            for i in range(len(W)):
                dpi = d_pre[i]
                if dpi != 0.0:
                    row = W[i]
                    for j in range(len(inp)):
                        row[j] -= lr * (dpi * inp[j] + l2 * row[j])
                self.db[li][i] -= lr * d_pre[i]
            d_h = vecmat_t(d_pre, W)   # grad wrt layer input

        # ---- cross branch backward ----
        d_x = d_cross_out
        for li in reversed(range(len(self.cW))):
            pre = cross_pre[li]
            x_in = cross_inputs[li]
            W = self.cW[li]
            # x_{l+1} = x0 ⊙ pre + x_in
            d_pre = [d_x[i] * x0[i] for i in range(self.in_dim)]     # through x0⊙pre
            # accumulate W, b grads:  pre = W x_in + b
            for i in range(self.in_dim):
                dpi = d_pre[i]
                if dpi != 0.0:
                    row = W[i]
                    for j in range(self.in_dim):
                        row[j] -= lr * (dpi * x_in[j] + l2 * row[j])
                self.cb[li][i] -= lr * d_pre[i]
            # grad wrt x_in = W^T d_pre + d_x (from the +x_in skip); x0 path ignored (input not trained)
            d_x = [a + b for a, b in zip(vecmat_t(d_pre, W), d_x)]
        return loss

    # ---- training -------------------------------------------------------

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int], epochs: int = 8,
            lr: float = 0.05, l2: float = 1e-6) -> RankReport:
        data = [(list(xi), float(yi)) for xi, yi in zip(X, y)]
        if not data:
            raise NotTrainedError("no ranking data")
        r = rng(99)
        history: list[float] = []
        for _ep in range(epochs):
            r.shuffle(data)
            total = 0.0
            for xi, yi in data:
                _, cache = self._forward(xi)
                total += self._backward(cache, yi, lr, l2)
            history.append(total / len(data))
        self._trained = True
        auc = self.auc(X, y)
        return RankReport(epochs=epochs, final_logloss=history[-1], auc=auc, history=history)

    # ---- metrics --------------------------------------------------------

    def auc(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> float:
        scores = [self._forward(list(xi))[0] for xi in X]
        pos = [s for s, yi in zip(scores, y) if yi == 1]
        neg = [s for s, yi in zip(scores, y) if yi == 0]
        if not pos or not neg:
            return 0.5
        # rank-based (Mann-Whitney U) AUC
        paired = sorted(zip(scores, y), key=lambda p: p[0])
        rank_sum = 0.0
        i = 0
        n = len(paired)
        while i < n:
            j = i
            while j < n and paired[j][0] == paired[i][0]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                if paired[k][1] == 1:
                    rank_sum += avg_rank
            i = j
        n_pos, n_neg = len(pos), len(neg)
        return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
