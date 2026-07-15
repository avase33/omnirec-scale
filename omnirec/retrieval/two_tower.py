"""Two-Tower retrieval model (trained from scratch, pure Python).

A dual encoder that maps users and items into a shared dense space where
dot-product approximates affinity — the retrieval stage of the funnel.

* **Item tower**: a linear projection of the fused multimodal item embedding
  (``W_i · fused + b_i``).
* **User tower**: a linear projection of ``[user-id embedding ⊕ mean of the
  user's history item embeddings]`` (``W_u · [id ⊕ hist] + b_u``). The history
  term is what lets the tower generalise to users and react to new clicks.

Training uses **in-batch sampled-softmax** (a.k.a. sampled cross-entropy): within
a mini-batch of clicked (user, item) pairs, every *other* item in the batch is a
negative, and we maximise the softmax probability of the true item. This is the
standard, scalable objective used by production two-tower retrievers. Gradients
are computed and applied by hand with SGD + weight decay.

The tower shapes and update rules mirror a PyTorch implementation; here they run
with no framework so retrieval trains and serves anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from ..errors import NotTrainedError
from ..linalg import axpy, dot, matvec, rng, softmax, vecmat_t, xavier, zeros
from ..models import Interaction, User


@dataclass
class TrainReport:
    epochs: int
    final_loss: float
    history: list[float] = field(default_factory=list)


class TwoTowerModel:
    def __init__(self, item_dim: int = 64, user_dim: int = 64, id_dim: int = 32, seed: int = 13) -> None:
        self.item_dim = item_dim
        self.user_dim = user_dim
        self.id_dim = id_dim
        self._r = rng(seed)
        self._seed = seed

        # Item tower: user_dim x item_dim
        self.W_i = xavier(user_dim, item_dim, self._r)
        self.b_i = zeros(user_dim)
        # User tower: user_dim x (id_dim + item_dim)
        self.W_u = xavier(user_dim, id_dim + item_dim, self._r)
        self.b_u = zeros(user_dim)
        # Learned per-user id embeddings (id_dim)
        self.user_emb: dict[str, list[float]] = {}

        self._item_emb: dict[str, list[float]] = {}   # fused multimodal embeddings
        self._trained = False

    # ---- helpers --------------------------------------------------------

    def _uid_vec(self, uid: str) -> list[float]:
        v = self.user_emb.get(uid)
        if v is None:
            v = [self._r.gauss(0.0, 0.1) for _ in range(self.id_dim)]
            self.user_emb[uid] = v
        return v

    def _history_mean(self, history: Sequence[str]) -> list[float]:
        vecs = [self._item_emb[i] for i in history if i in self._item_emb]
        if not vecs:
            return zeros(self.item_dim)
        acc = zeros(self.item_dim)
        for v in vecs:
            axpy(acc, 1.0, v)
        return [x / len(vecs) for x in acc]

    def _user_input(self, uid: str, history: Sequence[str]) -> list[float]:
        return self._uid_vec(uid) + self._history_mean(history)

    def item_vector(self, item_id: str) -> list[float]:
        fused = self._item_emb.get(item_id)
        if fused is None:
            raise NotTrainedError(f"unknown item {item_id}")
        out = matvec(self.W_i, fused)
        axpy(out, 1.0, self.b_i)
        return out

    def user_vector(self, user: User) -> list[float]:
        u_in = self._user_input(user.id, user.history)
        out = matvec(self.W_u, u_in)
        axpy(out, 1.0, self.b_u)
        return out

    def score(self, user: User, item_id: str) -> float:
        return dot(self.user_vector(user), self.item_vector(item_id))

    def all_item_vectors(self) -> dict[str, list[float]]:
        return {iid: self.item_vector(iid) for iid in self._item_emb}

    # ---- training -------------------------------------------------------

    def fit(self, interactions: Sequence[Interaction], item_embeddings: dict[str, list[float]],
            users: dict[str, User], epochs: int = 8, lr: float = 0.05, batch_size: int = 64,
            weight_decay: float = 1e-5) -> TrainReport:
        self._item_emb = dict(item_embeddings)
        positives = [(x.user_id, x.item_id) for x in interactions
                     if x.clicked == 1 and x.item_id in self._item_emb]
        if not positives:
            raise NotTrainedError("no positive interactions to train on")

        history: list[float] = []
        for ep in range(epochs):
            self._r.shuffle(positives)
            total_loss, n_batches = 0.0, 0
            for start in range(0, len(positives), batch_size):
                batch = positives[start:start + batch_size]
                loss = self._train_batch(batch, users, lr, weight_decay)
                total_loss += loss
                n_batches += 1
            avg = total_loss / max(1, n_batches)
            history.append(avg)
        self._trained = True
        return TrainReport(epochs=epochs, final_loss=history[-1], history=history)

    def _train_batch(self, batch, users, lr, weight_decay) -> float:
        # Candidate set = the unique items in this batch (in-batch negatives).
        cand_ids = list({iid for _, iid in batch})
        cand_index = {iid: j for j, iid in enumerate(cand_ids)}
        ivecs = [self.item_vector(iid) for iid in cand_ids]

        u_inputs, uvecs, targets = [], [], []
        for uid, pos_iid in batch:
            user = users.get(uid) or User(id=uid)
            u_in = self._user_input(uid, user.history)
            uvec = matvec(self.W_u, u_in)
            axpy(uvec, 1.0, self.b_u)
            u_inputs.append((uid, u_in))
            uvecs.append(uvec)
            targets.append(cand_index[pos_iid])

        # accumulators
        gW_i = [zeros(self.item_dim) for _ in range(self.user_dim)]
        gb_i = zeros(self.user_dim)
        gW_u = [zeros(self.id_dim + self.item_dim) for _ in range(self.user_dim)]
        gb_u = zeros(self.user_dim)
        # gradient wrt each candidate item vector (accumulated across batch rows)
        d_ivec = [zeros(self.user_dim) for _ in cand_ids]
        # gradient wrt each user id embedding
        d_uid: dict[str, list[float]] = {}

        loss = 0.0
        B = len(batch)
        for k in range(B):
            uvec = uvecs[k]
            logits = [dot(uvec, ivecs[j]) for j in range(len(cand_ids))]
            probs = softmax(logits)
            t = targets[k]
            loss += -math.log(max(probs[t], 1e-12))

            # dL/dlogit
            g = probs[:]
            g[t] -= 1.0

            # dL/duvec = sum_j g[j] * ivec_j
            d_uvec = zeros(self.user_dim)
            for j, gj in enumerate(g):
                if gj != 0.0:
                    axpy(d_uvec, gj, ivecs[j])
                    axpy(d_ivec[j], gj, uvec)

            # user tower linear grads
            uid, u_in = u_inputs[k]
            for row in range(self.user_dim):
                axpy(gW_u[row], d_uvec[row], u_in)
            axpy(gb_u, 1.0, d_uvec)
            # backprop into user input -> user id embedding (first id_dim dims)
            d_uin = vecmat_t(d_uvec, self.W_u)   # length id_dim + item_dim
            acc = d_uid.setdefault(uid, zeros(self.id_dim))
            for i in range(self.id_dim):
                acc[i] += d_uin[i]

        # item tower linear grads from d_ivec
        for j, iid in enumerate(cand_ids):
            fused = self._item_emb[iid]
            div = d_ivec[j]
            for row in range(self.user_dim):
                if div[row] != 0.0:
                    axpy(gW_i[row], div[row], fused)
            axpy(gb_i, 1.0, div)

        # SGD update (mean over batch) with weight decay
        scale = lr / B
        for row in range(self.user_dim):
            wi = self.W_i[row]
            gwi = gW_i[row]
            for c in range(self.item_dim):
                wi[c] -= scale * gwi[c] + lr * weight_decay * wi[c]
            wu = self.W_u[row]
            gwu = gW_u[row]
            for c in range(self.id_dim + self.item_dim):
                wu[c] -= scale * gwu[c] + lr * weight_decay * wu[c]
        for r_ in range(self.user_dim):
            self.b_i[r_] -= scale * gb_i[r_]
            self.b_u[r_] -= scale * gb_u[r_]
        for uid, gacc in d_uid.items():
            emb = self.user_emb[uid]
            for i in range(self.id_dim):
                emb[i] -= scale * gacc[i] + lr * weight_decay * emb[i]

        return loss / B
