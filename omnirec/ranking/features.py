"""Feature engineering for the ranking stage.

Builds a compact dense feature vector for each (user, item) candidate:

    [ proj(user_vec) ⊕ proj(item_vec) ⊕ (proj(user)·proj(item) elementwise)
      ⊕ retrieval_score ⊕ price_norm ]

The two towers' dense vectors are compressed with fixed random projections so the
cross network stays small and fast, while the strongest signal (the retrieval
dot-product) is passed through directly. Deterministic given the seed.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..linalg import dot, hadamard, matvec, rng, xavier
from ..models import Item


class RankingFeaturizer:
    def __init__(self, tower_dim: int = 32, proj_dim: int = 8, seed: int = 21) -> None:
        r = rng(seed)
        self.proj_dim = proj_dim
        self._Pu = xavier(proj_dim, tower_dim, r)
        self._Pi = xavier(proj_dim, tower_dim, r)
        self.dim = proj_dim * 3 + 2

    def build(self, user_vec: Sequence[float], item_vec: Sequence[float],
              item: Item, retrieval_score: float) -> list[float]:
        ur = matvec(self._Pu, user_vec)
        ir = matvec(self._Pi, item_vec)
        prod = hadamard(ur, ir)
        price_norm = math.log1p(max(0.0, item.price)) / 10.0
        return ur + ir + prod + [retrieval_score, price_norm]
