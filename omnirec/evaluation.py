"""Offline evaluation metrics for the recommender.

`retrieval_precision_at_k` measures how often the ANN retrieval stage surfaces
items from a user's ground-truth preferred category, compared with a random
baseline — a direct check that the two-tower learned something.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .engine import OmniRec
from .models import User


@dataclass
class EvalResult:
    precision_at_k: float
    random_baseline: float
    lift: float
    users_evaluated: int

    def to_dict(self) -> dict:
        return {"precision_at_k": round(self.precision_at_k, 4),
                "random_baseline": round(self.random_baseline, 4),
                "lift": round(self.lift, 2), "users_evaluated": self.users_evaluated}


def retrieval_precision_at_k(engine: OmniRec, user_pref: dict[str, str],
                             item_category: dict[str, str], k: int = 20,
                             max_users: int = 150) -> EvalResult:
    n_cat = len(set(item_category.values())) or 1
    baseline = 1.0 / n_cat
    hits = 0.0
    evaluated = 0
    for uid, pref in list(user_pref.items())[:max_users]:
        user = engine.users.get(uid) or User(id=uid)
        uvec, _ = engine.funnel.user_vector(user)
        cand = engine.index.search(uvec, k=k)
        if not cand:
            continue
        match = sum(1 for iid, _ in cand if item_category.get(iid) == pref)
        hits += match / len(cand)
        evaluated += 1
    precision = hits / evaluated if evaluated else 0.0
    return EvalResult(precision_at_k=precision, random_baseline=baseline,
                      lift=(precision / baseline) if baseline else 0.0,
                      users_evaluated=evaluated)
