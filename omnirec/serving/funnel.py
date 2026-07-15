"""The multi-stage recommendation funnel.

Ties the stages together for a single request:

    1. user vector   — from the online feature store if fresh, else the two-tower
    2. retrieval     — ANN index returns the top `retrieval_k` candidates
    3. ranking       — DCN-v2 scores each candidate's CTR
    4. top-K         — return the best `final_k` by predicted CTR

Stages 2–4 are the classic funnel that keeps latency flat as the catalog grows:
ANN does the 10M→500 cut, and the expensive ranker only ever sees `retrieval_k`
items.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from ..models import Item, Recommendation, User


@dataclass
class FunnelResult:
    recommendations: list[Recommendation]
    latency_ms: float
    n_candidates: int
    used_online_state: bool


class RecommendationFunnel:
    def __init__(self, tower, index, ranker, featurizer, feature_store, items: dict[str, Item],
                 retrieval_k: int = 500, final_k: int = 10, item_vector_fn=None) -> None:
        self.tower = tower
        self.index = index
        self.ranker = ranker
        self.featurizer = featurizer
        self.feature_store = feature_store
        self.items = items
        self.retrieval_k = retrieval_k
        self.final_k = final_k
        # cache-backed item-vector lookup (falls back to the tower)
        self._item_vec = item_vector_fn or tower.item_vector

    def user_vector(self, user: User) -> tuple[list[float], bool]:
        state = self.feature_store.get_online(user.id)
        if state is not None and state.vector:
            return state.vector, True
        return self.tower.user_vector(user), False

    def recommend(self, user: User, k: Optional[int] = None,
                  exclude_seen: bool = True) -> FunnelResult:
        t0 = time.perf_counter()
        k = k or self.final_k
        uvec, online = self.user_vector(user)

        # stage 2: ANN retrieval
        candidates = self.index.search(uvec, k=self.retrieval_k)
        seen = set(user.history) if exclude_seen else set()

        # stage 3: DCN ranking
        scored: list[Recommendation] = []
        for item_id, retr_score in candidates:
            if item_id in seen:
                continue
            item = self.items.get(item_id)
            if item is None:
                continue
            ivec = self._item_vec(item_id)
            feats = self.featurizer.build(uvec, ivec, item, retr_score)
            ctr = self.ranker.predict_features(feats)
            scored.append(Recommendation(item_id=item_id, title=item.title,
                                         retrieval_score=retr_score, ctr_score=ctr, rank=0))

        # stage 4: top-K by CTR
        scored.sort(key=lambda r: r.ctr_score, reverse=True)
        top = scored[:k]
        for i, rec in enumerate(top, 1):
            rec.rank = i
        latency = (time.perf_counter() - t0) * 1000.0
        return FunnelResult(recommendations=top, latency_ms=latency,
                            n_candidates=len(candidates), used_online_state=online)
