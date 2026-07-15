"""OmniRec — the high-level engine wiring the whole funnel together.

One object trains and serves the multimodal recommender:

    encode items (multimodal) -> train two-tower -> build ANN index ->
    train DCN-v2 ranker -> materialize online features -> serve funnel

It is what the CLI, API, tests and benchmarks instantiate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .config import Settings
from .data.mockdata import Dataset
from .feature_store.store import FeatureStore
from .indexing.ann import build_index
from .linalg import dot
from .logging_setup import get_logger
from .models import Interaction, Item, User
from .multimodal.fusion import MultimodalEncoder
from .pipeline.streaming import StreamProcessor
from .ranking.dcn import DCNv2Ranker, RankReport
from .ranking.features import RankingFeaturizer
from .retrieval.two_tower import TrainReport, TwoTowerModel
from .serving.cache import build_cache
from .serving.funnel import FunnelResult, RecommendationFunnel

log = get_logger("engine")


@dataclass
class FitReport:
    n_items: int
    n_users: int
    n_interactions: int
    retrieval: TrainReport
    ranking: RankReport


class OmniRec:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        s = self.settings
        self.encoder = MultimodalEncoder(s)
        self.tower = TwoTowerModel(item_dim=s.item_dim, user_dim=s.user_dim, seed=s.seed)
        self.index = build_index(s.index_backend, s.user_dim, s.hnsw_m, s.hnsw_ef, s.milvus_uri, s.seed)
        self.ranker: Optional[DCNv2Ranker] = None
        self.featurizer = RankingFeaturizer(tower_dim=s.user_dim, seed=s.seed + 8)
        self.feature_store = FeatureStore()
        self.cache = build_cache(s.cache_backend, s.redis_url)
        self.items: dict[str, Item] = {}
        self.users: dict[str, User] = {}
        self.item_emb: dict[str, list[float]] = {}
        self.funnel: Optional[RecommendationFunnel] = None
        self.stream: Optional[StreamProcessor] = None

    # ---- training -------------------------------------------------------

    def fit(self, items: Sequence[Item], users: dict[str, User],
            interactions: Sequence[Interaction], retrieval_epochs: int = 8,
            ranking_epochs: int = 8) -> FitReport:
        self.items = {it.id: it for it in items}
        self.users = dict(users)

        log.info("encoding %d items (multimodal)", len(items))
        self.item_emb = self.encoder.encode_items(items)

        log.info("training two-tower retrieval")
        r_report = self.tower.fit(interactions, self.item_emb, self.users, epochs=retrieval_epochs)

        log.info("building ANN index")
        item_vecs = self.tower.all_item_vectors()
        self.index.add_many(item_vecs)
        self.cache.warm(item_vecs)

        log.info("training DCN-v2 ranker")
        X, y = self._ranking_dataset(interactions)
        self.ranker = DCNv2Ranker(in_dim=self.featurizer.dim, seed=self.settings.seed + 3)
        k_report = self.ranker.fit(X, y, epochs=ranking_epochs)

        # materialize offline user vectors into the online store
        for uid, user in self.users.items():
            self.feature_store.put_offline_user(uid, self.tower.user_vector(user))
        self.feature_store.materialize()

        self._wire_serving()
        return FitReport(n_items=len(items), n_users=len(users), n_interactions=len(interactions),
                         retrieval=r_report, ranking=k_report)

    def _ranking_dataset(self, interactions: Sequence[Interaction]):
        X, y = [], []
        for ix in interactions:
            if ix.item_id not in self.item_emb:
                continue
            user = self.users.get(ix.user_id) or User(id=ix.user_id)
            uvec = self.tower.user_vector(user)
            ivec = self.tower.item_vector(ix.item_id)
            retr = dot(uvec, ivec)
            X.append(self.featurizer.build(uvec, ivec, self.items[ix.item_id], retr))
            y.append(ix.clicked)
        return X, y

    def _wire_serving(self) -> None:
        s = self.settings
        self.funnel = RecommendationFunnel(
            self.tower, self.index, self.ranker, self.featurizer, self.feature_store,
            self.items, retrieval_k=s.retrieval_k, final_k=s.final_k,
            item_vector_fn=self._item_tower_vec)
        self.stream = StreamProcessor(
            self.feature_store, item_vector_fn=self._item_tower_vec,
            base_vector_fn=self._base_user_vec, click_lr=s.click_lr)

    def _base_user_vec(self, user_id: str) -> list[float]:
        return self.tower.user_vector(self.users.get(user_id) or User(id=user_id))

    def _item_tower_vec(self, item_id: str) -> Optional[list[float]]:
        if item_id not in self.item_emb:
            return None
        return self.cache.get_or_compute(item_id, self.tower.item_vector)

    # ---- serving --------------------------------------------------------

    def recommend(self, user_id: str, k: Optional[int] = None) -> FunnelResult:
        if self.funnel is None:
            raise RuntimeError("engine not fitted")
        state = self.feature_store.get_online(user_id)
        history = state.history if state and state.history else \
            (self.users[user_id].history if user_id in self.users else [])
        user = User(id=user_id, history=history)
        return self.funnel.recommend(user, k=k)

    def record_click(self, user_id: str, item_id: str) -> Optional[str]:
        if self.stream is None:
            raise RuntimeError("engine not fitted")
        return self.stream.process({"user_id": user_id, "item_id": item_id, "clicked": 1})

    # ---- convenience ----------------------------------------------------

    def fit_dataset(self, ds: Dataset, **kw) -> FitReport:
        return self.fit(ds.items, ds.users, ds.interactions, **kw)

    def stats(self) -> dict:
        return {"items": len(self.items), "users": len(self.users),
                "index_size": len(self.index) if hasattr(self.index, "__len__") else None,
                "feature_store": self.feature_store.stats(),
                "embedding_cache": self.cache.stats.to_dict()}
