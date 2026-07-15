"""Feature store with online + offline tiers.

Mirrors the Feast pattern: an **offline** tier holds the historical, batch-computed
truth (used to train and to bootstrap), and an **online** tier holds the
low-latency, continuously-updated features that serving reads. The streaming
pipeline writes fresh user embeddings to the online tier on every click; a
``materialize`` step syncs offline → online.

Default is in-process dictionaries; a Feast adapter can back the same interface.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from ..errors import FeatureStoreError
from ..linalg import normalize


@dataclass
class UserState:
    user_id: str
    vector: list[float]
    history: list[str] = field(default_factory=list)
    clicks: int = 0
    updated_ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"user_id": self.user_id, "clicks": self.clicks,
                "history_len": len(self.history), "updated_ts": self.updated_ts}


class FeatureStore:
    def __init__(self) -> None:
        self._online: dict[str, UserState] = {}
        self._offline: dict[str, list[float]] = {}   # batch-computed user vectors
        self._item_features: dict[str, dict] = {}
        self._lock = threading.Lock()

    # ---- offline (batch) ------------------------------------------------

    def put_offline_user(self, user_id: str, vector: list[float]) -> None:
        self._offline[user_id] = list(vector)

    def put_item_features(self, item_id: str, features: dict) -> None:
        self._item_features[item_id] = features

    def get_item_features(self, item_id: str) -> Optional[dict]:
        return self._item_features.get(item_id)

    def materialize(self) -> int:
        """Sync offline user vectors into the online store (Feast materialize)."""
        with self._lock:
            n = 0
            for uid, vec in self._offline.items():
                if uid not in self._online:
                    self._online[uid] = UserState(user_id=uid, vector=list(vec))
                    n += 1
        return n

    # ---- online (serving) ----------------------------------------------

    def get_online(self, user_id: str) -> Optional[UserState]:
        return self._online.get(user_id)

    def upsert_online(self, state: UserState) -> None:
        with self._lock:
            self._online[state.user_id] = state

    def apply_click(self, user_id: str, item_id: str, item_vector: list[float],
                    lr: float = 0.35, base_vector: Optional[list[float]] = None) -> UserState:
        """Incrementally nudge the user's online embedding toward a clicked item.

        This is the real-time feature update: a streaming EMA that reacts to
        behaviour within a single event, no batch job required.
        """
        with self._lock:
            state = self._online.get(user_id)
            if state is None:
                start = base_vector if base_vector is not None else item_vector
                state = UserState(user_id=user_id, vector=list(start))
            if not state.vector:
                raise FeatureStoreError("empty user vector")
            state.vector = normalize([(1 - lr) * u + lr * v
                                      for u, v in zip(state.vector, item_vector)])
            state.history.append(item_id)
            if len(state.history) > 100:
                state.history = state.history[-100:]
            state.clicks += 1
            state.updated_ts = time.time()
            self._online[user_id] = state
            return state

    def online_size(self) -> int:
        return len(self._online)

    def stats(self) -> dict:
        return {"online_users": len(self._online), "offline_users": len(self._offline),
                "items": len(self._item_features)}
