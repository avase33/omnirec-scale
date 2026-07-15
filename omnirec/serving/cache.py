"""Dense-embedding cache.

Most of a 10M-item catalog is cold. Caching the dense tower embeddings of
inactive products keeps them out of recompute and out of expensive stores; a hit
is a dictionary lookup. Hit/miss stats back the README's cost-efficiency claim.
Default backend is process-local; a Redis backend shares the cache across
serving replicas.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional, Sequence


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return 0.0 if self.total == 0 else self.hits / self.total

    def to_dict(self) -> dict:
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hit_rate, 4)}


class InMemoryBackend:
    def __init__(self) -> None:
        self._d: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[list[float]]:
        return self._d.get(key)

    def set(self, key: str, value: Sequence[float]) -> None:
        with self._lock:
            self._d[key] = list(value)


class RedisBackend:  # pragma: no cover - requires redis
    def __init__(self, url: str, prefix: str = "omnirec:emb:") -> None:
        import redis  # type: ignore

        self._r = redis.Redis.from_url(url)
        self._prefix = prefix

    def get(self, key: str) -> Optional[list[float]]:
        import json
        v = self._r.get(self._prefix + key)
        return json.loads(v) if v else None

    def set(self, key: str, value: Sequence[float]) -> None:
        import json
        self._r.set(self._prefix + key, json.dumps(list(value)))


class EmbeddingCache:
    def __init__(self, backend=None) -> None:
        self.backend = backend or InMemoryBackend()
        self.stats = CacheStats()

    def get_or_compute(self, key: str, compute: Callable[[str], list[float]]) -> list[float]:
        v = self.backend.get(key)
        if v is not None:
            self.stats.hits += 1
            return v
        self.stats.misses += 1
        v = compute(key)
        self.backend.set(key, v)
        return v

    def warm(self, items: dict[str, Sequence[float]]) -> None:
        for k, v in items.items():
            self.backend.set(k, v)


def build_cache(backend: str = "memory", redis_url: str = "") -> EmbeddingCache:
    if backend == "redis" and redis_url:
        return EmbeddingCache(RedisBackend(redis_url))
    return EmbeddingCache(InMemoryBackend())
