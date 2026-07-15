"""Approximate nearest-neighbour index (HNSW-inspired, pure Python).

Implements a single-layer Navigable Small World graph — the core idea behind
HNSW that Milvus/Qdrant use to cut 10,000,000 items down to a few hundred
candidates in sub-millisecond time:

* **Insert** connects each new node to its ``M`` nearest neighbours found by
  greedy graph search over the current graph (bidirectional, degree-pruned).
* **Search** is greedy best-first with an ``ef`` dynamic candidate list.

This gives high recall at a fraction of brute-force cost. ``exact_search`` is kept
for measuring recall. For true 10M-scale, :class:`MilvusIndex` is a drop-in
adapter behind the same interface.
"""

from __future__ import annotations

import heapq
from typing import Optional, Sequence

from ..linalg import dot, norm


class HNSWIndex:
    def __init__(self, dim: int, m: int = 16, ef: int = 64, seed: int = 7) -> None:
        self.dim = dim
        self.M = m
        self.ef = ef
        self._ids: list[str] = []
        self._vecs: dict[str, list[float]] = {}
        self._norm: dict[str, float] = {}
        self._graph: dict[str, set[str]] = {}
        self._entry: Optional[str] = None
        import random
        self._r = random.Random(seed)

    def __len__(self) -> int:
        return len(self._ids)

    # ---- similarity -----------------------------------------------------

    def _sim(self, q: Sequence[float], qn: float, iid: str) -> float:
        n = self._norm[iid] * qn
        if n == 0:
            return 0.0
        return dot(q, self._vecs[iid]) / n

    # ---- build ----------------------------------------------------------

    def add(self, iid: str, vec: Sequence[float]) -> None:
        v = list(vec)
        self._vecs[iid] = v
        self._norm[iid] = norm(v) or 1e-12
        self._graph.setdefault(iid, set())
        if self._entry is None:
            self._entry = iid
            self._ids.append(iid)
            return
        # find M nearest via graph search, then connect bidirectionally
        neighbors = self._search_ids(v, self._norm[iid], k=self.M, ef=max(self.ef, self.M))
        for _, nid in neighbors:
            self._graph[iid].add(nid)
            self._graph[nid].add(iid)
            if len(self._graph[nid]) > self.M * 2:
                self._prune(nid)
        self._ids.append(iid)

    def add_many(self, items: dict[str, Sequence[float]]) -> None:
        for iid, vec in items.items():
            self.add(iid, vec)

    def _prune(self, iid: str) -> None:
        v, vn = self._vecs[iid], self._norm[iid]
        ranked = sorted(self._graph[iid], key=lambda n: self._sim(v, vn, n), reverse=True)
        keep = set(ranked[: self.M * 2])
        for dropped in set(self._graph[iid]) - keep:
            self._graph[iid].discard(dropped)

    # ---- search ---------------------------------------------------------

    def _search_ids(self, q: list[float], qn: float, k: int, ef: int) -> list[tuple[float, str]]:
        if self._entry is None:
            return []
        entries = {self._entry}
        # a couple of random entry points improve recall on a disconnected graph
        for _ in range(min(2, len(self._ids))):
            entries.add(self._r.choice(self._ids))

        visited: set[str] = set()
        candidates: list[tuple[float, str]] = []   # min-heap on -sim (explore best first)
        results: list[tuple[float, str]] = []      # min-heap on sim (keep top ef)
        for e in entries:
            s = self._sim(q, qn, e)
            visited.add(e)
            heapq.heappush(candidates, (-s, e))
            heapq.heappush(results, (s, e))

        while candidates:
            neg_s, c = heapq.heappop(candidates)
            c_sim = -neg_s
            if results and c_sim < results[0][0] and len(results) >= ef:
                break
            for nb in self._graph.get(c, ()):
                if nb in visited:
                    continue
                visited.add(nb)
                s = self._sim(q, qn, nb)
                if len(results) < ef or s > results[0][0]:
                    heapq.heappush(candidates, (-s, nb))
                    heapq.heappush(results, (s, nb))
                    if len(results) > ef:
                        heapq.heappop(results)
        results.sort(reverse=True)
        return results[:k]

    def search(self, query: Sequence[float], k: int = 10, ef: Optional[int] = None) -> list[tuple[str, float]]:
        q = list(query)
        qn = norm(q) or 1e-12
        res = self._search_ids(q, qn, k=k, ef=ef or self.ef)
        return [(iid, sim) for sim, iid in res]

    def exact_search(self, query: Sequence[float], k: int = 10) -> list[tuple[str, float]]:
        q = list(query)
        qn = norm(q) or 1e-12
        scored = [(iid, self._sim(q, qn, iid)) for iid in self._ids]
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:k]


class MilvusIndex:  # pragma: no cover - requires milvus
    """Adapter backing retrieval with Milvus HNSW at 10M+ scale."""

    def __init__(self, dim: int, uri: str, collection: str = "omnirec_items",
                 m: int = 16, ef: int = 64) -> None:
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility  # type: ignore

        connections.connect(uri=uri)
        self.dim = dim
        self.collection_name = collection
        self._ef = ef
        if not utility.has_collection(collection):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ]
            self._col = Collection(collection, CollectionSchema(fields))
            self._col.create_index("vec", {"index_type": "HNSW", "metric_type": "IP",
                                           "params": {"M": m, "efConstruction": 200}})
        else:
            self._col = Collection(collection)
        self._col.load()

    def add_many(self, items: dict) -> None:
        ids = list(items.keys())
        vecs = [list(v) for v in items.values()]
        self._col.insert([ids, vecs])
        self._col.flush()

    def search(self, query, k: int = 10, ef=None):
        res = self._col.search([list(query)], "vec", {"metric_type": "IP", "params": {"ef": ef or self._ef}},
                               limit=k, output_fields=["id"])
        return [(hit.entity.get("id"), float(hit.score)) for hit in res[0]]


def build_index(backend: str, dim: int, m: int = 16, ef: int = 64, uri: str = "", seed: int = 7):
    if backend == "milvus" and uri:
        return MilvusIndex(dim, uri, m=m, ef=ef)
    return HNSWIndex(dim, m=m, ef=ef, seed=seed)
