"""Latency + throughput benchmark for the serving funnel.

Fires a stream of recommendation requests through the full funnel and reports the
latency SLA table (p50/p95/p99) plus achieved throughput and the embedding-cache
hit-rate — the numbers a senior eng manager looks for. Absolute latency depends
on hardware; the percentiles and the shape of the funnel are what the benchmark
demonstrates.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass


@dataclass
class LatencyReport:
    n_requests: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    throughput_rps: float
    cache_hit_rate: float

    def to_dict(self) -> dict:
        return {"n_requests": self.n_requests, "p50_ms": round(self.p50_ms, 2),
                "p95_ms": round(self.p95_ms, 2), "p99_ms": round(self.p99_ms, 2),
                "mean_ms": round(self.mean_ms, 2),
                "throughput_rps": round(self.throughput_rps, 1),
                "cache_hit_rate": round(self.cache_hit_rate, 4)}


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round(p / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def benchmark_latency(engine, n_requests: int = 2000, seed: int = 3) -> LatencyReport:
    rng = random.Random(seed)
    user_ids = list(engine.users.keys())
    latencies: list[float] = []

    t_start = time.perf_counter()
    for _ in range(n_requests):
        uid = rng.choice(user_ids)
        res = engine.recommend(uid)
        latencies.append(res.latency_ms)
    wall = time.perf_counter() - t_start

    latencies.sort()
    return LatencyReport(
        n_requests=n_requests,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
        mean_ms=sum(latencies) / len(latencies),
        throughput_rps=n_requests / wall if wall else 0.0,
        cache_hit_rate=engine.cache.stats.hit_rate,
    )
