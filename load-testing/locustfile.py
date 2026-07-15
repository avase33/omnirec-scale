"""Locust load test simulating up to 10,000 requests/second against the API.

Run against a serving instance:

    omnirec serve                      # in one terminal
    locust -f load-testing/locustfile.py --host http://localhost:8000

Then drive it headless at a target rate, e.g.:

    locust -f load-testing/locustfile.py --host http://localhost:8000 \
           --headless -u 2000 -r 200 -t 2m

Locust reports the p50/p95/p99 latency distribution and achieved RPS. Tasks are
weighted so ~85% are recommendation reads and ~15% are click writes (which
trigger the real-time feature update), mirroring a realistic traffic mix.
"""

from __future__ import annotations

import random

try:
    from locust import HttpUser, between, task
except Exception:  # pragma: no cover - locust optional
    HttpUser = object  # type: ignore

    def task(*a, **k):
        def deco(f):
            return f
        return deco

    def between(*a, **k):
        return 0


N_USERS = 400
N_ITEMS = 1200


class RecommenderUser(HttpUser):  # pragma: no cover - executed by locust
    wait_time = between(0.0, 0.05)

    @task(17)
    def recommend(self):
        uid = f"user_{random.randrange(N_USERS)}"
        self.client.get(f"/recommend/{uid}?k=10", name="/recommend/[uid]")

    @task(3)
    def click(self):
        uid = f"user_{random.randrange(N_USERS)}"
        iid = f"item_{random.randrange(N_ITEMS)}"
        self.client.post("/click", json={"user_id": uid, "item_id": iid}, name="/click")

    @task(1)
    def metrics(self):
        self.client.get("/metrics", name="/metrics")
