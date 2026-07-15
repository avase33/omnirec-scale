"""FastAPI serving API.

Exposes the funnel and the streaming feature update:

    GET  /recommend/{user_id}   run the multi-stage funnel
    POST /click                 record a click -> real-time user-embedding update
    GET  /stats                 engine + cache stats
    GET  /metrics               Prometheus exposition (latency, cache, catalog)
    GET  /healthz               liveness

FastAPI/uvicorn are optional; import this module only when serving.
An in-process engine is trained on synthetic data at startup so the API is
demoable with zero external services.
"""

from __future__ import annotations

import time
from typing import Optional

from ..config import Settings
from ..data.mockdata import generate_dataset
from ..engine import OmniRec


def create_app(items: int = 1200, users: int = 400, seed: int = 13,
               engine: Optional[OmniRec] = None):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, PlainTextResponse
    from pydantic import BaseModel

    app = FastAPI(title="OmniRec-Scale", version="0.1.0",
                  description="Production-grade multimodal recommendation engine")

    if engine is None:
        ds = generate_dataset(n_items=items, n_users=users, seed=seed)
        engine = OmniRec(Settings())
        engine.fit_dataset(ds)
    app.state.engine = engine
    app.state.latencies: list[float] = []

    class Click(BaseModel):
        user_id: str
        item_id: str

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/recommend/{user_id}")
    def recommend(user_id: str, k: int = 10):
        eng: OmniRec = app.state.engine
        try:
            res = eng.recommend(user_id, k=k)
        except Exception as exc:  # unknown user etc.
            return JSONResponse({"error": str(exc)}, status_code=404)
        app.state.latencies.append(res.latency_ms)
        return {"user_id": user_id, "latency_ms": round(res.latency_ms, 3),
                "used_online_state": res.used_online_state,
                "recommendations": [r.to_dict() for r in res.recommendations]}

    @app.post("/click")
    def click(c: Click):
        eng: OmniRec = app.state.engine
        uid = eng.record_click(c.user_id, c.item_id)
        return {"updated_user": uid, "applied": uid is not None}

    @app.get("/stats")
    def stats() -> dict:
        return app.state.engine.stats()

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        eng: OmniRec = app.state.engine
        lat = sorted(app.state.latencies)
        p95 = lat[min(len(lat) - 1, int(0.95 * (len(lat) - 1)))] if lat else 0.0
        lines = [
            "# TYPE omnirec_requests_total counter",
            f"omnirec_requests_total {len(lat)}",
            "# TYPE omnirec_latency_p95_ms gauge",
            f"omnirec_latency_p95_ms {p95:.3f}",
            "# TYPE omnirec_cache_hit_rate gauge",
            f"omnirec_cache_hit_rate {eng.cache.stats.hit_rate:.4f}",
            "# TYPE omnirec_catalog_items gauge",
            f"omnirec_catalog_items {len(eng.items)}",
            "# TYPE omnirec_online_users gauge",
            f"omnirec_online_users {eng.feature_store.online_size()}",
        ]
        return "\n".join(lines) + "\n"

    return app


def run_server(host: str = "127.0.0.1", port: int = 8000, items: int = 1200,
               users: int = 400, seed: int = 13) -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(items=items, users=users, seed=seed), host=host, port=port)
