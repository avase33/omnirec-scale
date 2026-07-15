#!/usr/bin/env python3
"""Generate a synthetic catalog + interactions and (optionally) train the engine.

    python scripts/generate_mock_catalog.py                 # 1,200 items, train, show metrics
    python scripts/generate_mock_catalog.py -n 20000        # bigger catalog
    python scripts/generate_mock_catalog.py --jsonl out/    # dump items + interactions as JSONL
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Generate mock e-commerce data for OmniRec-Scale")
    ap.add_argument("-n", "--items", type=int, default=1200)
    ap.add_argument("-u", "--users", type=int, default=400)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--jsonl", default=None, help="directory to write items.jsonl + interactions.jsonl")
    ap.add_argument("--no-train", action="store_true", help="only generate data, do not train")
    args = ap.parse_args(argv)

    from omnirec.data.mockdata import generate_dataset

    ds = generate_dataset(n_items=args.items, n_users=args.users, seed=args.seed)
    print(f"Generated {len(ds.items)} items, {len(ds.users)} users, {len(ds.interactions)} interactions")

    if args.jsonl:
        os.makedirs(args.jsonl, exist_ok=True)
        with open(os.path.join(args.jsonl, "items.jsonl"), "w", encoding="utf-8") as f:
            for it in ds.items:
                f.write(json.dumps({"id": it.id, "title": it.title, "description": it.description,
                                    "category": it.category, "brand": it.brand,
                                    "price": it.price}) + "\n")
        with open(os.path.join(args.jsonl, "interactions.jsonl"), "w", encoding="utf-8") as f:
            for ix in ds.interactions:
                f.write(json.dumps({"user_id": ix.user_id, "item_id": ix.item_id,
                                    "clicked": ix.clicked, "ts": ix.ts}) + "\n")
        print(f"Wrote JSONL to {args.jsonl}/")

    if args.no_train:
        return 0

    from omnirec.config import Settings
    from omnirec.engine import OmniRec
    from omnirec.evaluation import retrieval_precision_at_k
    from omnirec.benchmark import benchmark_latency

    engine = OmniRec(Settings())
    report = engine.fit_dataset(ds)
    ev = retrieval_precision_at_k(engine, ds.user_pref, ds.item_category, k=20)
    lat = benchmark_latency(engine, n_requests=1000)

    print(f"\nTrained: two-tower loss {report.retrieval.final_loss:.4f}, "
          f"DCN-v2 AUC {report.ranking.auc:.3f}")
    print(f"Retrieval precision@20 {ev.precision_at_k:.3f} ({ev.lift:.1f}x over random)")
    print(f"Latency p50={lat.p50_ms:.2f}ms p95={lat.p95_ms:.2f}ms p99={lat.p99_ms:.2f}ms, "
          f"{lat.throughput_rps:,.0f} req/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
