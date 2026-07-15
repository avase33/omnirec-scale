"""Command-line interface for OmniRec-Scale."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .benchmark import benchmark_latency
from .config import Settings
from .data.mockdata import generate_dataset
from .engine import OmniRec
from .evaluation import retrieval_precision_at_k
from .logging_setup import configure_logging
from .version import __version__


def _reconfigure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def _build_engine(args) -> tuple[OmniRec, object]:
    ds = generate_dataset(n_items=args.items, n_users=args.users, seed=args.seed)
    engine = OmniRec(Settings())
    report = engine.fit_dataset(ds, retrieval_epochs=args.epochs, ranking_epochs=args.epochs)
    return engine, (ds, report)


def cmd_demo(args) -> int:
    engine, (ds, report) = _build_engine(args)
    ev = retrieval_precision_at_k(engine, ds.user_pref, ds.item_category, k=20)
    lat = benchmark_latency(engine, n_requests=args.requests)

    print(f"OmniRec-Scale demo — {report.n_items} items, {report.n_users} users, "
          f"{report.n_interactions} interactions")
    print(f"  retrieval two-tower  : final loss {report.retrieval.final_loss:.4f}")
    print(f"  DCN-v2 ranker        : logloss {report.ranking.final_logloss:.4f}, "
          f"AUC {report.ranking.auc:.3f}")
    print(f"  retrieval precision@20: {ev.precision_at_k:.3f} "
          f"(random {ev.random_baseline:.3f}, {ev.lift:.1f}x lift)")
    print(f"  latency  p50={lat.p50_ms:.2f}ms  p95={lat.p95_ms:.2f}ms  p99={lat.p99_ms:.2f}ms")
    print(f"  throughput           : {lat.throughput_rps:,.0f} req/s (single process)")
    print(f"  embedding cache hit  : {lat.cache_hit_rate:.0%}")

    uid = next(iter(engine.users))
    print(f"\nTop recommendations for {uid} (prefers '{ds.user_pref[uid]}'):")
    for rec in engine.recommend(uid, k=args.top).recommendations:
        cat = ds.item_category.get(rec.item_id, "?")
        print(f"  {rec.rank:2d}. {rec.title:<28} [{cat:<11}] ctr={rec.ctr_score:.3f}")
    return 0


def cmd_recommend(args) -> int:
    engine, (ds, _) = _build_engine(args)
    uid = args.user or next(iter(engine.users))
    res = engine.recommend(uid, k=args.top)
    print(json.dumps({"user": uid, "latency_ms": round(res.latency_ms, 2),
                      "recommendations": [r.to_dict() for r in res.recommendations]}, indent=2))
    return 0


def cmd_bench(args) -> int:
    engine, _ = _build_engine(args)
    lat = benchmark_latency(engine, n_requests=args.requests)
    print(json.dumps(lat.to_dict(), indent=2))
    return 0


def cmd_stats(args) -> int:
    engine, _ = _build_engine(args)
    print(json.dumps(engine.stats(), indent=2))
    return 0


def cmd_serve(args) -> int:
    from .serving.api import run_server

    run_server(host=args.host, port=args.port, items=args.items, users=args.users, seed=args.seed)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="omnirec", description="Multimodal recommendation engine")
    p.add_argument("--version", action="version", version=f"omnirec {__version__}")
    p.add_argument("--items", type=int, default=1200)
    p.add_argument("--users", type=int, default=400)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="train and show metrics + recommendations")
    d.add_argument("--requests", type=int, default=500)
    d.add_argument("--top", type=int, default=10)
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("recommend", help="recommend for a user")
    r.add_argument("--user", default=None)
    r.add_argument("--top", type=int, default=10)
    r.set_defaults(func=cmd_recommend)

    b = sub.add_parser("bench", help="latency + throughput benchmark")
    b.add_argument("--requests", type=int, default=2000)
    b.set_defaults(func=cmd_bench)

    s = sub.add_parser("stats", help="engine stats")
    s.set_defaults(func=cmd_stats)

    sv = sub.add_parser("serve", help="run the FastAPI serving API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    _reconfigure_stdout()
    args = build_parser().parse_args(argv)
    configure_logging("DEBUG" if args.verbose else "WARNING")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
