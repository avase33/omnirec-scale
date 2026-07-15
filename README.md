<div align="center">

# OmniRec-Scale

### Production-grade, dual-tower multimodal recommendation engine

Two-Tower retrieval → ANN (HNSW) filtering → DCN-v2 ranking, with multimodal
(text + image) item embeddings and **real-time** user-feature updates. The exact
multi-stage funnel behind Amazon / Netflix / Meta-scale recommenders.

[![CI](https://github.com/avase33/omnirec-scale/actions/workflows/ci.yml/badge.svg)](https://github.com/avase33/omnirec-scale/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-000000.svg)](https://github.com/astral-sh/ruff)

</div>

---

## The scale challenge

Sift ~10,000,000 items down to ~10 hyper-personalised products per user in tens
of milliseconds. You cannot score 10M items per request, so you build a **funnel**:

```
user → [two-tower user vector] → ANN retrieval (10M→500) → DCN-v2 ranking (500→10)
```

ANN does the huge cut cheaply; the expensive ranker only ever sees a few hundred
candidates, so latency stays flat as the catalog grows.

## What's implemented (and actually runs)

Every model here is **built and trained from scratch in pure Python** — no
framework needed to run it — and the whole thing trains, serves and benchmarks
with **zero external services**:

- **Two-Tower retrieval** — dual encoder trained with in-batch sampled-softmax;
  the user tower reacts to click history, the item tower feeds the ANN index.
- **HNSW-style ANN index** — single-layer navigable small-world graph with greedy
  `ef`-search; ≥ 0.8 recall@10 vs exact on the test set. Milvus adapter for 10M+.
- **DCN-v2 ranker** — Deep & Cross Network v2 (full-matrix cross layers + deep
  MLP) predicting CTR, trained with hand-derived backprop.
- **Multimodal fusion** — text (BERT-style) + image (CLIP-style) embeddings fused
  into one item vector.
- **Streaming feature updates** — clicks update the user embedding in the online
  feature store in real time (Feast/Bytewax adapters for production).
- **Serving funnel + FastAPI** — `/recommend`, `/click`, `/metrics`, with a
  dense-embedding cache on the hot path.

Production adapters (PyTorch, CLIP/BERT, Milvus, Feast, Kafka, Redis,
ONNX/Triton) switch on via environment variables and never change the funnel
logic.

## Quickstart (no dependencies)

```bash
pip install -e .

# Train the full funnel on synthetic data and print metrics + recommendations
omnirec demo

# Or generate a catalog and train
python scripts/generate_mock_catalog.py -n 5000
```

Example output:

```
OmniRec-Scale demo — 1200 items, 400 users, 8000 interactions
  retrieval two-tower  : final loss 1.83
  DCN-v2 ranker        : logloss 0.44, AUC 0.86
  retrieval precision@20: 0.71 (random 0.100, 7.1x lift)
  latency  p50=0.42ms  p95=0.71ms  p99=0.98ms
  throughput           : 2,300 req/s (single process)
  embedding cache hit  : 100%
```

> Absolute latency depends on hardware and catalog size; the funnel shape, the
> percentile SLA table and the recall/AUC lifts are what the benchmark proves.

## Serve the API

```bash
pip install -e ".[serve]"
omnirec serve                     # http://localhost:8000
```

```
GET  /recommend/{user_id}?k=10    run the funnel
POST /click {user_id,item_id}     record a click -> real-time feature update
GET  /metrics                     Prometheus exposition
GET  /stats                       engine + cache stats
```

## Scale validation

Latency SLA table (single process, synthetic 1,200-item catalog — illustrative):

| metric | value |
|---|---|
| p50 | ~0.4 ms |
| p95 | ~0.7 ms |
| p99 | ~1.0 ms |
| retrieval precision@20 | ~7x over random |
| DCN-v2 AUC | ~0.86 |
| embedding cache hit-rate | ~100% on hot path |

Reproduce:

```bash
omnirec bench --requests 3000
```

**Load testing** at 10,000 req/s uses Locust against a live server:

```bash
locust -f load-testing/locustfile.py --host http://localhost:8000 \
       --headless -u 2000 -r 200 -t 2m
```

## Full infrastructure

```bash
# Kafka + Redis + Milvus + etcd/MinIO + Prometheus + Grafana + serving API
docker compose -f docker-compose.infra.yml up --build

# Kubernetes with Horizontal Pod Autoscaler (CPU + p95 latency)
kubectl apply -f kubernetes/deployment.yaml -f kubernetes/hpa.yaml
# or Helm
helm install omnirec kubernetes/helm
```

## Repository layout

```
omnirec/
  multimodal/    text (BERT-style) + image (CLIP-style) encoders + fusion
  retrieval/     two-tower model (trained from scratch)
  indexing/      HNSW ANN index + Milvus adapter
  ranking/       DCN-v2 CTR model + feature engineering
  feature_store/ online/offline feature store (Feast-style)
  pipeline/      streaming click -> real-time user-embedding updates
  serving/       funnel, embedding cache, FastAPI
  data/          synthetic e-commerce data generator
  engine.py      end-to-end façade   |   benchmark.py, evaluation.py, cli.py
kubernetes/      Deployment, HPA, Helm chart
load-testing/    Locust script (10k req/s)
monitoring/      Prometheus + Grafana
docker-compose.infra.yml, Dockerfile, .github/workflows/ci.yml
```

## Development

```bash
pip install -e ".[serve,dev]"
pytest --cov=omnirec
ruff check omnirec scripts
python verify_omnirec.py       # offline end-to-end self-check
```

## License

MIT © Akhil Vase
