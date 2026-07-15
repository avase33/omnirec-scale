# Architecture

OmniRec-Scale is a **multi-stage recommendation funnel** — the pattern behind
large-scale industrial recommenders. The design goal: pick the best ~10 items
for a user out of ~10,000,000, in tens of milliseconds, by spending compute only
where it pays off.

## The funnel

```
                         user request
                              │
                 ┌────────────▼─────────────┐
                 │  user vector             │  online feature store (fresh)
                 │  (online store or tower) │  ── else two-tower user encoder
                 └────────────┬─────────────┘
                              │  dense query vector
                 ┌────────────▼─────────────┐
   STAGE 1       │  ANN retrieval (HNSW)    │  10,000,000 ──► top 500
   retrieval     │  Milvus / in-proc index  │  sub-ms, high recall
                 └────────────┬─────────────┘
                              │  500 candidates
                 ┌────────────▼─────────────┐
   STAGE 2       │  DCN-v2 ranking          │  500 ──► scored by CTR
   ranking       │  explicit feature crosses│  only 500 expensive scores
                 └────────────┬─────────────┘
                              │  sort by CTR
                        top-K (e.g. 10)
```

Latency stays flat as the catalog grows because ANN does the huge cut and the
expensive ranker only ever sees `retrieval_k` items.

## Multimodal item embeddings

Each item is encoded from **both** modalities and fused into one vector:

* **Text tower** — a transformer text encoder (BERT/Sentence-Transformers in
  production; a deterministic hashing encoder offline) over title + description +
  attributes.
* **Image tower** — a CLIP vision encoder (offline: a fixed projection of the
  item's raw image features).
* **Fusion** — concatenate + project + L2-normalise into a single coherent item
  embedding. Both modalities carry the category signal, so items cluster by
  meaning, which is what makes retrieval and ranking learn.

## Retrieval: two-tower

A dual encoder maps users and items into a shared space where dot-product ≈
affinity. The **item tower** projects the fused embedding; the **user tower**
projects `[user-id embedding ⊕ mean of history item embeddings]`. Trained with
**in-batch sampled-softmax** — every other item in a mini-batch is a negative —
the standard scalable retrieval objective. The item tower's outputs are what the
ANN index stores.

## Ranking: DCN-v2

The Deep & Cross Network v2 scores each retrieved candidate's click-through rate.
A **cross network** (`x_{l+1} = x0 ⊙ (W_l x_l + b_l) + x_l`, full-matrix "v2"
form) learns explicit bounded-degree feature interactions cheaply, in parallel
with a **deep MLP** for implicit ones; their outputs combine into a CTR logit.
Trained on click labels with binary cross-entropy.

## Real-time features (streaming)

User taste drifts within a session. A streaming pipeline (Bytewax/Flink in
production; an in-memory dataflow offline) consumes the click stream and, per
click, nudges the user's embedding in the **online feature store** via an EMA
toward the clicked item. Serving reads that fresh vector, so the next request
already reflects the click — no batch job in the loop. A `materialize` step syncs
the batch-computed **offline** vectors into the online tier.

## Scale & cost

* **ANN (HNSW)** turns the 10M→500 cut into a graph walk instead of a full scan.
* **Embedding cache** keeps dense vectors of cold products out of recompute; on
  the hot serving path item-vector lookups become dictionary hits (the benchmark
  reports the hit-rate).
* **Horizontal autoscaling** — the serving API is stateless and exposes
  `/metrics`; a Kubernetes HPA scales on CPU and p95 latency (see
  `kubernetes/hpa.yaml`).
* **Inference optimisation** — production ranking models export to ONNX/TensorRT
  and serve on Triton for dynamic GPU batching (adapters; the offline path uses
  the pure-Python model).

## Offline-first

Every heavy dependency has a pure-Python default: mock encoders, hashing
embeddings, in-process HNSW index, in-memory feature store + cache + event
stream. The whole funnel therefore **trains, serves and benchmarks with zero
external services**, and the same code swaps in CLIP/BERT, Milvus, Feast, Kafka,
Redis and Triton via configuration for production.
