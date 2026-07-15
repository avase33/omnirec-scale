"""Central configuration resolved from the environment.

Offline-first defaults keep the whole funnel in-process (mock encoders, in-memory
ANN index + feature store + cache). Point the adapters at CLIP/BERT, Milvus,
Feast, Kafka and Redis for production via environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Embedding dimensions (compact by default so the pure-Python path is fast;
    # bump these when running on the PyTorch backend).
    text_dim: int = 32
    image_dim: int = 32
    item_dim: int = 32          # fused multimodal item embedding
    user_dim: int = 32          # two-tower shared space

    # Encoders: mock | clip-bert
    encoder_backend: str = "mock"

    # Vector index: hnsw | milvus
    index_backend: str = "hnsw"
    milvus_uri: str = "http://localhost:19530"
    hnsw_m: int = 16
    hnsw_ef: int = 64

    # Feature store: memory | feast
    feature_store: str = "memory"

    # Cache: memory | redis
    cache_backend: str = "memory"
    redis_url: str = "redis://localhost:6379/0"

    # Funnel — production retrieves ~500; the offline pure-Python ranker uses a
    # smaller candidate set to keep single-process latency low.
    retrieval_k: int = 200      # candidates from ANN stage
    final_k: int = 10           # returned after ranking

    # Streaming
    click_lr: float = 0.35      # EMA rate for online user-embedding updates

    # Training
    seed: int = 13

    @classmethod
    def from_env(cls) -> "Settings":
        g = os.environ.get
        return cls(
            text_dim=int(g("OMNIREC_TEXT_DIM", "32")),
            image_dim=int(g("OMNIREC_IMAGE_DIM", "32")),
            item_dim=int(g("OMNIREC_ITEM_DIM", "32")),
            user_dim=int(g("OMNIREC_USER_DIM", "32")),
            encoder_backend=g("OMNIREC_ENCODER", "mock"),
            index_backend=g("OMNIREC_INDEX", "hnsw"),
            milvus_uri=g("MILVUS_URI", "http://localhost:19530"),
            feature_store=g("OMNIREC_FEATURE_STORE", "memory"),
            cache_backend=g("OMNIREC_CACHE", "memory"),
            redis_url=g("REDIS_URL", "redis://localhost:6379/0"),
            retrieval_k=int(g("OMNIREC_RETRIEVAL_K", "200")),
            final_k=int(g("OMNIREC_FINAL_K", "10")),
            click_lr=float(g("OMNIREC_CLICK_LR", "0.35")),
            seed=int(g("OMNIREC_SEED", "13")),
        )
