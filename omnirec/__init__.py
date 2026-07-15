"""OmniRec-Scale — production-grade multimodal recommendation engine.

A multi-stage recommendation funnel modelled on the systems that power
Amazon / Netflix / Meta-scale recommenders:

    retrieval (two-tower) -> ANN vector filter -> DCN-v2 ranking -> top-K

with multimodal (text + image) item embeddings, a Feast-style online feature
store, and a streaming pipeline that updates user embeddings in real time as
clicks arrive.

Offline-first: every heavy dependency (PyTorch, CLIP/BERT, Milvus, Feast, Kafka,
ONNX/Triton, FastAPI, Redis) has a pure-Python default so the whole funnel
trains, serves and benchmarks with zero external services. Real adapters wire in
via configuration for production.
"""

from .version import __version__
from .config import Settings
from .engine import OmniRec

__all__ = ["__version__", "Settings", "OmniRec"]
