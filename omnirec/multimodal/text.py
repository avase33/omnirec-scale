"""Text encoder.

Default backend is a deterministic hashing bag-of-embeddings encoder that stands
in for a transformer text tower (BERT/Sentence-Transformers): tokens are hashed
into a fixed-width space, summed with sublinear tf weighting and L2-normalised.
It needs no model download and is fully reproducible, so downstream fusion,
retrieval and ranking behave identically everywhere.

Set ``OMNIREC_ENCODER=clip-bert`` to use a real sentence-transformer via
:class:`BertTextEncoder`.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

from ..linalg import normalize

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("the a an of to and or for with in on at is are this that".split())


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


def _h(token: str) -> int:
    return int.from_bytes(hashlib.md5(token.encode()).digest()[:8], "big")


class HashingTextEncoder:
    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, text: str) -> list[float]:
        counts: dict[int, float] = {}
        for tok in _tokens(text):
            h = _h(tok)
            counts[h] = counts.get(h, 0.0) + 1.0
        vec = [0.0] * self.dim
        for h, c in counts.items():
            sign = 1.0 if (h >> 62) & 1 else -1.0
            vec[h % self.dim] += sign * (1.0 + math.log(c))
        return normalize(vec)

    def encode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.encode(t) for t in texts]


class BertTextEncoder:  # pragma: no cover - optional dependency
    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._m = SentenceTransformer(model)
        self.dim = int(self._m.get_sentence_embedding_dimension())

    def encode(self, text: str) -> list[float]:
        return normalize([float(x) for x in self._m.encode(text)])

    def encode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [normalize([float(x) for x in v]) for v in self._m.encode(list(texts))]


def build_text_encoder(backend: str = "mock", dim: int = 64):
    if backend == "clip-bert":
        return BertTextEncoder()
    return HashingTextEncoder(dim=dim)
