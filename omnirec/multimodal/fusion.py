"""Multimodal fusion.

Text and image embeddings are concatenated and projected into a single coherent
item embedding, then L2-normalised. A learned modality gate (here a fixed,
deterministic weighting) balances the two streams — in production this projection
and gate are the trainable fusion head; offline they are a fixed random
projection so item embeddings are stable across runs and machines.
"""

from __future__ import annotations

from typing import Sequence

from ..config import Settings
from ..linalg import matvec, normalize, rng, xavier
from ..models import Item
from .image import build_image_encoder
from .text import build_text_encoder


class MultimodalEncoder:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        s = self.settings
        self.text = build_text_encoder(s.encoder_backend, s.text_dim)
        self.image = build_image_encoder(s.encoder_backend, s.image_dim, s.image_dim, s.seed)
        self.dim = s.item_dim
        # fusion projection: (text_dim + image_dim) -> item_dim
        self._proj = xavier(s.item_dim, s.text_dim + s.image_dim, rng(s.seed + 1))
        self._text_gate = 0.6
        self._image_gate = 0.4

    def encode_item(self, item: Item) -> list[float]:
        t = self.text.encode(item.text)
        v = self.image.encode(item.image_features, key=item.id)
        gated = [self._text_gate * x for x in t] + [self._image_gate * x for x in v]
        return normalize(matvec(self._proj, gated))

    def encode_items(self, items: Sequence[Item]) -> dict[str, list[float]]:
        return {it.id: self.encode_item(it) for it in items}

    def encode_query_text(self, text: str) -> list[float]:
        """Encode a free-text query into the fused space (image stream zeroed)."""
        t = self.text.encode(text)
        gated = [self._text_gate * x for x in t] + [0.0] * self.settings.image_dim
        return normalize(matvec(self._proj, gated))
