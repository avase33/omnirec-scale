"""Image encoder.

Default backend projects a raw image feature vector (the stand-in for CLIP pixel
features carried on each :class:`~omnirec.models.Item`) into the shared image
embedding space with a fixed random projection, then L2-normalises. When an item
carries no image, a deterministic hash of its identity/attributes is used so the
pipeline degrades gracefully instead of dropping the modality.

Set ``OMNIREC_ENCODER=clip-bert`` to use a real CLIP vision tower via
:class:`ClipImageEncoder`.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from ..linalg import matvec, normalize, rng, xavier


class ProjectionImageEncoder:
    def __init__(self, in_dim: int = 64, dim: int = 64, seed: int = 7) -> None:
        self.in_dim = in_dim
        self.dim = dim
        # A fixed random projection — deterministic given the seed.
        self._proj = xavier(dim, in_dim, rng(seed))

    def _from_identity(self, key: str) -> list[float]:
        digest = hashlib.sha256(key.encode()).digest()
        return [((digest[i % len(digest)]) / 127.5 - 1.0) for i in range(self.in_dim)]

    def encode(self, image_features: Sequence[float], key: str = "") -> list[float]:
        feats = list(image_features)
        if not feats:
            feats = self._from_identity(key)
        if len(feats) < self.in_dim:
            feats = feats + [0.0] * (self.in_dim - len(feats))
        elif len(feats) > self.in_dim:
            feats = feats[: self.in_dim]
        return normalize(matvec(self._proj, feats))


class ClipImageEncoder:  # pragma: no cover - optional dependency
    def __init__(self, model: str = "ViT-B/32") -> None:
        import clip  # type: ignore
        import torch  # type: ignore

        self._torch = torch
        self._model, self._preprocess = clip.load(model, device="cpu")
        self.dim = 512

    def encode(self, image, key: str = "") -> list[float]:
        with self._torch.no_grad():
            t = self._preprocess(image).unsqueeze(0)
            v = self._model.encode_image(t)[0]
        return normalize([float(x) for x in v])


def build_image_encoder(backend: str = "mock", in_dim: int = 64, dim: int = 64, seed: int = 7):
    if backend == "clip-bert":
        return ClipImageEncoder()
    return ProjectionImageEncoder(in_dim=in_dim, dim=dim, seed=seed)
