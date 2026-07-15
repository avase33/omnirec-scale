"""Minimal pure-Python linear-algebra + init helpers.

No numpy. Vectors are ``list[float]``, matrices are ``list[list[float]]``. This
keeps the models dependency-free so they train and run identically on any
machine and in CI. In production the same math is delegated to PyTorch (the
tensor shapes and update rules match), but nothing here needs a GPU.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

Vector = list
Matrix = list


# ---- element-wise / reductions ------------------------------------------

def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def scale(a: Sequence[float], s: float) -> list[float]:
    return [x * s for x in a]


def hadamard(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x * y for x, y in zip(a, b)]


def axpy(y: list[float], a: float, x: Sequence[float]) -> None:
    """In-place y += a * x (the workhorse of SGD updates)."""
    for i, xi in enumerate(x):
        y[i] += a * xi


def norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def normalize(a: Sequence[float]) -> list[float]:
    n = norm(a)
    return [x / n for x in a] if n > 0 else list(a)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


def matvec(m: Matrix, v: Sequence[float]) -> list[float]:
    return [dot(row, v) for row in m]


def vecmat_t(v: Sequence[float], m: Matrix) -> list[float]:
    """v (len R) times m (R x C) -> len C.  == sum_r v[r] * m[r]."""
    cols = len(m[0]) if m else 0
    out = [0.0] * cols
    for r, vr in enumerate(v):
        row = m[r]
        for c in range(cols):
            out[c] += vr * row[c]
    return out


# ---- activations ---------------------------------------------------------

def relu(v: Sequence[float]) -> list[float]:
    return [x if x > 0 else 0.0 for x in v]


def relu_grad(v: Sequence[float]) -> list[float]:
    return [1.0 if x > 0 else 0.0 for x in v]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softmax(scores: Sequence[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


# ---- initialisation ------------------------------------------------------

def rng(seed: int = 0) -> random.Random:
    return random.Random(seed)


def xavier(rows: int, cols: int, r: random.Random) -> Matrix:
    limit = math.sqrt(6.0 / (rows + cols))
    return [[r.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


def zeros(n: int) -> list[float]:
    return [0.0] * n


def rand_vec(n: int, r: random.Random, scale_: float = 0.1) -> list[float]:
    return [r.gauss(0.0, scale_) for _ in range(n)]
