"""Core domain models (dataclasses)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Item:
    id: str
    title: str
    description: str
    category: str
    brand: str
    price: float
    # Raw image feature vector (stand-in for CLIP pixel features); may be empty.
    image_features: list[float] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return f"{self.title}. {self.description} category {self.category} brand {self.brand}"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "category": self.category,
                "brand": self.brand, "price": round(self.price, 2)}


@dataclass
class User:
    id: str
    # Recent interaction history (item ids, most recent last).
    history: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class Interaction:
    user_id: str
    item_id: str
    clicked: int          # 1 = click/positive, 0 = impression-only/negative
    ts: float = 0.0


@dataclass
class Recommendation:
    item_id: str
    title: str
    retrieval_score: float
    ctr_score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "title": self.title,
                "retrieval_score": round(self.retrieval_score, 4),
                "ctr_score": round(self.ctr_score, 4), "rank": self.rank}
