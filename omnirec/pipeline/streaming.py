"""Streaming feature-engineering pipeline (Bytewax / Flink-style dataflow).

Consumes a stream of user interaction events and, for each click, updates the
user's embedding in the online feature store in real time. Modelled as a small
dataflow: ``source -> filter(click) -> map(lookup item vec) -> update store``.

Offline this drains an in-memory queue synchronously with identical semantics to
a distributed streaming job; a Kafka source (:class:`KafkaEventSource`) plugs in
for production without changing the operator logic.
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from ..feature_store.store import FeatureStore
from ..logging_setup import get_logger
from ..models import Interaction

log = get_logger("streaming")


@dataclass
class StreamStats:
    consumed: int = 0
    clicks_applied: int = 0
    skipped: int = 0

    def to_dict(self) -> dict:
        return {"consumed": self.consumed, "clicks_applied": self.clicks_applied,
                "skipped": self.skipped}


class EventQueue:
    """Thread-safe in-memory event source (stands in for a Kafka topic)."""

    def __init__(self) -> None:
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()

    def publish(self, event: dict) -> None:
        self._q.put(json.dumps(event))

    def size(self) -> int:
        return self._q.qsize()

    def drain(self):
        while True:
            try:
                raw = self._q.get_nowait()
            except queue.Empty:
                return
            if raw is None:
                return
            yield json.loads(raw)


class StreamProcessor:
    def __init__(self, store: FeatureStore, item_vector_fn: Callable[[str], Optional[list]],
                 base_vector_fn: Optional[Callable[[str], Optional[list]]] = None,
                 click_lr: float = 0.35) -> None:
        self.store = store
        self.item_vector_fn = item_vector_fn
        self.base_vector_fn = base_vector_fn
        self.click_lr = click_lr
        self.stats = StreamStats()

    def process(self, evt: dict) -> Optional[str]:
        self.stats.consumed += 1
        if int(evt.get("clicked", 0)) != 1:
            self.stats.skipped += 1
            return None
        item_id = evt.get("item_id")
        item_vec = self.item_vector_fn(item_id) if item_id else None
        if not item_vec:
            self.stats.skipped += 1
            return None
        base = self.base_vector_fn(evt["user_id"]) if self.base_vector_fn else None
        self.store.apply_click(evt["user_id"], item_id, item_vec, lr=self.click_lr, base_vector=base)
        self.stats.clicks_applied += 1
        return evt["user_id"]

    def process_interaction(self, ix: Interaction) -> Optional[str]:
        return self.process({"user_id": ix.user_id, "item_id": ix.item_id, "clicked": ix.clicked})

    def run(self, source) -> StreamStats:
        for evt in source.drain() if hasattr(source, "drain") else source:
            self.process(evt)
        return self.stats


class KafkaEventSource:  # pragma: no cover - requires kafka
    def __init__(self, brokers: str, topic: str = "user-events", group: str = "omnirec") -> None:
        from kafka import KafkaConsumer  # type: ignore

        self._consumer = KafkaConsumer(topic, bootstrap_servers=brokers, group_id=group,
                                       value_deserializer=lambda b: json.loads(b.decode()))

    def drain(self):
        for msg in self._consumer:
            yield msg.value
