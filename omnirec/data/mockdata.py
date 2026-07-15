"""Realistic synthetic e-commerce dataset with *learnable* signal.

Generates a product catalog, users with latent category preferences, and click
interactions where a click is more likely when an item matches the user's taste.
Both the text (category words in the title/description) and the image features
(a per-category centroid + noise) carry the category signal, so the multimodal
embeddings cluster by category and the models have something real to learn.

Deterministic given the seed, so training curves, recall and AUC are reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..models import Interaction, Item, User

CATEGORIES = ["electronics", "books", "home", "fashion", "sports", "beauty",
              "toys", "grocery", "automotive", "garden"]
BRANDS = ["acme", "zenith", "orion", "nova", "pulse", "vertex", "lumen", "atlas"]
ADJECTIVES = ["premium", "compact", "wireless", "eco", "durable", "smart",
              "classic", "lightweight", "ergonomic", "deluxe"]
NOUNS = {
    "electronics": ["headphones", "charger", "speaker", "camera", "monitor"],
    "books": ["novel", "guide", "cookbook", "textbook", "anthology"],
    "home": ["lamp", "blanket", "kettle", "organizer", "cushion"],
    "fashion": ["jacket", "sneakers", "watch", "backpack", "scarf"],
    "sports": ["dumbbell", "yoga mat", "bottle", "racket", "tracker"],
    "beauty": ["serum", "lipstick", "cream", "brush", "perfume"],
    "toys": ["puzzle", "blocks", "drone", "figure", "boardgame"],
    "grocery": ["coffee", "granola", "sauce", "tea", "snack"],
    "automotive": ["dashcam", "cleaner", "charger", "cover", "mount"],
    "garden": ["planter", "hose", "trimmer", "gloves", "seeds"],
}


@dataclass
class Dataset:
    items: list[Item]
    users: dict[str, User]
    interactions: list[Interaction]
    user_pref: dict[str, str]        # ground-truth preferred category per user
    item_category: dict[str, str]


def _category_centroids(dim: int, rng: random.Random) -> dict[str, list[float]]:
    return {c: [rng.gauss(0.0, 1.0) for _ in range(dim)] for c in CATEGORIES}


def generate_dataset(n_items: int = 1200, n_users: int = 400, image_dim: int = 64,
                     seed: int = 13, interactions_per_user: int = 20) -> Dataset:
    rng = random.Random(seed)
    centroids = _category_centroids(image_dim, rng)

    items: list[Item] = []
    item_category: dict[str, str] = {}
    by_category: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    for i in range(n_items):
        cat = CATEGORIES[i % len(CATEGORIES)] if i < len(CATEGORIES) else rng.choice(CATEGORIES)
        brand = rng.choice(BRANDS)
        adj = rng.choice(ADJECTIVES)
        noun = rng.choice(NOUNS[cat])
        iid = f"item_{i}"
        title = f"{adj} {brand} {noun}"
        desc = f"a {adj} {noun} for {cat} lovers, {rng.choice(ADJECTIVES)} and reliable"
        # image features = category centroid + noise
        img = [centroids[cat][d] + rng.gauss(0.0, 0.6) for d in range(image_dim)]
        price = round(rng.uniform(5, 500), 2)
        items.append(Item(id=iid, title=title, description=desc, category=cat,
                          brand=brand, price=price, image_features=img, tags=[cat, brand]))
        item_category[iid] = cat
        by_category[cat].append(iid)

    users: dict[str, User] = {}
    user_pref: dict[str, str] = {}
    interactions: list[Interaction] = []
    for u in range(n_users):
        uid = f"user_{u}"
        pref = rng.choice(CATEGORIES)
        second = rng.choice([c for c in CATEGORIES if c != pref])
        user_pref[uid] = pref
        liked_pool = by_category[pref] + by_category[second]
        history = rng.sample(liked_pool, k=min(8, len(liked_pool)))
        users[uid] = User(id=uid, history=history, context={"pref": pref})

        ts = 0.0
        for _ in range(interactions_per_user):
            ts += 1.0
            if rng.random() < 0.6:
                iid = rng.choice(liked_pool)
                clicked = 1 if rng.random() < 0.85 else 0
            else:
                iid = f"item_{rng.randrange(n_items)}"
                clicked = 1 if item_category[iid] in (pref, second) and rng.random() < 0.7 else 0
            interactions.append(Interaction(user_id=uid, item_id=iid, clicked=clicked, ts=ts))

    return Dataset(items=items, users=users, interactions=interactions,
                   user_pref=user_pref, item_category=item_category)
