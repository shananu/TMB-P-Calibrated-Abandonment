from __future__ import annotations

import random


def make_rng(seed: int) -> random.Random:
    """Create a dedicated deterministic RNG."""
    return random.Random(seed)


def set_global_seed(seed: int) -> None:
    """Set global random module seed for reproducible behavior."""
    random.seed(seed)
