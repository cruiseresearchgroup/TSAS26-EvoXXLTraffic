"""Versioned, observation-aware benchmark primitives; legacy caches are not accepted."""
from .data import prepare_year, load_year, save_year, window_batch, split_origins
from .identity import transition
from .metrics import score
