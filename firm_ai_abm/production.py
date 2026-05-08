"""Production kernel — STUB at Stage 1. Mode enum only; kernel comes in Stage 2."""
from enum import IntEnum


class Mode(IntEnum):
    H = 0  # human
    A = 1  # augmented human
    T = 2  # automated
