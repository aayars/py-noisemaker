"""Shared seed data for parity tests.

This module centralizes the random seeds used across the parity test suite so
that the number of exercised seeds can be tuned in a single location.  The
tests currently use a reduced subset of five seeds to keep the cross-language
checks fast enough for routine execution while still providing coverage across
multiple randomly selected states.
"""

from __future__ import annotations

__all__ = ["ALL_PARITY_SEEDS", "PARITY_SEEDS", "LONG_SEQUENCE_SEEDS"]


# 20 randomly chosen 32-bit seeds retained for future adjustments if the test
# suite needs to exercise a larger sample again.
ALL_PARITY_SEEDS: list[int] = [
    3626764237,
    1654615998,
    3255389356,
    3823568514,
    1806341205,
    173879092,
    1112038970,
    4146640122,
    2195908194,
    2087043557,
    1739178872,
    3943786419,
    3366389305,
    3564191072,
    1302718217,
    4156669319,
    2046968324,
    1537810351,
    2505606783,
    3829653368,
]

# Limit parity runs to the first five seeds to keep the test suite fast.  Tests
# that need even fewer seeds can slice this list as needed.
PARITY_SEEDS: list[int] = ALL_PARITY_SEEDS[:5]

# Convenience subset for tests that exercise long-running parity sequences.
LONG_SEQUENCE_SEEDS: list[int] = PARITY_SEEDS[:3]

