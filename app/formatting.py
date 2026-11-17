"""Readable event counts for compact dashboard cards."""

from __future__ import annotations


def format_count(value: int) -> str:
    """Use Indian large-number units; exact counts remain in tables/tooltips."""
    for scale, unit in ((10_000_000, "crore"), (100_000, "lakh"), (1_000, "thousand")):
        if abs(value) >= scale:
            number = f"{value / scale:.2f}".rstrip("0").rstrip(".")
            return f"{number} {unit}"
    return str(value)
