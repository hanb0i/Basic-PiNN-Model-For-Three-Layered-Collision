"""Lightweight logging helpers."""

from __future__ import annotations

from typing import Dict


def format_losses(losses: Dict[str, float]) -> str:
    parts = [f"{k}: {v:.3e}" for k, v in losses.items()]
    return " | ".join(parts)
