"""Config loading + world-frame constants (single source of truth).

Everything geometric is expressed in the CLAUDE.md world frame:
right-handed metres, origin at the centre of the floor, +X along the 51.73 m
length, +Y across the 26.3 m depth, +Z up, floor = Z=0.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path = REPO_ROOT

    # ---- convenience accessors -------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve a path from config.paths.<key> against the repo root."""
        rel = self.raw["paths"][key]
        return (self.root / rel).resolve()

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    # ---- world frame ------------------------------------------------------
    @property
    def floor_length_m(self) -> float:
        return float(self.raw["world"]["floor_length_m"])

    @property
    def floor_depth_m(self) -> float:
        return float(self.raw["world"]["floor_depth_m"])

    @property
    def floor_extent(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) of the floor in metres (origin = centre)."""
        hx = self.floor_length_m / 2.0
        hy = self.floor_depth_m / 2.0
        return (-hx, hx, -hy, hy)

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 0))


def load_config(path: str | os.PathLike | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, "r") as f:
        raw = yaml.safe_load(f)
    cfg = Config(raw=raw, root=p.resolve().parent)
    return cfg


def ensure_dirs(cfg: Config) -> None:
    """Create the artifact directories that the pipeline writes to."""
    for key in ("artifacts", "frames", "matches"):
        cfg.path(key).mkdir(parents=True, exist_ok=True)
