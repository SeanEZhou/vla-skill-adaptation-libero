"""YAML config loading. Configs are merged left to right (later files override earlier keys)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(*paths: str | Path) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for p in paths:
        p = Path(p)
        if not p.is_absolute() and not p.exists():
            p = CONFIG_DIR / p
        with open(p) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def parse_id_list(spec: str | list[int]) -> list[int]:
    """Parse "0-9,20,30-32" (inclusive ranges) or a list into sorted unique ints."""
    if isinstance(spec, list):
        return sorted({int(x) for x in spec})
    ids: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(part))
    return sorted(ids)
