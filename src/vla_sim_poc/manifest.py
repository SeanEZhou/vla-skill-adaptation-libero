"""Environment manifest saved next to every result, so numbers can be traced to exact versions."""

from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

PACKAGES = [
    "lerobot", "torch", "torchvision", "transformers", "mujoco", "robosuite", "hf-libero", "bddl",
    "gymnasium", "numpy", "huggingface_hub", "datasets", "torchcodec", "safetensors",
]

# Hub repos whose content affects results. Assets are fetched unpinned by hf-libero at first use,
# so we record the revision that was current when the cache was populated.
HUB_REFERENCES = {
    "lerobot/libero-assets (LIBERO meshes/textures)": "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6",
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct (tokenizer/VLM config)": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
}


def redact_home(path: str | os.PathLike) -> str:
    """Replace the user's home directory with ``~`` so saved records don't leak local usernames."""
    p, home = str(path), os.path.expanduser("~")
    return "~" + p[len(home):] if p == home or p.startswith(home + os.sep) else p


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _lerobot_source() -> Any:
    try:
        return json.loads(md.distribution("lerobot").read_text("direct_url.json") or "null")
    except Exception:
        return None


def _gpu() -> dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "name": torch.cuda.get_device_name(0),
                "cuda_runtime": torch.version.cuda,
                "total_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
                "driver": subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True, text=True,
                ).stdout.strip(),
            }
    except Exception as e:  # pragma: no cover
        return {"error": repr(e)}
    return {"name": None}


def build_manifest() -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: _version(p) for p in PACKAGES},
        "lerobot_source": _lerobot_source(),
        "gpu": _gpu(),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "hub_references": HUB_REFERENCES,
    }
