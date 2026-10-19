"""Load a SmolVLA checkpoint plus its own pre/post-processors at an immutable revision.

Inference path for one control step (what ``PolicyBundle.act`` does):

    frame (from observation_adapter)
      -> preprocessor: rename cameras -> add batch -> tokenize instruction -> to device
                       -> normalize state (MEAN_STD, checkpoint stats)
      -> policy.select_action: if the action queue is empty, run the VLM + action expert
                       (10 flow-matching steps) to predict a 50-step chunk and enqueue the first
                       ``n_action_steps``; then pop one action
      -> postprocessor: un-normalize action (MEAN_STD) -> to CPU
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from .manifest import redact_home

ACTION_KEY = "action"


@dataclass
class PolicyBundle:
    policy: SmolVLAPolicy
    preprocessor: Any
    postprocessor: Any
    checkpoint_dir: Path
    source: dict[str, Any]
    last_call_inferred: bool = False
    inference_latencies_s: list[float] = field(default_factory=list)

    @property
    def config(self):
        return self.policy.config

    @property
    def image_keys(self) -> list[str]:
        return list(self.policy.config.image_features)

    def reset(self) -> None:
        self.policy.reset()
        self.inference_latencies_s = []

    @torch.inference_mode()
    def act(self, frame: dict[str, Any]) -> torch.Tensor:
        batch = self.preprocessor(frame)
        will_infer = len(self.policy._queues[ACTION_KEY]) == 0
        if will_infer and batch["observation.state"].is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        action = self.policy.select_action(batch)
        if will_infer:
            if action.is_cuda:
                torch.cuda.synchronize()
            self.inference_latencies_s.append(time.perf_counter() - t0)
        self.last_call_inferred = will_infer
        return self.postprocessor(action)


def resolve_checkpoint(repo_id_or_path: str, revision: str | None) -> Path:
    p = Path(repo_id_or_path)
    if p.is_dir():
        return p
    return Path(snapshot_download(repo_id_or_path, revision=revision))


def load_policy(ckpt_cfg: dict[str, Any], policy_cfg: dict[str, Any]) -> PolicyBundle:
    ckpt_dir = resolve_checkpoint(ckpt_cfg["repo_id"], ckpt_cfg.get("revision"))
    cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    shipped_n_action_steps = cfg.n_action_steps
    cfg.n_action_steps = int(policy_cfg["n_action_steps"])
    cfg.num_steps = int(policy_cfg["num_steps"])
    cfg.device = policy_cfg["device"]
    cfg.use_amp = bool(policy_cfg.get("use_amp", False))
    cfg.pretrained_path = str(ckpt_dir)

    rename_map = dict(ckpt_cfg.get("rename_map") or {})
    saved_rename = saved_rename_map(ckpt_dir)
    if saved_rename != rename_map:
        raise ValueError(
            f"Configured rename_map {rename_map} differs from the checkpoint's saved one {saved_rename}. "
            "Fix configs rather than guessing which is right."
        )

    policy = SmolVLAPolicy.from_pretrained(ckpt_dir, config=cfg)
    policy.to(cfg.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(ckpt_dir),
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    source = {
        "repo_id": ckpt_cfg["repo_id"],
        "revision": ckpt_cfg.get("revision"),
        "local_dir": redact_home(ckpt_dir),
        "shipped_n_action_steps": shipped_n_action_steps,
        "n_action_steps": cfg.n_action_steps,
        "chunk_size": cfg.chunk_size,
        "num_steps": cfg.num_steps,
        "use_amp": cfg.use_amp,
        # SmolVLM backbone loads in bf16; record what actually ran rather than what the config implies.
        "param_dtypes": sorted({str(p.dtype) for p in policy.parameters()}),
        "device": cfg.device,
        "rename_map": rename_map,
    }
    return PolicyBundle(policy, preprocessor, postprocessor, ckpt_dir, source)


def saved_rename_map(ckpt_dir: Path) -> dict[str, str]:
    pre = json.loads((Path(ckpt_dir) / "policy_preprocessor.json").read_text())
    for step in pre["steps"]:
        if step.get("registry_name") == "rename_observations_processor":
            return dict(step["config"].get("rename_map") or {})
    return {}
