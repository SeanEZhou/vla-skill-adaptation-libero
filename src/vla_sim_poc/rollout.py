"""One closed-loop episode: observe -> adapt -> policy -> adapt -> step, until success or timeout."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from .action_adapter import summarize_actions, to_env_action
from .environment import EpisodeSpec, TaskEnv
from .observation_adapter import make_env_preprocessor, raw_to_policy_frame, validate_frame
from .video import viewer_frame


@dataclass
class EpisodeResult:
    key: str
    suite: str
    task_id: int
    task: str
    instruction: str
    init_state_id: int | None
    seed: int
    policy_name: str
    success: bool
    steps: int
    max_steps: int
    termination: str  # success | timeout | env_done
    wall_time_s: float
    n_inferences: int
    inference_latency_mean_s: float
    inference_latency_p95_s: float
    action_stats: dict[str, float] = field(default_factory=dict)
    video_path: str | None = None
    failure_tag: str | None = None  # filled in by manual review

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


ActFn = Callable[[dict[str, Any]], Any]


def run_episode(
    env: TaskEnv,
    spec: EpisodeSpec,
    policy,  # PolicyBundle, or anything with reset() / act(frame) / inference_latencies_s / image_keys
    policy_name: str,
    image_size: int,
    max_steps: int | None = None,
    record_frames: bool = True,
) -> tuple[EpisodeResult, list[np.ndarray], np.ndarray]:
    max_steps = max_steps or env.max_steps
    env_pre = make_env_preprocessor()
    policy.reset()

    t_start = time.perf_counter()
    raw_obs, _ = env.reset(spec)
    frames = [viewer_frame(raw_obs)] if record_frames else []
    raw_actions: list[np.ndarray] = []
    success, termination, steps = False, "timeout", 0

    for steps in range(1, max_steps + 1):
        frame = raw_to_policy_frame(raw_obs, env.task_description, env_pre)
        if steps == 1:
            validate_frame(frame, ["observation.images.image", "observation.images.image2"], image_size)
        env_action = to_env_action(policy.act(frame))
        raw_actions.append(env_action.raw)
        raw_obs, _, terminated, _, info = env.step(env_action.action)
        if record_frames:
            frames.append(viewer_frame(raw_obs))
        if info.get("is_success"):
            success, termination = True, "success"
            break
        if terminated:
            termination = "env_done"
            break

    lat = np.asarray(policy.inference_latencies_s or [np.nan])
    acts = np.asarray(raw_actions)
    result = EpisodeResult(
        key=spec.key,
        suite=spec.suite,
        task_id=spec.task_id,
        task=env.task_name,
        instruction=env.task_description,
        init_state_id=spec.init_state_id,
        seed=spec.seed,
        policy_name=policy_name,
        success=success,
        steps=steps,
        max_steps=max_steps,
        termination=termination,
        wall_time_s=time.perf_counter() - t_start,
        n_inferences=len(policy.inference_latencies_s),
        inference_latency_mean_s=float(np.nanmean(lat)),
        inference_latency_p95_s=float(np.nanpercentile(lat, 95)),
        action_stats=summarize_actions(acts),
    )
    return result, frames, acts
