"""Thin wrapper around LeRobot's ``LiberoEnv`` for single-episode, explicitly-indexed rollouts.

Why not ``lerobot-eval`` directly? It ties init-state selection to the vector-env batch layout
(lerobot#2850/#4152), which makes it awkward to say exactly which initial state each episode used.
Here every episode names its initial condition explicitly:

* ``init_state_id`` (0..49): one of LIBERO's official stored initial states for the task. Used for
  the frozen final evaluation set.
* ``seed`` with ``init_state_id=None``: a fresh object layout sampled by LIBERO's placement
  initializer, deterministic in the seed. Used for task screening so the official states stay
  untouched until the final comparison.

The environment itself (observation formatting, settle steps, success check) is LeRobot's code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Must be set before mujoco/robosuite import. Project-local LIBERO config avoids an interactive prompt.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", os.path.join(os.path.dirname(__file__), "..", "..", ".libero")
)

from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv, _get_suite  # noqa: E402

# Camera keys exposed by LiberoEnv (before LeRobot's ``observation.images.`` prefix is added).
CAMERA_NAMES = "agentview_image,robot0_eye_in_hand_image"


@dataclass(frozen=True)
class EpisodeSpec:
    suite: str
    task_id: int
    seed: int
    init_state_id: int | None = None  # None -> sampled layout from ``seed``

    @property
    def key(self) -> str:
        init = f"init{self.init_state_id:02d}" if self.init_state_id is not None else f"seed{self.seed}"
        return f"{self.suite}_t{self.task_id:02d}_{init}"


class TaskEnv:
    """One LIBERO task; resets to an explicitly chosen initial condition each episode."""

    def __init__(self, suite: str, task_id: int, env_cfg: dict[str, Any], use_init_states: bool):
        self.suite_name = suite
        self.task_id = task_id
        self.use_init_states = use_init_states
        suite_obj = _get_suite(suite)
        self.env = LiberoEnv(
            task_suite=suite_obj,
            task_id=task_id,
            task_suite_name=suite,
            episode_length=env_cfg.get("episode_length"),
            camera_name=CAMERA_NAMES,
            obs_type="pixels_agent_pos",
            observation_width=env_cfg["image_size"],
            observation_height=env_cfg["image_size"],
            init_states=use_init_states,
            num_steps_wait=env_cfg["num_steps_wait"],
            control_freq=env_cfg["control_freq"],
            control_mode=env_cfg["control_mode"],
            hard_reset=env_cfg.get("hard_reset", True),
        )
        self.task_description: str = self.env.task_description
        self.task_name: str = self.env.task
        self.max_steps: int = self.env._max_episode_steps

    @property
    def n_init_states(self) -> int:
        return len(self.env._init_states) if self.env._init_states is not None else 0

    def reset(self, spec: EpisodeSpec):
        if (spec.init_state_id is not None) != self.use_init_states:
            raise ValueError(f"{spec} does not match env mode use_init_states={self.use_init_states}")
        if spec.init_state_id is not None:
            if not 0 <= spec.init_state_id < self.n_init_states:
                raise ValueError(f"init_state_id {spec.init_state_id} out of range [0, {self.n_init_states})")
            # LiberoEnv advances init_state_id after each reset; pin it so the episode is explicit.
            self.env.init_state_id = spec.init_state_id
        return self.env.reset(seed=spec.seed)

    def step(self, action):
        return self.env.step(action)

    def check_success(self) -> bool:
        return bool(self.env._env.check_success())

    @property
    def sim(self):
        return self.env._env.env.sim

    def close(self):
        self.env.close()


def default_max_steps(suite: str) -> int:
    return TASK_SUITE_MAX_STEPS.get(suite, 500)
