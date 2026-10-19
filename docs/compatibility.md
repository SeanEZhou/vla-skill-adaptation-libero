# Compatibility audit (Milestone 0)

Audited 2026-10-18. Every claim below points at a file, a script output in `results/`, or upstream
source / issues. Re-run the evidence with the commands in the last section.

## 1. Chosen integration path

| Component | Pinned version / revision | Notes |
|---|---|---|
| LeRobot | **v0.6.1**, commit `7e241bd630a3719a56157a497ce5d08f244784f1` | latest release; provides `LiberoEnv`, `LiberoProcessorStep`, SmolVLA, `lerobot-train` |
| Python | 3.12.12 (uv-managed) | LeRobot requires >=3.12 |
| PyTorch | 2.11.0+cu128 | CUDA available, RTX 4070 SUPER 12 GB, driver 610.88 (WSL2) |
| MuJoCo | **3.3.7** (explicit pin) | >=3.4.0 breaks stored LIBERO init states (lerobot#4390), and hf-libero allows `<3.9` |
| robosuite | 1.4.0 | pinned by hf-libero 0.1.4 |
| LIBERO | hf-libero 0.1.4 | HF-maintained LIBERO fork |
| LIBERO assets | `lerobot/libero-assets@0b3ea86b` | **downloaded unpinned at first use** by hf-libero; revision recorded in `manifest.py` |
| Checkpoint | **`lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d`** | "Fix LIBERO checkpoint feature metadata", 2026-10-12 |
| VLM backbone | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b` | tokenizer/config pulled at load time |
| Training data of checkpoint | `lerobot/libero@a1aaacb7` | 1,693 episodes, 40 tasks (Spatial/Object/Goal/10) |
| Candidate target-task data | `jesbu1/libero_90_lerobot@fa5ab6b7` (LeRobot v2.0), `yifengzhu-hf/LIBERO-datasets@f13aa24a` (original HDF5) | for Milestone 2 |

Lockfile: `uv.lock`. Every eval output dir contains `run_config_*.json` with the full manifest
(package versions, LeRobot source commit, GPU, config, checkpoint revision).

**Rendering:** `MUJOCO_GL=egl`. Under WSL2, `/dev/dri/renderD128` is not accessible, so Mesa falls
back to the `kms_swrast` software rasterizer. This is fast enough (~30 sim steps/s with two 256x256
cameras), but it is not the GPU EGL path the training data was rendered with (see section 6).

## 2. Checkpoint selection and the alternatives rejected

| Checkpoint | Verdict | Evidence |
|---|---|---|
| `lerobot/smolvla_base` | **Not usable as a LIBERO baseline.** | SO-100 pretraining: state 6-D, action 6-D, 3 cameras (`camera1..3`). LIBERO needs a 7-D action, 8-D state and 2 cameras, and its normalization stats belong to SO-100. A zero-shot rollout would only measure the schema mismatch (our action adapter rejects 6-D actions, `tests/test_action_adapter.py`). |
| `HuggingFaceVLA/smolvla_libero` | Not chosen | `expert_width_multiplier=0.5` and `num_vlm_layers=0` (all 32 layers) differ from the paper's 0.75 / 16 (lerobot#2354), and it ships `n_action_steps=1`. |
| **`lerobot/smolvla_libero`** | **Chosen** | Same interface as LIBERO (8-D state, 7-D action, 2 cameras), paper architecture (16 VLM layers, 0.75 expert width), `pretrained_path=lerobot/smolvla_base`, trained 25k steps x batch 32 on `lerobot/libero`. Reproduced at 92% on libero_object (section 5). |

**Consequence for the experiment:** `smolvla_libero` has seen all 40 Spatial/Object/Goal/10 tasks,
so the target task comes from **LIBERO-90**, which is outside its training set. See
`docs/experiment_log.md`.

## 3. Observation schema (env -> policy)

Verified by `scripts/inspect_env.py` (`results/smoke/env_schema.json`) and unit tests.

| Stage | Key | Shape / dtype | Meaning |
|---|---|---|---|
| raw `LiberoEnv` obs | `pixels/image` | 256x256x3 uint8, **upside down** | `agentview_image` (third person) |
| | `pixels/image2` | 256x256x3 uint8, upside down | `robot0_eye_in_hand_image` (wrist) |
| | `robot_state/eef/pos` | (3,) float64 | EEF position, robot base / world frame, metres |
| | `robot_state/eef/quat` | (4,) float64, **(x,y,z,w)** | EEF orientation |
| | `robot_state/gripper/qpos` | (2,) float64 | finger joint positions (+-0.04 open) |
| after `preprocess_observation` + `LiberoProcessorStep` | `observation.images.image`, `.image2` | (1,3,256,256) float32 in [0,1], **rotated 180 deg (upright)** | matches dataset orientation (`results/smoke/dataset_vs_env_libero_object_t0.png`) |
| | `observation.state` | (1,8) float32 | `[eef_pos(3), axis_angle(3), gripper_qpos(2)]` |
| after policy preprocessor | `observation.images.camera1`, `.camera2` | same | **renamed** (checkpoint's `rename_map`) |
| | `observation.state` | (1,8) | MEAN_STD-normalized with the checkpoint stats |
| | `task` -> tokens | max_length 48, right padded | SmolVLM2 tokenizer; newline appended by `smolvla_new_line_processor` |
| inside SmolVLA | images | resized with padding to 512x512, scaled to [-1,1] | `resize_imgs_with_padding=[512,512]`; state/actions zero-padded to 32-D |

* **Axis-angle convention:** `LiberoProcessorStep._quat2axisangle` matches robosuite's
  `quat2axisangle` exactly, including its non-canonical output for w<0 (angle in (pi, 2pi]). That is
  the function the dataset states were produced with (`tests/test_observation_adapter.py`). The
  orientation sits near the +-pi wrap (state dim 3 mean is 2.97), which explains dim 4's large std
  (0.91) in the normalization stats. This is inherited from the data, not introduced by us.
* **State frame check:** frame-0 `observation.state` of 10 `lerobot/libero` demos for
  libero_object task 0 versus the env after reset on 10 official init states differ by at most
  7.6 mm / 2.5 mrad in the mean, with per-dimension std of 4-10 mm
  (`results/smoke/dataset_check_libero_object_t0.json`). Same frame, same units.
* **Normalization stats** in the checkpoint equal `lerobot/libero`'s `meta/stats.json` to within
  4e-8 (`results/smoke/checkpoint_schema.json`). The pre- and post-processor share the same stats.

## 4. Action schema (policy -> env)

| Dim | Meaning | Units after controller scaling |
|---|---|---|
| 0-2 | EEF position delta (dx, dy, dz) | x 0.05 m per step |
| 3-5 | EEF orientation delta (axis-angle) | x 0.5 rad per step |
| 6 | gripper | -1 open, +1 close (demo labels are exactly +-1; every demo starts at -1) |

* Controller: robosuite `OSC_POSE`, `control_delta=True` (`control_mode=relative`), `kp=150`,
  input clipped to [-1,1] (`results/smoke/env_schema.json`; `env_args` in the LIBERO HDF5 confirm
  the same controller config was used for the demonstrations).
* **Control frequency 20 Hz.** Sanity check: 10 steps of `dz=+1` raise the EEF by 11.7 cm (upper
  bound 50 cm; OSC tracks with lag), and 10 no-ops drift 0.6 mm.
* Policy output: MEAN_STD-unnormalized by the postprocessor, so values arrive in controller
  units. `action_adapter.to_env_action` checks shape (7,) and finiteness, clips to [-1,1] and records
  the clip rate.
* **Chunking:** `chunk_size=50` predicted per inference, `n_action_steps=10` executed before
  replanning (see D3), `num_steps=10` flow-matching denoising steps.

## 5. Gates

| # | Gate | Status | Evidence |
|---|---|---|---|
| 1 | Versions pinned | PASS | section 1, `uv.lock`, run manifests |
| 2 | Checkpoint + immutable revision | PASS | `a24f1f38...`, local snapshot loaded by path |
| 3 | Cameras / sizes / order / channels | PASS | section 3, `dataset_vs_env_*.png`, `test_policy_interface.py` |
| 4 | State dims / units / normalization | PASS | section 3 frame-0 comparison, stats diff < 1e-7 |
| 5 | Action dims / semantics | PASS | section 4 |
| 6 | Chunk horizon / replanning / frequency / clipping | PASS | `configs/eval.yaml`, D3, D4 |
| 7 | Dataset schema == policy input schema | PASS for `lerobot/libero`. LIBERO-90 data (different keys) is a Milestone 2 item. | `dataset_check_*.json` |
| 8 | Success detection with known-good expert rollouts | PASS | `scripts/replay_demos.py`: 48/50 original HDF5 demos succeed when replayed open-loop (2 drift); the no-op negative control never succeeds (`results/smoke/replay/replay_libero_object_t00.json`) |
| 9 | Single episode reset/render/step/record | PASS | `results/smoke/render_*.png`, `results/smoke/rollout_libero_object/videos/` |
| 10 | Non-degenerate actions, full rollout, deterministic seeds | PASS | 2/2 smoke rollouts succeed; seeded resets reproduce layouts exactly (same seed gives identical qpos, different seeds move objects ~2 cm) |
| — | **In-distribution reproduction** | **PASS: 46/50 = 92% (95% CI 81-97%)** on libero_object tasks 0-9 x init states 0-4 | `results/validation/libero_object_n10/`. lerobot#4614 reports 42/50 for the same protocol; the CIs overlap. |

## 6. Discrepancies found (with evidence) and how they are handled

| ID | Discrepancy | Evidence | Handling |
|---|---|---|---|
| D1 | Checkpoint expects `observation.images.camera1/2`; env emits `image/image2`. | `config.json` input features vs `LiberoEnv` `camera_name_mapping` | The checkpoint's saved preprocessor holds the rename map, **but lerobot-eval 0.6.1 replaces it with the empty CLI default** (lerobot#4578). We pass it explicitly and `load_policy` asserts it equals the saved one. Without it SmolVLA raises "All image features are missing" (`test_missing_rename_fails_loudly`), so it fails loudly rather than silently. |
| D2 | Env renders at **360x360** by default (`LiberoEnv` config); training data and checkpoint features are **256x256**. | `envs/configs.py`, `train_config.json` env section | We render at 256. Untested hypothesis: this may explain part of our 92% vs #4614's 84% (which used the 360 default). |
| D3 | Checkpoint ships `n_action_steps=50` (= chunk_size). | `config.json`; SmolVLA paper Table 13; lerobot#4614 (-20 pp) | Override to **10** for all arms. Recorded as `shipped_n_action_steps` in run configs. |
| D4 | Rate metadata: `lerobot/libero` says `fps=10`, checkpoint `train_config.json` says `env.fps=30`, simulator runs at 20 Hz. Since LeRobot #4124, `env.fps` really sets `control_freq`. | `meta/info.json`, `train_config.json` | `control_freq: 20` is hard-set in `configs/eval.yaml`. Delta actions make this critical (#4614: fps 10 or 30 gives ~0% success). |
| D5 | MuJoCo >=3.4.0 changes box-box contacts, so the libero_spatial task-5 bowl no longer settles onto the ramekin. | lerobot#4390 | Pinned `mujoco==3.3.7`. |
| D6 | LIBERO's first import prompts interactively for a dataset path, and assets are fetched from the Hub unpinned. | `libero/__init__.py`, `download_utils.py` | Project-local `LIBERO_CONFIG_PATH=.libero`; asset revision recorded. |
| D7 | Weights run in **bf16** (SmolVLM backbone dtype) although the config says `use_amp=false`. | `param_dtypes` in run config | Recorded and identical across arms; bf16 autocast gives no speedup here. |
| D8 | Renderer: our frames come from software EGL; dataset frames look slightly lighter and softer (AV1 compression and/or different renderer). | `dataset_vs_env_libero_object_t0.png` | Minor visual domain gap, identical for both arms. Validation shows it does not break the in-distribution policy. |
| D9 | `lerobot-eval` couples init-state selection to the vector-env batch layout (lerobot#2850, #4152). | upstream issues | Our `evaluate.py` names each episode's init state explicitly. It reuses LeRobot's env and processors, so only the loop is ours. |
| D10 | Demo `states[0]` are mid-settle (non-zero velocity, time 0.25 s); official init states are settled snapshots. | `replay_demos.py` | Leakage is compared on qpos only: demo starts are >=0.086 from any official init state, while official states are a median 0.036 apart from each other, so no eval init state appears in the demos. |

## 7. Open integration questions (status after M0/M1)

1. **LeRobot revision with LIBERO eval + SmolVLA:** v0.6.1 (`7e241bd`). Both are native; verified end to end.
2. **Checkpoint that demonstrably runs:** `lerobot/smolvla_libero@a24f1f38` on LIBERO-Object
   (92%, 50 episodes) under the settings above.
3. **Features and normalization:** sections 3-4. MEAN_STD for state and action using `lerobot/libero` stats, identity for images.
4. **LIBERO demos -> LeRobot without losing time alignment:** *Milestone 2.* Plan: use the LeRobot
   LIBERO-90 conversion (OpenVLA regeneration: 256x256, no-op actions removed, failed replays
   dropped, images upright) and verify it against the original HDF5 (state/action match, frame
   alignment) before training.
5. **Task with low baseline and enough expert data:** chosen by pre-registered screening; see `docs/experiment_log.md`.
6. **Cheapest training setup:** *Milestone 3.* Inference uses ~1 GB VRAM. Full fine-tuning of the
   450M model with batch 32 likely fits in 12 GB with bf16 and a frozen vision encoder; this must be measured.

## 8. Remaining reproducibility risks

* MuJoCo behaviour differs from 3.4 onward (D5): never upgrade without re-running the validation gate.
* hf-libero assets are unpinned upstream; a changed mesh would silently change scenes.
* Software rendering on WSL vs GPU EGL elsewhere: re-run validation on any new machine (cloud GPU).
* The 50 official init states per task cap the final eval at 50 episodes per task.
* Inference latency (~340 ms per chunk on WSL2, GPU ~40% utilised: launch-bound) is a property of
  this machine, not of the model.

## Reproduce the evidence

```bash
uv sync
uv run pytest -q                                   # unit + checkpoint integration tests
uv run python scripts/inspect_env.py               # schemas + PNGs -> results/smoke/
uv run python scripts/inspect_checkpoint.py        # checkpoint features/stats -> results/smoke/
uv run python scripts/inspect_dataset.py           # dataset vs env frame-0 cross-check
uv run python scripts/replay_demos.py --suite libero_object --task-id 0 \
    --hdf5 libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
uv run python scripts/evaluate.py --suite libero_object --task-ids 0-9 --init-states 0-4 \
    --out results/validation/libero_object_n10     # in-distribution reproduction
```
