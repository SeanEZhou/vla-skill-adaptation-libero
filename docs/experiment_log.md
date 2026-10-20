# Experiment log

Chronological record of decisions, screening runs and anything that touches the final comparison.
Final held-out results live in `docs/results.md`; screening numbers here are **not** final results.

## 2026-10-18 — Milestone 0/1

### Experimental design decision

`lerobot/smolvla_libero` was fine-tuned on all 40 tasks of LIBERO-Spatial/Object/Goal/10
(`lerobot/libero`, 1,693 demos). Any task from those four suites is in its training set, so they
offer little headroom and no "new skill" story. `lerobot/smolvla_base` was pretrained on SO-100
community data (6-D state/action, 3 cameras) and cannot drive the 7-D LIBERO controller without a
new action head and normalization: a "frozen base" baseline would only measure the schema mismatch.

**Design:** start from `lerobot/smolvla_libero` (same robot, cameras, state and action interface)
and adapt it to a task from **LIBERO-90**, which is *not* in its training set, using LIBERO-90
expert demonstrations for that task. Both arms use the same checkpoint initialization, adapters and
eval settings. Caveat for the write-up: LIBERO-90 shares scenes/objects with LIBERO-10, so the
target task is "unseen task, related training distribution", not an unseen domain.

### Validation gate (in-distribution)

Before screening, reproduce the checkpoint on libero_object (in its training set) with the published
protocol from lerobot#4614 (tasks 0-9, init states 0-4, `n_action_steps=10`, reported 42/50).
Output: `results/validation/libero_object_n10/`.

### Task screening — pre-registered selection rule (written before any screening result)

* Candidates: 15 LIBERO-90 tasks whose instruction is **unique** within LIBERO-90 (the LeRobot-format
  LIBERO-90 data groups demos by instruction text, so duplicated instructions would mix scenes) and
  **not identical** to any training instruction. IDs: 6, 16, 18, 19, 27, 33, 34, 35, 39, 42, 56, 65,
  71, 76, 87. All have 37-50 demos in `jesbu1/libero_90_lerobot`.
* Screening episodes: 10 **seeded sampled layouts** per task (`--seeds 10000-10009`,
  `init_states=False`). The 50 official init states are not touched during screening; they are the
  frozen final evaluation set.
* Selection rule: among candidates with screening success **<= 30%**, prefer (1) a single-object
  pick-and-place / placement primitive (so failures are attributable), (2) more demonstrations,
  (3) shorter demos (cheaper eval). If none qualifies, take the lowest-success candidate.
* Settings identical to the final eval (`configs/eval.yaml`), baseline checkpoint pinned.

### Positive controls on LIBERO-90 (added after early screening episodes were all failures)

The first 56 screening episodes across 7 candidates were all failures. To separate "novel task" from
"LIBERO-90-specific pipeline problem", run the baseline on LIBERO-90 tasks that are near-duplicates
of training tasks (excluded from selection because their instructions are duplicated):

* libero_90 task 46, LIVING_ROOM_SCENE1 "pick up the alphabet soup and put it in the basket": same
  scene as libero_10 task 7, same object/receptacle as libero_object task 0.
* libero_90 task 20, KITCHEN_SCENE3 "turn on the stove": identical instruction to libero_goal task 7,
  same scene as libero_10 task 2.

Same seeds (10000-10009) and settings. If these are also ~0%, suspect the pipeline, not novelty.
Observation from a failure video (t56, butter -> tray): the arm reaches for the familiar cans and
hovers. Purposeful but wrong target, not degenerate actions. Gripper switches per episode average
15.3 on screening vs 4.3 on validation (dithering).

### Screening result and task selection

Screening (frozen baseline, 10 sampled layouts per task; `results/screening/libero_90/summary.md`,
`results/plots/screening_libero_90.png`): **5/150 = 3% overall**. Non-zero only on *close the
microwave* (3/10) and *put the moka pot on the stove* (2/10), both sub-skills of libero_10 training
tasks (KITCHEN_SCENE6 microwave, KITCHEN_SCENE3/8 moka pots). Every other candidate scored 0/10.

Applying the rule: all 15 are <= 30%. Single-object pick-and-place candidates: 18, 19, 27, 34, 42,
56, 65, 71, 76, 87. Most demos: **56** (50). Shortest demos: also **56** (mean 101 steps).
**Selected: libero_90 task 56, LIVING_ROOM_SCENE3 "pick up the butter and put it in the tray"**
(screening 0/10, 95% CI 0-28%). Frozen in `configs/selected_task.yaml`; final eval set = official
init states 0-49.

Why this is a reasonable skill-acquisition target: butter is a libero_object training object and
baskets are a training receptacle, but the **tray** receptacle and the LIVING_ROOM_SCENE3 layout are
absent from the 40 training tasks. The failure mode (reaching for familiar cans) is a grounding
failure. Claim to make: "not successful under our baseline evaluation", not "absent from pretraining"
(SmolVLA base pretraining data is community SO-100 data and cannot be audited here).

### Positive control result

`results/screening/positive_controls/`: task 46 (alphabet soup -> basket, near-duplicate of training)
**5/10**. Task 20 ("turn on the stove", word-for-word a libero_goal training instruction, but a
different scene layout) **0/10**. The pipeline produces successes on LIBERO-90, so the screening
zeros are about the tasks, not a suite-specific bug. Identical language does not transfer across
scenes for this checkpoint.

## 2026-10-18 — Milestone 2 (dataset pipeline)

* **Built our own dataset from the official HDF5** rather than using the third-party
  `jesbu1/libero_90_lerobot`: replayed through our eval observation path (`scripts/prepare_dataset.py`),
  so train frames equal eval frames bit for bit. The third-party set is used only as a cross-check:
  50/50 episodes match by identical action sequences. Details: `docs/dataset.md`.
* 50/50 demos kept, 5,027 frames. Split by episode: 45 train / 5 val (seed 0). Val is for offline
  loss monitoring only; closed-loop eval stays on the 50 official init states.
* Bug caught by a test: an exact-image alignment test failed on the wrist camera when replaying
  float32 labels. Root cause: 3e-8 float32 rounding of actions shifts arm-mounted camera pixels.
  Dataset confirmed exact (float64 replay is byte-identical on 103/103 frames); test fixed to replay
  originals.
* Frozen-checkpoint offline loss: 4.41 on the target task vs 0.068 on a training task.
* **Normalization finding:** `lerobot-train` silently swaps in the fine-tuning dataset's stats.
  Here that would rescale state inputs up to 10×. Decision (pending review): keep the checkpoint's
  stats in Milestone 3.
* Exit-gate forward pass at batch 8 peaked at 8.0 GB GPU memory (no grad). Training memory on
  12 GB is the main Milestone 3 risk; measure before choosing batch size / freezing.

## 2026-10-19 — Milestone 3 (fine-tuning), arm B1: action expert only

Memory/speed probe (real fwd+bwd+AdamW steps, task-56 data, RTX 4070 SUPER 12 GB, WSL2):

| trainable | params | batch | peak GB | s/step (GPU only) |
|---|---|---|---|---|
| action expert only | 100M | 8 / 32 | 3.1 / 8.3 | 0.29 / 0.60 |
| all but vision encoder | 307M | 16 | 6.4 | 0.43 |
| everything | 393M | 8 / 16 / 32 | 9.6 / 16.5 / OOM | 0.47 / **6.8** / - |

Full fine-tuning at batch 16 "fits" only because the WSL2 driver silently spills to system RAM:
15x slower instead of an OOM. Training memory must be checked against the physical 12 GB, not just
"did it crash".

Decision (user): train the **action expert only** first (SmolVLA's default fine-tuning recipe; VLM
and vision encoder frozen), batch 32, 6,000 steps, checkpoint normalization stats, SmolVLA
optimizer/scheduler presets (`configs/train_expert_only.yaml`, `scripts/train.py`). Full fine-tuning
is optional, only if expert-only is insufficient.

Engineering note: data loading was the bottleneck (175 ms per sample: 2 PNG decodes plus a 50-step
action query); 4 workers gave 1.5 s/step, 8 workers make it GPU-bound.

**Checkpoint selection protocol (pre-registered before any fine-tuned rollout):**
* Candidates: checkpoints at steps 1000, 2000, 4000, 6000.
* Selection set: 20 **seeded sampled layouts**, seeds 20000-20019 (disjoint from screening seeds
  10000-10009 and from the 50 official init states).
* Rule: highest selection-set success. Ties go to the lower held-out-episode validation loss.
* Only the selected checkpoint is evaluated on the frozen set (official init states 0-49, seed
  1000, same `configs/eval.yaml`), plus the forgetting check (libero_object tasks 0-9 x init 0-4,
  paired with the 46/50 baseline).

### Arm B1 results (2026-10-19)

* Training: 6,000 steps, 122 min, 8.3 GB peak, $0 (local). Held-out-demo loss is minimal at step
  ~250 (0.70) and rises to 2.72 by 6,000; train loss reaches 0.041.
* Selection (seeds 20000-20019): step 1000 20/20, 2000 19/20, 4000 20/20, 6000 20/20. Tie broken
  by held-out loss: **step 1000** (1.17 vs 2.49 / 2.72). Config: `configs/finetuned_expert_only.yaml`.
* Frozen eval, task 56, init 0-49: **45/50 (90%)** vs baseline 0/50; McNemar p = 6e-14. Failures
  all `no_grasp` (videos reviewed for all 5).
* Retention, LIBERO-Object tasks 0-9 x init 0-4: **0/50** vs 46/50 before. Catastrophic forgetting
  despite the frozen VLM.
* Data-loading note: inference latency during selection runs was inflated (540-780 ms) by training
  sharing the GPU; the frozen eval was run alone (358 ms).
