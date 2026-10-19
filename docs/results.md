# Results

## Arm A: frozen baseline (Milestone 1)

**Task:** `libero_90` task 56, LIVING_ROOM_SCENE3, *"pick up the butter and put it in the tray"*
(selected by the pre-registered rule; see `docs/experiment_log.md`).
**Policy:** `lerobot/smolvla_libero@a24f1f38`, unchanged weights.
**Eval set:** the 50 official LIBERO init states (0-49), env seed 1000, 400-step limit. None of these
states was used for screening, and no demo starts at one of them (`replay_demos.py` leakage check).

| Metric | Value |
|---|---|
| **Success** | **0/50 = 0.0% (Wilson 95% CI 0.0%-7.1%)** |
| Terminations | 50/50 timeout at 400 steps |
| Inference settings | chunk_size 50, **n_action_steps 10**, 10 flow-matching steps, bf16 weights, CUDA, 256x256 cameras, 20 Hz |
| Chunk inference latency | mean 393 ms, p95 579 ms (RTX 4070 SUPER, WSL2, run alone) |
| Eval cost | 23.9 min wall total for 50 episodes (28.6 s/episode), local GPU, $0 |
| Arm saturation (any of dims 0-5 outside [-1,1]) | 0.02% of 20,000 steps |
| Gripper sign switches per episode | mean 26.6 (min 8, max 65); in-distribution validation episodes average 4.3 |

Per-episode records: `results/baseline/libero_90_t56/episodes.jsonl`. Videos (every episode):
`results/baseline/libero_90_t56/videos/`. Raw actions: `.../actions/*.npy`. Run manifest:
`.../run_config_*.json`.

### Failure analysis (preliminary, 6 of 50 episodes reviewed by eye)

Init states 0, 7, 13, 21, 34 and 46, all tagged **`missed_object`**. The arm leaves home and moves
to the back-right of the table, where the **ketchup and alphabet soup** stand (libero_object training
objects, simulator positions around x≈-0.11..-0.23, y≈-0.17). It never approaches the butter at
(0.035, 0.057) or the tray at y=0.26, then hovers there with the gripper toggling. Actions are not
degenerate (no saturation, mean |Δpos| ≈ 0.19). The failure is **target grounding**: the policy acts
out the familiar "pick a can" behaviour from its training distribution instead of following the
instruction. In our failure taxonomy this is *model generalization* (perception/language
grounding), not an action-interface or control problem: the same interface reaches 92% in
distribution.

Supporting context (screening, not final results):
* 15 LIBERO-90 candidates: 5/150 (3%). Successes only on sub-skills of libero_10 training tasks.
* Positive controls: near-duplicate training task (alphabet soup -> basket) 5/10. Word-for-word
  training instruction in a different scene ("turn on the stove") 0/10.
* In-distribution validation (libero_object): 46/50 = 92% (81-97%).

**Interpretation:** under our evaluation the checkpoint does not perform this skill (0/50, true rate
<= 7.1% at 95% confidence). This does **not** show the skill is absent from pretraining. It is the
starting point for Milestone 3: can 50 expert demos move this number?
