# Experiment Summary

## Stage 1 — Parallel Columns
- **Goal:** Train Task 1 and Task 2 simultaneously with coarse mask splitting the SNN into two columns.
- **Command:** `python scripts/run_experiment.py --config configs/stage1_parallel.yaml`
- **Outcome:** Single training stage reached 99.69 % accuracy on Task 1 and 96.25 % on Task 2, showing that column-wise masking can support clean co-training without interference.

## Stage 1 — Sequential Columns
- **Goal:** Train Task 1 then Task 2 sequentially while measuring forgetting without replay.
- **Command:** `python scripts/run_experiment.py --config configs/stage1_sequential.yaml`
- **Key Stages:**
  - Initial (cold start): Task 1 = 47.69 %, Task 2 = 51.19 %.
  - After Task 1: Task 1 = 99.69 %, Task 2 = 51.19 % (Task 2 untouched).
  - After Task 2: Task 1 = 99.69 %, Task 2 = 96.13 % (no forgetting thanks to masking).

## Stage 2 — Shared Columns with Held-Out Tasks
- **Goal:** Reuse the Task 1/2 columns to learn the held-out Task 1′/2′ sequentially; no replay; measure overwrite.
- **Command:** `python scripts/run_experiment.py --config configs/stage2_shared_columns.yaml`
- **Observations:** When Task 1′/2′ are trained, they overwrite their parent columns, dropping Task 1 accuracy to 57.25 % and Task 2 to 92.12 %, while the primed tasks reach ~98–100 %. This quantifies catastrophic interference when columns are shared.

## Stage 2 — Disjoint Columns
- **Goal:** Give Task 1′/2′ dedicated columns to see if parent accuracy remains stable.
- **Command:** `python scripts/run_experiment.py --config configs/stage2_disjoint_columns.yaml`
- **Final Stage Accuracies:** Task 1 = 99.81 %, Task 2 = 95.44 %, Task 1′ = 97.56 %, Task 2′ = 99.81 %. Separate columns eliminate forgetting and allow the primed tasks to converge cleanly.

## Stage 2 — Residual Columns
- **Goal:** Train Task 1′/2′ in new columns while *inheriting* their parents’ masks (parent column stays active for inference but frozen for learning).
- **Command:** `python scripts/run_experiment.py --config configs/stage2_residual_columns.yaml`
- **Result:** Residual masks retain Task 1/2 accuracy near 99.8 % while Task 1′/2′ still hit ~97–100 %. Compared to disjoint columns, residual masking offers similar retention but allows Task 1′/2′ to leverage the parent activations during inference.
