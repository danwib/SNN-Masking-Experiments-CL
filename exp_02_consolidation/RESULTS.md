# Experiment Summary

The consolidation workspace focuses on SplitMNIST continual learning where a shared sparse SNN is divided into columns that are selectively activated by task prompts. Every run follows the same training schedule (Tasks 1 → 2 → 1′ → 2′) but varies how the primed tasks interact with their parents and how we consolidate knowledge afterward.

## Stage 2 — Disjoint Columns
- **Command:** `python scripts/run_experiment.py --config configs/stage2_disjoint_columns.yaml`
- **Idea:** Give Task 1′/2′ their own columns so we can measure forgetting when tasks are compartmentalized.
- **Outcome:** Final accuracies were roughly Task 1 = 99.8 %, Task 2 = 95.4 %, Task 1′ = 97.6 %, Task 2′ = 99.8 %. Disjoint capacity eliminates catastrophic interference—the parent tasks remain stable even after the primed variants finish training.

## Stage 2 — Residual Columns
- **Command:** `python scripts/run_experiment.py --config configs/stage2_residual_columns.yaml`
- **Idea:** Let Task 1′/2′ learn in extra columns that *reuse* their parents’ activations for inference (via `residual_from`) while updates are confined to the residual columns.
- **Outcome:** Task 1/2 stay around 99.8 %, and Task 1′/2′ also converge near 97–100 %, showing that residual masking can protect the parents while boosting primed tasks with shared context.

## Stage 3 — Sleep Consolidation (Disjoint Columns)
- **Command:** `python scripts/run_experiment.py --config configs/stage3_sleep_consolidation.yaml`
- **Setup:** Start from the disjoint Stage 2 model, snapshot a teacher, and run a “sleep” phase where we perform **pure distillation** (label weight = 0) into the parent columns. The student only sees Task 1/2 prompts, but the replay buffer mixes both X and X′ with `held_out_replay=2` so primed samples appear twice as often.
- **Key Numbers:** After sleep the shared prompts reach Task 1 = 99.22 %, Task 2 = 95.38 %, Task 1′ = 93.47 % (queried with the Task 1 prompt), Task 2′ = 98.69 %. Error-overlap values (20–78 %) show that many teacher mistakes persist even though every task now uses a single prompt—evidence that we are compressing multiple experiences without supervision.
- **Takeaway:** We can collapse Task 1′/2′ back into Task 1/2 prompts purely via distillation, incurring only modest accuracy drops while keeping the held-out tasks functional under the parent context.

## Stage 3 — Sleep Consolidation (Residual Columns)
- **Command:** `python scripts/run_experiment.py --config configs/stage3_residual_sleep.yaml`
- **Setup:** Same distillation pipeline as above but starting from the residual Stage 2 run. Residual columns already co-activate with the parents, so this experiment tests whether sleep can remove the auxiliary columns altogether.
- **Key Numbers:** After sleep we observe Task 1 = 97.22 %, Task 2 = 95.41 %, Task 1′ = 95.69 % (with the Task 1 prompt), Task 2′ = 99.09 %. Error-overlap is 42–73 %, indicating substantial retention of the teacher’s behavior in spite of the column merge.
- **Takeaway:** Even when residual sharing keeps the parent context alive, the sleep phase still consolidates Task 1′/2′ knowledge into the parent masks so we no longer need the auxiliary columns. The slight Task 1 drop suggests a trade-off between compression and fidelity, but the primed tasks remain highly accurate under the shared prompts, demonstrating continual learning without replaying labels.
