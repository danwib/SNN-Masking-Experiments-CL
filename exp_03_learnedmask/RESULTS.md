# Stage 4 Soft Allocation + Sleep

We extended the SplitMNIST SNN to learn soft column masks and to distil primed tasks back into their parent prompts. Two representative runs show that continual learning holds even when columns are shared or reallocated.

## Residual Columns + Sleep
- **Command:** `python -m scripts.run_experiment --config configs/stage4_sequential_residual_sleep.yaml`
- **Highlights:**
  - Parents (Task 1/2) stay near 100 % accuracy throughout sequential training.
  - Primed tasks (Task 1′/2′) use residual columns during training, then switch to parent prompts during sleep.
  - Sleep distillation (label weight = 0) keeps all tasks above 94 % while collapsing prompts: Task 1′/2′ inherit the parent masks and still score 95 % / 99 %.
  - Mask logs confirm the same columns are reused, and delta metrics show <3 % degradation post-sleep.

## Soft Base/Novel Banks + Sleep
- **Command:** `python -m scripts.run_experiment --config configs/stage4_sequential_X_Xprime_soft_sleep.yaml`
- **Highlights:**
  - Learned soft masks allocate base columns to Task 1/2 and novel columns to Task 1′/2′.
  - After training, primed tasks achieve 98–100 % in their novel bank, while parents remain at ~99/97 %.
  - Sleep with prompt overrides results in Task 1′/2′ adopting exactly the parent masks (confirmed by entropy/occupancy logs) with only ~3 % loss in accuracy.
  - Error-overlap metrics remain high (≈40–80 %), showing the consolidated model reproduces the teacher’s behavior.

Together these experiments demonstrate that soft allocation plus sleep consolidation preserves continual-learning performance while letting us re-use the same columns for both base and primed tasks.
