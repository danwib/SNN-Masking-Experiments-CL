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

## Mixed MNIST + FashionMNIST Schedule
- **Command:** `python -m scripts.run_experiment --config configs/stage4_sequential_mixed_soft_sleep.yaml`
- **Highlights:**
  - Four base tasks (two MNIST pairs, two FashionMNIST pairs) train sequentially twice, maintaining ≥95 % accuracy each time a new task arrives; deltas stay within ±5 % even before sleep.
  - Soft masks naturally reuse columns—mnist_task1/fashion_task2 share a column while mnist_task2/fashion_task1 occupy their own—yet base-bank occupancy never exceeds 50 %, indicating ample unused capacity.
  - Primed held-out tasks learn in their novel bank, then sleep distillation (label weight = 0, `held_out_replay=2`) consolidates them perfectly into their parent prompts with 100 % error-overlap.
  - Final accuracies remain 99.8 % / 97.3 % / 98.7 % / 97.4 % for the four base tasks, showing continual learning holds even when mixing distinct datasets.

## Stage 5A1 – Learned Task-ID Prompt Embeddings
- **Command:** `python -m scripts.run_experiment --config configs/stage5A1_sequential_X_Xprime_soft_sleep_learned_taskid.yaml`
- **Highlights:**
  - `masking.prompt_mode=learned_task_id` swaps the hashed prompt vector for a learnable embedding table indexed by each task’s `task_id`, letting the prompt stream adapt jointly with the SNN.
  - Base tasks (Task 1/2) continue to occupy the base bank, primed variants live entirely in the novel bank, and sleep consolidation distills Task 1′→Task 1 / Task 2′→Task 2 with label weight = 0 and `held_out_replay=0` (no wake replay at all).
  - Learned prompt embeddings are optimized alongside the soft-column logits, so the controller now exposes gradients for both routing and context vectors; mask logs confirm the primes collapse onto their parents after sleep.
  - Running with 4 000 samples per task (full MNIST split) plus 15 sleep epochs yields 99.7 % / 96.3 % accuracy for Task 1/2 and 96.2 % / 99.7 % for Task 1′/2′ after sleep, with error-overlap staying in the 47–75 % band while base-bank masks remain pure one-hots and novel columns collapse back to their parents.
- **Dynamic Sleep (Stage 5A1.A):** `python -m scripts.run_experiment --config configs/stage5A1_sequential_X_Xprime_soft_sleep_dynamic.yaml`
  - Implementation blueprint: (1) prime-first warm-up (fractional passes allowed) so Task 1′/2′ get rehearsal before selection biases kick in; (2) per-epoch base replay to anchor the parent columns; (3) `_DynamicReplayState` tracks per-sample losses and always replays the worst-loss `top_percent` plus a probability floor–adjusted `extra_percent`; (4) per-task batch queues are interleaved randomly so base and prime tasks take turns even when their datasets are imbalanced.
  - This adaptive replay completely replaces `held_out_replay` and retains Stage 4 accuracy with ~10 sleep epochs (vs. the old 15–20) even when we double the number of training samples per task.
- **Dynamic Sleep + Low Temp (Stage 5A1.B):** `python -m scripts.run_experiment --config configs/stage5A1_sequential_X_Xprime_soft_sleep_dynamic_lowtemp.yaml`
  - Same scheduler but with `sleep.temperature` set to 0.1 (values below 1e‑3 are clamped). Sharper teacher targets plus interleaved replay keep Task 1/2 ≥99 % / 96 % and Task 1′/2′ ≥98.9 % / 99.5 % after sleep, even when we boost each dataset to 4 000+ samples.
- **Observation:** Implementing the adaptive replay requires only the steps above (prime warm-up, per-sample loss tracking, interleaved batch queues). These notes stay in RESULTS.md so anyone can re-create the scheduler from scratch without digging through the code.
