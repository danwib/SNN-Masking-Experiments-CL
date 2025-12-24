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

## Stage 5B0 – Key/Query Routing (Static Baseline)
- **Command:** `python -m scripts.run_experiment --config configs/stage5B0_sequential_X_Xprime_soft_sleep_keyquery_static.yaml`
- **Highlights:**
  - This is a pure routing swap over the Stage 5A1 schedule: we keep the prompt embeddings (task-id + optional input projection) exactly as-is, but replace the per-task mask logits with column keys and query attention. No prime warm-up, no adaptive replay, no gradient-similarity nudging—just the standard Task 1 → Task 2 → Task 1′ → Task 2′ → sleep sequence on SplitMNIST.
  - Base tasks sit in the base bank, primes sit in the novel bank (`consolidate_into` maps the primes back to parents). After sleep the prompt overrides collapse the primes onto the parent columns just like Stage 4.
  - The goal of this run is to answer “Does key/query routing work as a drop-in replacement when the context embeddings are already good?”—the answer is yes: accuracy and error-overlap match the Stage 5A1 baseline (e.g., Task 1≈99.7 %, Task 2≈96.2 %, Task 1′≈96.3 %, Task 2′≈99.7 % after sleep) while giving us the benefits of learned routing, albeit with much longer sleep (80 epochs) compared to the adaptive Stage 5B1 variants.

## Stage 5A2 – Input-Aware Context Embeddings
- **Command:** `python -m scripts.run_experiment --config configs/stage5A2_sequential_X_Xprime_soft_sleep_input.yaml`
- **Highlights:**
  - Extends Stage 5A1.B by weighting the prompt vectors with per-task input statistics (`include_input_in_prompt=true`, `prompt_input_scale=0.8`). Task/Task′ pairs now occupy nearby points in embedding space, and the dynamic scheduler interleaves replay batches based on these embeddings.
  - With 8 000 samples per task (double the Stage 4 baseline), the adaptive sleep phase (10 epochs, temperature = 0.1) still finishes at 99.7 % / 96.5 % for Task 1/2 and 98.9 % / 99.5 % for Task 1′/2′, while error-overlap stays ≥46 %. Mask entropy collapses to zero (pure one-hots), confirming the context embeddings drive the primes cleanly onto their parent columns.
  - Because the implementation is just “task-mean inputs → linear projection → add to task embedding → normalise”, it can be replicated easily: compute per-task feature means, set `include_input_in_prompt=true`, pick a `prompt_input_scale`, and reuse the Stage 5A1 dynamic sleep config.
- **Scaled Results + Similarity-Aware Sampling:** Pushing the config further (`max_train_samples=32000`, `sleep.epochs=5`, `sleep.temperature=0.1`, `gradient_similarity_influence≈0.2`, `gradient_update_interval=5`) yields genuine continual-learning behaviour: Task 1≈99.6 %, Task 2≈96.3 %, Task 1′≈94.1 %, Task 2′≈99.1 % after sleep with only five adaptive epochs. The key ingredients:
  1. **Input-aware prompts:** each task’s ID embedding is augmented with its dataset mean, projected via a learnable linear layer and normalised.
  2. **Prime-first warm-up + base pass:** every sleep epoch starts by replaying each prime task (fractional passes allowed) and running one base-task pass, anchoring both banks.
  3. **Adaptive replay:** `_DynamicReplayState` always selects the worst-loss `top_percent` plus an `extra_percent` of the remainder, with EMA loss tracking and a probability floor.
  4. **Gradient-similarity nudging:** every N batches we grab the prompt-vector gradient, normalise the stored context embeddings, and adjust neighbouring tasks’ probability biases by `gradient_similarity_influence * ||grad|| * cosine_similarity`. This gently increases replay frequency for tasks whose embeddings lie near the ones currently struggling, without tampering with the distillation loss itself.
  5. **Interleaved queues:** per-task batch queues are shuffled together so base/prime tasks take turns even when dataset sizes differ, preventing starvation.
  Following these steps reproduces the reported 32 k-sample run: five adaptive epochs, low-temperature distillation, and near-perfect retention across all four tasks.

## Stage 5B1 – Key/Query Mask Routing
- **Command:** `python -m scripts.run_experiment --config configs/stage5B1_sequential_X_Xprime_soft_sleep_keyquery.yaml`
- **Highlights:**
  - Each column now owns a learnable key vector in the prompt space, and each task’s prompt embedding (ID + input projection) acts as a query. Routing is computed as `softmax((K·q)/τ)` over the bank allowed for that task (base vs. novel), so mask weights emerge directly from embedding similarity instead of per-task logits.
  - We kept the Stage 5A2 scheduler intact (prime warm-up, per-epoch base replay, adaptive replay, gradient-similarity nudging, interleaved queues) and simply replaced the per-task mask parameters with column keys. Because the same embedding drives both the SNN input and the routing query, continual learning becomes even more consistent: similar tasks naturally share columns while remaining differentiable.
  - The 32 k-sample / 5-epoch run now achieves ≥99.5 % / 96.3 % / 94 % / 99 % across Task 1/2/1′/2′ with the attention-based controller, demonstrating that key/query routing can fully replace the old mask logits while preserving the adaptive-sleep gains. Re-implementing it is straightforward: add column keys, clamp the attention temperature, restrict attention to the configured bank, and reuse the existing Mask/Sleep machinery.
- **Drift-Resistant Variant (Stage 5B1.2):**
  - **Command:** `python -m scripts.run_experiment --config configs/stage5B1_sequential_X_Xprime_soft_sleep_keyquery.yaml` (with `training.base_lr_scale_during_novel≈0.2`, `masking.key_query.novel_temperature=0.05`, `masking.key_query.novel_entropy_weight=0.5`, `masking.key_query.base_regularization=0.1`)
  - **Results:** With `max_train_samples=32000` and just five sleep epochs, the final accuracies are task1 = 98.98 %, task2 = 96.27 %, task1′ = 97.45 %, task2′ = 99.16 % (error-overlap ≈60–86 %). Task 2 only drops ~0.7 % between novel insertions and recovers fully after the final sleep, while the primes consolidate back into the base prompts with <1 % loss.
  - **Implementation details:** Let `q_t∈ℝ^d` be the prompt embedding for task t and `K∈ℝ^{C×d}` the column keys. For each bank we compute attention weights
    \[
      α_t = \text{softmax}\left(\frac{K q_t}{τ_b}\right), \quad τ_b =
      \begin{cases}
        τ_{\text{novel}} \approx 0.05, & t\in \text{novel bank}\\
        τ_{\text{base}}, & t\in \text{base bank}
      \end{cases}
    \]
    Novel columns therefore receive almost one-hot weights, while base columns remain softer. We add a bank-specific entropy penalty `λ_novel * H(α_t)` to push the novel masks toward δ-functions.
    - **Margin lock:** once a column `c` repeatedly wins for task t, we store `c` as `owner(t)` and enforce `α_t[c] ≥ α_t[j] + m` for all other columns `j` in the same bank via a hinge loss `λ_margin * max(0, m - (logit_c - logit_j))`.
    - **Base regularizer:** for any two tasks `t,u` sharing the base bank we penalize positive cosine similarity between their normalized queries,
      \[
        L_{\text{base}} = λ_{\text{base}} \cdot \max(0, q_t^\top q_u / (\|q_t\|\|q_u\|)).
      \]
      This keeps occupied base keys separated even when the underlying encoder drifts.
  - **Shared-weight drift control:** during wake training on a novel task we scale the shared encoder/readout learning rate by `base_lr_scale_during_novel` (e.g., 0.2). Formally, the update step becomes `θ ← θ − (η·s) ∇θ L`, where `s` is the scale (0 freezes the base). This keeps base columns close to their previous optimum without slowing the novel column or requiring replay.
  - **Sleep promotions:** after Task 2 finishes, we trigger sleep immediately (`training.sleep_after: [task2]`) so Task 1/Task 2 are distilled back into the base bank before Task 1′ arrives, then repeat after Task 2′. This staged consolidation plus the controlled LR ensures each novel task learns in isolation, then hands off its column cleanly to the long-term base bank.
