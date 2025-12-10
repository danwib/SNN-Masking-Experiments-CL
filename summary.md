# Masking Experiments Summary

## Shared Building Blocks
- **Column-masked SNN core:** `masking_experiments.snn.SparseSNN` (both experiments) keeps a single spiking neural network with columns of 64–128 hidden units per task. Columns are gated by prompt-conditional masks, so every task reuses the same encoder/readout weights but only the selected column receives gradient updates. This isolates representation drift and lets us reason about how many columns are required for each task family.
- **Prompt-derived masks:** `masking_experiments.masking.MaskController` hashes each prompt (e.g., "Task 1'") into a deterministic vector and produces per-column learning-rate scales plus threshold shifts. `residual_from` activates multiple columns at inference while freezing parents; `consolidate_into` (experiment 02) lets several prompts share the same parent during sleep training. This mechanism is what enforces compartmentalization vs. sharing.
- **SplitMNIST pipelines:** `masking_experiments.data.SplitMNISTLoader` slices the MNIST digits into task-specific pairs (0/1, 2/3, …) and can hold out 20% of the samples for replay or evaluation. Every config limits training to 2k–4k samples to make the ablations fast while keeping accuracy high.
- **Experiment runner:** `scripts/run_experiment.py` reads YAML configs, builds datasets, trains either in parallel or following a user-defined schedule, and records accuracy after each stage. In experiment 02 the runner also computes error-overlap metrics by comparing pre/post sleep predictions.

## Experiment 01 — Handmade Mask Schedules (`exp_01_handmade`)
### Stage 1: establishing basic masking behavior
- **Parallel columns (`configs/stage1_parallel.yaml`):** Tasks 1 and 2 (digits 0/1 vs. 2/3) train simultaneously in separate columns while Task 1′/2′ columns stay frozen. Resulting accuracies reach 99.69 % (Task 1) and 96.25 % (Task 2), confirming that prompt-gated updates prevent mutual interference even without sequential fine-tuning.
- **Sequential columns (`configs/stage1_sequential.yaml`):** Tasks 1 → 2 train one after another to measure forgetting. Accuracy improves from ~50 % at cold start to 99.69 % on Task 1 after its stage, then Task 2 climbs to 96.13 % without knocking Task 1 off its 99.69 %. This verifies that column-wise masking alone eliminates catastrophic forgetting for the original tasks.
### Stage 2: column allocation ablations
- **Shared columns (`stage2_shared_columns.yaml`):** Primed tasks Task 1′/2′ reuse their parents’ columns. They reach ~98–100 % accuracy but overwrite the parents, dropping Task 1 to 57.25 % and Task 2 to 92.12 %. This quantifies the interference incurred when two related prompts share weights.
- **Disjoint columns (`stage2_disjoint_columns.yaml`):** Giving Task 1′/2′ fresh columns restores stability. Final metrics are Task 1 = 99.81 %, Task 2 = 95.44 %, Task 1′ = 97.56 %, Task 2′ = 99.81 %. Maintaining independent capacity eliminates forgetting and lets every task converge cleanly.
- **Residual columns (`stage2_residual_columns.yaml`):** Task 1′/2′ learn in new columns but inherit their parents via `residual_from`, so inference combines the parent + residual activations while only the new columns update. Parents stay near 99.8 % and the primed tasks reach ~97–100 %. This shows that residual composition can deliver disjoint-like stability while sharing parent context for transfer.

## Experiment 02 — Consolidation & Sleep (`exp_02_consolidation`)
Stage 2 repeats the disjoint and residual baselines above, serving as starting checkpoints for the sleep studies.

### Stage 3: sleep consolidation with disjoint parents (`configs/stage3_sleep_consolidation.yaml`)
- **Mechanism:** After sequential training a copy of the model becomes a frozen teacher. During sleep the student only activates the parent prompts (Task 1/2), but the replay buffer interleaves Task 1′/2′ samples and the update loss is pure distillation (label_weight = 0). `consolidate_into` maps primed prompts onto their parents so we can compress four experiences into two prompts.
- **Results:** Post-sleep accuracies are Task 1 = 99.22 %, Task 2 = 95.38 %, Task 1′ = 93.47 % (queried with the Task 1 prompt), Task 2′ = 98.69 %. Error-overlap scores between 20–78 % indicate that many of the teacher’s mistakes persist, meaning the compression preserves behavioral quirks even without explicit labels.
- **Why it matters:** This run demonstrates that knowledge captured in disjoint columns can be distilled back into the parents without real labels, trading a small accuracy drop for a 2× smaller prompt vocabulary.

### Stage 3: sleep consolidation with residual parents (`configs/stage3_residual_sleep.yaml`)
- **Mechanism:** Same sleep schedule as above, but the starting point already uses residual sharing (`residual_from`). Sleep aims to remove the auxiliary residual columns entirely by pushing their behavior into the parent prompts.
- **Results:** After sleep we observe Task 1 = 97.22 %, Task 2 = 95.41 %, Task 1′ = 95.69 %, Task 2′ = 99.09 %, each measured with the parent prompts. Error-overlap stays high (42–73 %), showing that most pre-sleep decision boundaries survive consolidation even though we no longer activate the extra columns.
- **Why it matters:** Residual sharing already kept the parent context alive, yet sleep still fuses the primed knowledge into the base columns, hinting that prompt-level consolidation can remove redundant capacity after transfer learning.

## Overall Takeaways
- Column-wise masking plus prompt-conditioned gating provides a simple way to compartmentalize continual-learning tasks on SplitMNIST. Sequential runs confirm that interference only appears when two prompts share a column.
- Allocating extra columns (`disjoint` or `residual`) solves forgetting but increases model footprint. Sleep consolidation shows how to claw back capacity by distilling related prompts into shared parents, at the cost of modest accuracy drops.
- The experiments are implemented entirely with lightweight PyTorch modules and YAML configs, so extending the study (e.g., new prompts, longer sleep, different replay weights) only requires editing config files rather than core code.
