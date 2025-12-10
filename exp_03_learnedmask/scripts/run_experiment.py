#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from masking_experiments.config import load_config
from masking_experiments.experiments import run_experiment


def _format_metric(metric) -> str:
    prompt_note = f" [{metric.prompt_used}]" if metric.prompt_used != metric.task_name else ""
    overlap = (
        f" (error_overlap={metric.error_overlap:.2%})" if metric.error_overlap is not None else ""
    )
    return f"{metric.task_name}: {metric.accuracy:.2%}{prompt_note}{overlap}"


def format_stage(stage) -> str:
    metrics = ", ".join(_format_metric(metric) for metric in stage.metrics)
    if stage.metric_deltas:
        deltas = ", ".join(f"{name}:{delta:+.2%}" for name, delta in stage.metric_deltas.items())
        return f"{stage.label} -> {metrics} | Δ {deltas}"
    return f"{stage.label} -> {metrics}"


def _print_mask_logs(stage) -> None:
    if not stage.mask_logs:
        return
    for task_id, info in stage.mask_logs.items():
        if task_id == "_occupancy":
            continue
        mask_values = ", ".join(f"{value:.2f}" for value in info["mask"])
        print(f"   [mask] {task_id}: entropy={info['entropy']:.3f} mask=[{mask_values}]")
    occupancy = stage.mask_logs.get("_occupancy")
    if occupancy:
        for bank, values in occupancy.items():
            formatted = ", ".join(f"{value:.2f}" for value in values)
            print(f"   [occupancy] {bank}: [{formatted}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a masking experiment.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    result = run_experiment(config)

    print(f"[experiment] {result.name}")
    for stage in result.stages:
        print(" - " + format_stage(stage))
        _print_mask_logs(stage)


# Stage 4 soft-column experiments summary:
#   New SoftColumnMaskController plus configs stage4_parallel_soft.yaml, stage4_sequential_X_soft.yaml,
#   stage4_sequential_X_Xprime_soft.yaml, stage4_sequential_X_soft_sleep.yaml,
#   stage4_sequential_X_Xprime_soft_sleep.yaml, and stage4_sequential_residual_sleep.yaml
#   enable learned masks over many micro-columns or residual sleep variants.
#   Run them via:
#     python -m scripts.run_experiment --config configs/stage4_parallel_soft.yaml
#     python -m scripts.run_experiment --config configs/stage4_sequential_X_soft.yaml
#     python -m scripts.run_experiment --config configs/stage4_sequential_X_Xprime_soft.yaml
#     python -m scripts.run_experiment --config configs/stage4_sequential_X_soft_sleep.yaml
#     python -m scripts.run_experiment --config configs/stage4_sequential_X_Xprime_soft_sleep.yaml
#     python -m scripts.run_experiment --config configs/stage4_sequential_residual_sleep.yaml


if __name__ == "__main__":
    main()
