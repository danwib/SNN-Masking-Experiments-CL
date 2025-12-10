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
    return f"{stage.label} -> {metrics}"


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


if __name__ == "__main__":
    main()
