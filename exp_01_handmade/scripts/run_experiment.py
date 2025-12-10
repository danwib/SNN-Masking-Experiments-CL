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


def format_stage(stage) -> str:
    metrics = ", ".join(f"{metric.task_name}: {metric.accuracy:.2%}" for metric in stage.metrics)
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
