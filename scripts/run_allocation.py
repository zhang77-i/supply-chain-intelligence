from __future__ import annotations

# Preload CP-SAT before the project imports pandas on Windows.
from ortools.sat.python import cp_model as _cp_model  # noqa: F401

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cainiao_inventory.allocation import run_allocation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coordinate national target inventory across five stores."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "project.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_allocation(parse_args().config)
