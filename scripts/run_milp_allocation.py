from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cainiao_inventory.milp_allocation import run_milp_allocation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve coordinated multi-warehouse inventory with MILP."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "project.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_milp_allocation(parse_args().config)
