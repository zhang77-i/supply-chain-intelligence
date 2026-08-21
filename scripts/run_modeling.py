from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cainiao_inventory.modeling import run_modeling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rolling demand forecasts and cost-sensitive inventory evaluation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "project.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_modeling(parse_args().config)
