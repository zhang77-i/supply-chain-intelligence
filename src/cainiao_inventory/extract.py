from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from .config import PipelineConfig

EXPECTED_FILES = {
    "item_feature2.csv",
    "item_store_feature2.csv",
    "config2.csv",
}


def ensure_raw_data(config: PipelineConfig) -> list[Path]:
    """Extract only the three competition CSVs into stable, normalized paths."""
    destination = config.raw["extracted_dir"]
    destination.mkdir(parents=True, exist_ok=True)

    existing = {path.name for path in destination.glob("*.csv")}
    if EXPECTED_FILES.issubset(existing):
        return sorted(destination.glob("*.csv"))

    archive = config.raw["archive"]
    if not archive.exists():
        raise FileNotFoundError(f"Missing competition archive: {archive}")

    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as bundle:
        members = {
            Path(info.filename).name: info
            for info in bundle.infolist()
            if not info.is_dir() and Path(info.filename).name in EXPECTED_FILES
        }
        missing = EXPECTED_FILES.difference(members)
        if missing:
            raise ValueError(f"Archive is missing required files: {sorted(missing)}")

        for filename in sorted(EXPECTED_FILES):
            target = destination / filename
            with bundle.open(members[filename]) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            extracted.append(target)

    return extracted
