"""
Quick inspection of tasksource GT mappings.

Run with:
    python test_ts_mapping.py [N]

Prints, for each of the first N datasets discovered via tasksource:
  - The dataset id and task type
  - The raw HF columns (loaded via streaming)
  - The GT mapping extracted by get_ts_gt_mapping()
  - Whether the task is single-text or paired-text
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from datasets import load_dataset
from src.eval_ts import get_ts_gt_mapping
from notebooks.module import discover_datasets_ts


def get_raw_columns(hf_name: str, hf_config: str | None) -> set:
    """Load just the feature schema (no data downloaded) via streaming."""
    from src.standardize_api import _load_split
    try:
        ds = _load_split(hf_name, hf_config)
        return set(ds.features.keys())
    except Exception:
        return set()


def inspect(n: int = 20, task_types: list = None):
    datasets = discover_datasets_ts(task_types=task_types or ["Classification", "MultipleChoice"])
    datasets = datasets[:n]

    print(f"Inspecting {len(datasets)} tasksource datasets\n")
    print(f"{'ID':<40} {'Type':<22} {'GT mapping'}")
    print("-" * 110)

    for exp in datasets:
        prep      = exp["preprocessing"]
        ts_id     = exp["id"]
        ttype     = exp["task_type"]
        hf_name   = exp["hf_name"]
        hf_config = exp["hf_config"]

        raw_cols = get_raw_columns(hf_name, hf_config)
        gt       = get_ts_gt_mapping(prep, raw_columns=raw_cols)

        paired    = "sentence2" in gt
        structure = "paired" if paired else "single"

        gt_str    = "  ".join(f"{k}←{v}" for k, v in gt.items())
        raw_str   = "{" + ", ".join(sorted(raw_cols)) + "}" if raw_cols else "(unavailable)"

        print(f"{ts_id:<40} {ttype+'/'+structure:<22} {gt_str}")
        print(f"  raw cols: {raw_str}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    inspect(n)
