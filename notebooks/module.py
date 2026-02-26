"""Experiment helpers: dataset config, evaluation loop, tables, and plots."""
import os
import sys
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import logging
import warnings
from functools import partial
from src.eval import evaluate
from src.eval_ts import evaluate_ts
from src.baselines import baseline_keyword_match, baseline_embedding_match
from src.standardize_api import load_standardized_dataset
from src.standardize_local import load_standardized_dataset_local

# Allow Unitxt cards that use FilterByExpression
os.environ.setdefault("UNITXT_ALLOW_UNVERIFIED_CODE", "True")

# Silence noisy third-party loggers and HF progress bars
for _name in ("unitxt", "datasets", "huggingface_hub", "transformers", "filelock"):
    logging.getLogger(_name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
try:
    import datasets as _ds
    _ds.disable_progress_bar()
except Exception:
    pass


# ── Card discovery ───────────────────────────────────────────────────────────

# How each task family maps to substrings found in the Unitxt task field.
# NLI uses "relation" (tasks.classification.multi_class.relation).
_TASK_MATCHERS = {
    "classification": lambda t: "classification" in t and "relation" not in t,
    "nli":            lambda t: "relation" in t or "nli" in t,
    "qa":             lambda t: "qa" in t or "question_answering" in t,
    "generation":     lambda t: "generation" in t or "summarization" in t or "translation" in t,
}


def discover_datasets(task_types: list = None) -> list[dict]:
    """
    Scan the Unitxt local catalog and return all cards whose task matches
    one of the requested task families.

    Args:
        task_types: Subset of ["classification", "nli", "qa", "generation"].
                    Defaults to all four.

    Returns:
        List of dicts with keys: card_id, hf_name, hf_config, task_type.
        Only cards backed by a HuggingFace loader are included.
    """
    from unitxt.catalog import get_local_catalogs_paths

    task_types = task_types or list(_TASK_MATCHERS.keys())
    matchers = {k: _TASK_MATCHERS[k] for k in task_types if k in _TASK_MATCHERS}

    results = []
    seen = set()

    for catalog_path in get_local_catalogs_paths():
        cards_dir = os.path.join(catalog_path, "cards")
        if not os.path.exists(cards_dir):
            continue

        for root, _, files in os.walk(cards_dir):
            for file in sorted(files):
                if not file.endswith(".json"):
                    continue

                full_path = os.path.join(root, file)
                try:
                    with open(full_path) as f:
                        card = json.load(f)
                except Exception:
                    continue

                if card.get("__type__") != "task_card":
                    continue

                loader = card.get("loader") or {}
                if loader.get("__type__") != "load_hf":
                    continue

                task_str = card.get("task") or ""
                matched_type = next(
                    (t for t, match in matchers.items() if match(task_str)), None
                )
                if matched_type is None:
                    continue

                hf_name   = loader.get("path")
                hf_config = loader.get("name")

                # Build card_id: relative path without .json, slashes → dots
                rel = os.path.relpath(full_path, cards_dir)
                card_id = os.path.splitext(rel)[0].replace(os.sep, ".")

                key = card_id
                if key in seen:
                    continue
                seen.add(key)

                results.append({
                    "card_id":   card_id,
                    "hf_name":   hf_name,
                    "hf_config": hf_config
                })

    return results


# ── Dataset registry ─────────────────────────────────────────────────────────

GLUE_DATASETS = [
    {"card_id": "sst2", "hf_name": "glue", "hf_config": "sst2"},
    {"card_id": "mrpc", "hf_name": "glue", "hf_config": "mrpc"},
    {"card_id": "qnli", "hf_name": "glue", "hf_config": "qnli"},
    {"card_id": "mnli", "hf_name": "glue", "hf_config": "mnli"},
    {"card_id": "wnli", "hf_name": "glue", "hf_config": "wnli"},
]

SIMULATED_LOCAL_MODEL_ID = "google/gemma-3n-e4b-it"

METHODS = {
    "keyword":   baseline_keyword_match,
    "embedding": baseline_embedding_match,
    "local_llm": load_standardized_dataset_local, #partial(load_standardized_dataset, model_id=SIMULATED_LOCAL_MODEL_ID),
    "api_llm":   load_standardized_dataset,
}

COL_ORDER = ["keyword", "embedding", "local_llm", "api_llm"]


# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_evaluation(datasets: list = None, methods: dict = None) -> pd.DataFrame:
    """
    Run all (dataset, method) combinations and return a tidy results DataFrame.

    Args:
        datasets: List of dataset dicts (default: GLUE_DATASETS).
        methods:  Dict of {name: standardize_fn} (default: METHODS).

    Returns:
        DataFrame with columns: dataset, method, score, struct_score, annot_score.
    """
    datasets = datasets or GLUE_DATASETS
    methods  = methods  or METHODS

    print(f"{len(datasets)} datasets  x  {len(methods)} methods  =  {len(datasets) * len(methods)} evaluations\n")

    rows = []
    for exp in datasets:
        card_id, hf_name, hf_config = exp["card_id"], exp["hf_name"], exp["hf_config"]
        print(f"── {card_id} ──")

        for method_name, standardize_fn in methods.items():
            print(f"  {method_name}...", end=" ", flush=True)
            try:
                result = evaluate(
                    hf_name=hf_name,
                    hf_config=hf_config,
                    card_id=card_id,
                    standardize_fn=standardize_fn,
                )
                score        = result["score"]
                struct_score = result["struct_score"]
                print(f"Mapping: {result['mapping']}")
                print(f"  score={score:.3f}  struct={struct_score:.3f}")
            except Exception as e:
                score = struct_score = None
                print(f"ERROR: {e}")

            rows.append({
                "dataset":      card_id,
                "method":       method_name,
                "score":        score,
                "struct_score": struct_score,
            })

    print("\nEvaluation complete.")
    return pd.DataFrame(rows)


# ── Tasksource discovery & evaluation ─────────────────────────────────────────

def discover_datasets_ts(task_types: list = None) -> list[dict]:
    """
    Discover datasets from the tasksource catalog via list_tasks().

    Args:
        task_types: Subset of ["Classification", "MultipleChoice",
                    "TokenClassification"]. Defaults to Classification only.

    Returns:
        List of dicts with keys: id, hf_name, hf_config, task_type,
        preprocessing (the tasksource Preprocessing object used as GT).
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from tasksource.access import list_tasks

    task_types = task_types or ["Classification"]
    df = list_tasks()
    df = df[df.task_type.isin(task_types)]

    results = []
    for _, row in df.iterrows():
        results.append({
            "id":           row["id"],
            "hf_name":      row["dataset_name"],
            "hf_config":    row["config_name"] if pd.notna(row["config_name"]) else None,
            "task_type":    row["task_type"],
            "preprocessing": row["mapping"],
        })
    return results


def run_evaluation_ts(datasets: list = None, methods: dict = None) -> pd.DataFrame:
    """
    Run all (dataset, method) combinations using tasksource as ground truth.

    Args:
        datasets: List of dicts from discover_datasets_ts().
        methods:  Dict of {name: standardize_fn} (default: METHODS).

    Returns:
        DataFrame with columns: dataset, method, score, struct_score.
    """
    datasets = datasets or []
    methods  = methods  or METHODS

    print(f"{len(datasets)} datasets  x  {len(methods)} methods  =  {len(datasets) * len(methods)} evaluations\n")

    rows = []
    for exp in datasets:
        ts_id         = exp["id"]
        hf_name       = exp["hf_name"]
        hf_config     = exp["hf_config"]
        preprocessing = exp["preprocessing"]
        print(f"── {ts_id} ──")

        for method_name, standardize_fn in methods.items():
            print(f"  {method_name}...", end=" ", flush=True)
            try:
                result = evaluate_ts(
                    hf_name=hf_name,
                    hf_config=hf_config,
                    preprocessing=preprocessing,
                    standardize_fn=standardize_fn,
                )
                score        = result["score"]
                struct_score = result["struct_score"]
                print(f"Mapping: {result['mapping']}")
                print(f"  score={score:.3f}  struct={struct_score:.3f}")
            except Exception as e:
                score = struct_score = None
                print(f"ERROR: {e}")

            rows.append({
                "dataset":      ts_id,
                "method":       method_name,
                "score":        score,
                "struct_score": struct_score,
            })

    print("\nEvaluation complete.")
    return pd.DataFrame(rows)


# ── Pivot tables ──────────────────────────────────────────────────────────────

def make_pivot(df_results: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Return a dataset x method pivot table for the given score column."""
    p = (
        df_results
        .pivot(index="dataset", columns="method", values=score_col)
        .reindex(columns=[c for c in COL_ORDER if c in df_results["method"].values])
    )
    p.loc["Average"] = p.mean()
    return p.round(3)


def show_tables(df_results: pd.DataFrame) -> None:
    """Print the structural score pivot table."""
    from IPython.display import display
    print("=== Structural score  (Jaccard on field names, annotations excluded) ===")
    display(make_pivot(df_results, "score"))


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_results(df_results: pd.DataFrame, save_dir: str = "../results") -> None:
    """
    Plot three heatmaps (one per score) and an average bar chart.
    Saves to save_dir/comparison_heatmaps.png and comparison_avg.png.
    """
    os.makedirs(save_dir, exist_ok=True)

    pivot_score  = make_pivot(df_results, "score").drop(index="Average").astype(float)
    avg_score    = make_pivot(df_results, "score").loc["Average"].astype(float)

    heatmap_cfg = dict(annot=True, fmt=".2f", cmap="YlGn", vmin=0, vmax=1,
                       linewidths=0.5, linecolor="white")

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot_score) * 0.4)))
    sns.heatmap(pivot_score, ax=ax, **heatmap_cfg)
    ax.set_title("Structural score  (Jaccard on raw column names)", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    colors = sns.color_palette("Set2", len(avg_score))
    bars = ax2.bar(avg_score.index, avg_score.values, color=colors, edgecolor="black", width=0.5)
    ax2.set_title("Average structural score per method", fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Score")
    ax2.tick_params(axis="x", rotation=30)
    for bar in bars:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison_avg.png", dpi=150, bbox_inches="tight")
    plt.show()
