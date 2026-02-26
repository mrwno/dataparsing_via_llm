"""Evaluation functions using tasksource preprocessing objects as ground truth."""

import pandas as pd
from src.standardize_api import load_standardized_dataset
from src.eval import apply_llm_mapping


# Default field values in tasksource dataclasses.
# A field at its default could mean "not used" (no such column) OR
# "used and the column happens to share the canonical name" (e.g. sentence1→sentence1).
_TS_DEFAULTS = {
    "sentence1": "sentence1",
    "sentence2": "sentence2",
    "labels":    "labels",
    "inputs":    "input",
    "tokens":    "tokens",
    "prompt":    "prompt",
    "output":    "output",
}

# Annotation/metadata keys — Unitxt-specific, ignored when scoring against tasksource GT
def _is_annotation(k: str) -> bool:
    return k in ("classes", "type_of_class", "type_of_relation") or k.endswith("_type")


def get_ts_gt_mapping(preprocessing, raw_columns: set = None) -> dict:
    """
    Extract raw-column assignments from a tasksource preprocessing object.

    A field is included when:
    - Its value differs from the default (explicit non-identity mapping), OR
    - Its value equals the default AND that column actually exists in the
      raw dataset (the column is named the same as the canonical field).

    When ``raw_columns`` is None the second condition cannot be checked, so
    only non-default assignments are returned (useful for quick inspection
    without loading the dataset).

    Args:
        preprocessing:  A tasksource Preprocessing object.
        raw_columns:    Set of column names from the raw HF dataset (optional).

    Returns:
        Dict like {"sentence1": "premise", "sentence2": "hypothesis",
                   "labels": "label"}.
    """
    d = preprocessing.to_dict() if hasattr(preprocessing, "to_dict") else {}
    mapping = {}
    for field, default in _TS_DEFAULTS.items():
        if field not in d:
            continue
        val = d[field]
        if not isinstance(val, str):
            continue
        if val != default:
            # Explicitly mapped to a different column name
            mapping[field] = val
        elif raw_columns is not None and val in raw_columns:
            # Default value but the column actually exists (e.g. MRPC sentence1/sentence2)
            mapping[field] = val

    # MultipleChoice: collect choice0, choice1, … attributes
    for k, v in d.items():
        if k.startswith("choice") and isinstance(v, str):
            mapping[k] = v

    return mapping


def compute_ts_struct_score(ts_gt: dict, pred_mapping: dict) -> float:
    """
    Structural score: Jaccard similarity between the raw columns selected by
    the tasksource GT and the predicted mapping.

    Because both mappings reference the same raw HF column names, this score
    is canonical-name-agnostic: it does not matter whether the method uses
    Unitxt field names (text_a, label) or tasksource names (sentence1, labels).

    Args:
        ts_gt:        Dict {ts_field: raw_col} from get_ts_gt_mapping().
        pred_mapping: Dict {unitxt_field: raw_col | literal} from the method.

    Returns:
        Jaccard score in [0, 1].
    """
    gt_cols = set(ts_gt.values())
    pred_cols = {
        v for k, v in pred_mapping.items()
        if k != "task" and not _is_annotation(k) and isinstance(v, str)
    }
    if not gt_cols and not pred_cols:
        return 1.0
    intersection = len(gt_cols & pred_cols)
    union = len(gt_cols | pred_cols)
    return intersection / union if union > 0 else 0.0


def evaluate_ts(
    hf_name: str,
    hf_config: str | None,
    preprocessing,
    n_samples: int = 10,
    standardize_fn=None,
) -> dict:
    """
    Evaluate a standardization method against a tasksource preprocessing GT.

    Args:
        hf_name:        HuggingFace dataset name (e.g. "glue").
        hf_config:      Dataset config name (e.g. "sst2"), or None.
        preprocessing:  A tasksource Preprocessing object used as ground truth.
        n_samples:      Number of raw rows to load for the LLM-mapped DataFrame.
        standardize_fn: Callable (dataset_name, config) -> dict.
                        Defaults to load_standardized_dataset (API-based).

    Returns:
        Dict with keys: dataset, score, struct_score, ts_gt, mapping, df_llm, error.
    """
    fn = standardize_fn or load_standardized_dataset
    llm_result = fn(hf_name, config=hf_config)
    mapping = llm_result.get("mapping", {})

    ds_raw = llm_result.get("dataset")
    if not ds_raw:
        raise ValueError("standardize_fn failed to return a valid dataset object.")

    raw_columns = set(ds_raw.features.keys())
    df_raw = pd.DataFrame(list(ds_raw.take(n_samples)))
    df_llm = apply_llm_mapping(df_raw, mapping)

    ts_gt        = get_ts_gt_mapping(preprocessing, raw_columns=raw_columns)
    struct_score = compute_ts_struct_score(ts_gt, mapping)
    score        = round(struct_score, 3)

    dataset_id = f"{hf_name}/{hf_config}" if hf_config else hf_name

    return {
        "dataset":      dataset_id,
        "score":        score,
        "struct_score": round(struct_score, 3),
        "ts_gt":        ts_gt,
        "mapping":      mapping,
        "df_llm":       df_llm,
        "error":        None,
    }
