"""Evaluation functions for Unitxt LLM Agent."""

import os
import json
import pandas as pd
from datasets import load_dataset
from unitxt import load_dataset as unitxt_load
from src.standardize_api import load_standardized_dataset


UNITXT_METADATA_FIELDS = {'metadata', 'data_classification_policy'}

# Annotation/metadata fields — ignored when computing the structural score
def _is_annotation(k: str) -> bool:
    return k in ("classes", "type_of_class", "type_of_relation") or k.endswith("_type")


def extract_unitxt_standardized(unitxt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract clean standardized data from Unitxt's task_data column.

    Args:
        unitxt_df: DataFrame containing Unitxt formatted data with task_data column.

    Returns:
        DataFrame with clean standardized data excluding metadata fields.
    """
    if 'task_data' not in unitxt_df.columns:
        return pd.DataFrame()
    
    rows = []
    for task_data in unitxt_df['task_data']:
        if isinstance(task_data, str):
            task_data = json.loads(task_data)
        
        clean_row = {k: v for k, v in task_data.items() if k not in UNITXT_METADATA_FIELDS}
        rows.append(clean_row)
    
    return pd.DataFrame(rows)


def extract_task_data_fields(unitxt_df: pd.DataFrame) -> set:
    """
    Extract the actual data field names from Unitxt's task_data column.

    Args:
        unitxt_df: DataFrame containing Unitxt formatted data with task_data column.

    Returns:
        Set of field names excluding metadata fields.
    """
    if 'task_data' not in unitxt_df.columns:
        return set()
    
    sample = unitxt_df['task_data'].iloc[0]
    if isinstance(sample, str):
        sample = json.loads(sample)
    
    return {k for k in sample.keys() if k not in UNITXT_METADATA_FIELDS}


def apply_llm_mapping(raw_df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Apply LLM mapping to raw dataset and return standardized DataFrame.

    - Column references (value matches a raw column name) are renamed.
    - Literal string values (e.g. text_type, type_of_class) are added as constant columns.
    - Integer labels are converted to text using the classes list.
    - The classes list is added as a string column.

    Args:
        raw_df: Raw DataFrame to be standardized.
        mapping: Dictionary mapping standard field names to raw column names or literal values.

    Returns:
        Standardized DataFrame with renamed columns and injected metadata.
    """
    classes = mapping.get("classes", [])

    # Keys that are always semantic annotations, never column references:
    # - "task", "classes" (handled separately)
    # - keys ending in "_type" (e.g. text_a_type, text_b_type)
    # - "type_of_class", "type_of_relation"
    def _is_annotation(k: str) -> bool:
        return k in ("task", "classes", "type_of_class", "type_of_relation") or k.endswith("_type")

    # Column references: value matches a real column name AND key is not an annotation
    col_refs = {
        v: k for k, v in mapping.items()
        if not _is_annotation(k) and isinstance(v, str) and v in raw_df.columns
    }
    # Literals: annotation keys + any non-annotation key whose value is not a real column
    literals = {
        k: v for k, v in mapping.items()
        if k not in ("task", "classes") and isinstance(v, str)
        and (_is_annotation(k) or v not in raw_df.columns)
    }

    # Keep only referenced columns and rename them
    df = raw_df[list(col_refs.keys())].rename(columns=col_refs)

    # Convert integer labels to text class names
    if "label" in df.columns and classes:
        df["label"] = df["label"].map(
            lambda x: classes[x] if isinstance(x, int) and 0 <= x < len(classes) else x
        )

    # Inject literal metadata as constant columns
    for col, val in literals.items():
        df[col] = val

    # Inject classes as a string column
    if classes:
        df["classes"] = str(classes)

    return df


def get_gt_raw_columns(df_gt_raw: pd.DataFrame, df_raw: pd.DataFrame, n: int = 10) -> set:
    """
    Identify which raw HF column names are used as GT by matching Unitxt task_data
    field values against raw column values.

    Text fields are matched by value-set overlap (n rows). The label field falls
    back to name-based matching when the raw label is integer-encoded (Unitxt maps
    it to strings, so value matching fails).

    Args:
        df_gt_raw: DataFrame from unitxt_load containing a task_data column.
        df_raw:    DataFrame of the raw HuggingFace dataset.
        n:         Number of rows to use for value matching.

    Returns:
        Set of raw column names that correspond to GT task_data fields.
    """
    if "task_data" not in df_gt_raw.columns or df_raw.empty:
        return set()

    task_data_rows = []
    for td in df_gt_raw["task_data"].iloc[:n]:
        if isinstance(td, str):
            td = json.loads(td)
        task_data_rows.append(td)
    if not task_data_rows:
        return set()

    gt_fields = [
        k for k in task_data_rows[0].keys()
        if k not in UNITXT_METADATA_FIELDS and not _is_annotation(k)
    ]

    raw_cols = list(df_raw.columns)
    matched = set()
    unmatched = []

    for field in gt_fields:
        gt_vals = {
            str(row[field])
            for row in task_data_rows
            if field in row and row[field] is not None
        }
        found = False
        for col in raw_cols:
            raw_vals = {str(v) for v in df_raw[col].iloc[:n] if v is not None}
            # Accept if at least half the GT values appear in the raw column
            if gt_vals and len(gt_vals & raw_vals) >= max(1, len(gt_vals) // 2):
                matched.add(col)
                found = True
                break
        if not found:
            unmatched.append(field)

    # Fallback for label fields whose values were normalized by Unitxt (int → string)
    _LABEL_NAMES = {"label", "labels", "target", "answer", "class"}
    for field in unmatched:
        if field in raw_cols:
            matched.add(field)
        elif field in _LABEL_NAMES:
            for name in _LABEL_NAMES:
                if name in raw_cols:
                    matched.add(name)
                    break

    return matched


def _raw_jaccard(gt_cols: set, pred_cols: set) -> float:
    """Jaccard similarity between two sets of raw column names."""
    if not gt_cols and not pred_cols:
        return 1.0
    intersection = len(gt_cols & pred_cols)
    union = len(gt_cols | pred_cols)
    return intersection / union if union > 0 else 0.0


def get_raw_columns(hf_name: str, config: str = None) -> set:
    """Return the set of column names from a HuggingFace dataset."""
    from src.standardize_api import _load_split
    ds = _load_split(hf_name, config)
    return set(ds.features.keys())


def check_task_match(gt_task: str, pred_task: str) -> bool:
    """Return True if gt_task and pred_task refer to the same task family."""
    gt, pred = gt_task.lower(), pred_task.lower()
    task_groups = [
        {"classification", "binary", "multi_class", "sentiment"},
        {"nli", "entailment"},
        {"qa", "question_answering"},
        {"generation", "summarization", "translation"},
    ]
    for group in task_groups:
        if any(t in gt for t in group) and any(t in pred for t in group):
            return True
    return gt == pred


def check_columns_valid(mapping: dict, raw_columns: set) -> bool:
    """Return True if every column reference in the mapping exists in raw_columns."""
    return all(
        v in raw_columns
        for k, v in mapping.items()
        if k != "task" and isinstance(v, str)
    )


def compute_mapping_recall(gt_fields: dict, pred_mapping: dict) -> float:
    """Fraction of ground-truth fields that appear in the predicted mapping."""
    if not gt_fields:
        return 0.0
    gt_keys = set(gt_fields.keys())
    pred_keys = set(pred_mapping.keys()) - {"task"}
    return len(gt_keys & pred_keys) / len(gt_keys)


def evaluate(hf_name: str, hf_config: str, card_id: str, save_dir: str | None = None,
             n_samples: int = 100, standardize_fn=None) -> dict:
    """
    Evaluate LLM standardization against Unitxt ground truth.

    Args:
        hf_name: HuggingFace dataset name (e.g., "glue")
        hf_config: Dataset config (e.g., "sst2")
        card_id: Unitxt card ID (e.g., "sst2")
        save_dir: Directory to save artifacts (optional).
                  When provided, n_samples rows are saved to CSV for manual inspection.
        n_samples: Number of rows to read for scoring AND for the saved CSVs.
                   Scoring only needs 1 GT row; increase this only if you want
                   richer CSV outputs. Default is 5.
        standardize_fn: Callable with signature (dataset_name, config) -> dict.
                        Defaults to load_standardized_dataset (API-based).

    Returns:
        dict with evaluation results
    """
    fn = standardize_fn or load_standardized_dataset
    llm_result = fn(hf_name, config=hf_config)
    mapping = llm_result.get("mapping", {})

    ds_raw = llm_result.get("dataset")
    if not ds_raw:
        raise ValueError("Agent failed to return a valid dataset object.")

    df_raw = pd.DataFrame(list(ds_raw.take(n_samples)))
    df_llm = apply_llm_mapping(df_raw, mapping)

    recipe = f"card=cards.{card_id},loader_limit={n_samples}"
    df_gt_raw = None
    for _split in ("train", "test", "validation"):
        try:
            gt_data = unitxt_load(recipe, split=_split, streaming=True)
            df_gt_raw = pd.DataFrame(list(gt_data.take(n_samples)))
            break
        except Exception:
            continue
    if df_gt_raw is None:
        raise ValueError(f"No accessible split for Unitxt card {card_id}")

    # GT: raw columns identified by value-matching task_data fields against the raw dataset
    gt_cols  = get_gt_raw_columns(df_gt_raw, df_raw)
    # Pred: raw columns referenced in the predicted mapping (non-annotation values)
    pred_cols = {
        v for k, v in mapping.items()
        if k != "task" and not _is_annotation(k) and isinstance(v, str)
    }
    struct_score = _raw_jaccard(gt_cols, pred_cols)
    score        = round(struct_score, 3)

    if save_dir:
        df_gt = extract_unitxt_standardized(df_gt_raw)
        os.makedirs(save_dir, exist_ok=True)
        df_llm.to_csv(f"{save_dir}/llm_standardized.csv", index=False)
        df_gt.to_csv(f"{save_dir}/unitxt_standardized.csv", index=False)

    return {
        "dataset":      card_id,
        "score":        score,
        "struct_score": round(struct_score, 3),
        "gt_cols":      sorted(gt_cols),
        "pred_cols":    sorted(pred_cols),
        "mapping":      mapping,
        "eval_card":    llm_result.get("code", ""),
        "df_llm":       df_llm,
        "error":        None
    }