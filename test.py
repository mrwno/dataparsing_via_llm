"""End-to-end evaluation pipeline for Unitxt LLM Agent."""

import os
import sys
import json
import pandas as pd
from unitxt import load_dataset as unitxt_load
from standardize import load_standardized_dataset


# ============================================================================
# CONFIGURATION
# ============================================================================
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

GLUE_DATASETS = [
    {"card_id": "sst2", "hf_name": "glue", "hf_config": "sst2"},
    {"card_id": "mrpc", "hf_name": "glue", "hf_config": "mrpc"},
    {"card_id": "qnli", "hf_name": "glue", "hf_config": "qnli"},
    {"card_id": "mnli", "hf_name": "glue", "hf_config": "mnli"},
    {"card_id": "wnli", "hf_name": "glue", "hf_config": "wnli"},
]

# Fields inside task_data that are Unitxt metadata, not actual data columns
UNITXT_METADATA_FIELDS = {'metadata', 'data_classification_policy'}


def check_api_key():
    """Ensure API key is set before starting expensive tests."""
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY is missing.")
        print("   Run: export OPENROUTER_API_KEY='your_key_here'")
        sys.exit(1)


def extract_unitxt_standardized(unitxt_df: pd.DataFrame) -> pd.DataFrame:
    """Extract clean standardized data from Unitxt's task_data column."""
    if 'task_data' not in unitxt_df.columns:
        return pd.DataFrame()
    
    rows = []
    for task_data in unitxt_df['task_data']:
        # Parse if string
        if isinstance(task_data, str):
            task_data = json.loads(task_data)
        
        # Filter out metadata fields
        clean_row = {k: v for k, v in task_data.items() if k not in UNITXT_METADATA_FIELDS}
        rows.append(clean_row)
    
    return pd.DataFrame(rows)


def extract_task_data_fields(unitxt_df: pd.DataFrame) -> set:
    """Extract the actual data field names from Unitxt's task_data column."""
    if 'task_data' not in unitxt_df.columns:
        return set()
    
    sample = unitxt_df['task_data'].iloc[0]
    if isinstance(sample, str):
        sample = json.loads(sample)
    
    return {k for k in sample.keys() if k not in UNITXT_METADATA_FIELDS}


def apply_llm_mapping(raw_df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Apply LLM mapping to raw dataset and return standardized DataFrame."""
    # Build rename dict: raw_column -> standard_field
    rename_dict = {
        v: k for k, v in mapping.items() 
        if k != "task" and isinstance(v, str) and v in raw_df.columns
    }
    
    # Apply renaming
    df_standardized = raw_df.rename(columns=rename_dict)
    
    # Keep only mapped columns
    mapped_cols = list(rename_dict.values())
    if mapped_cols:
        df_standardized = df_standardized[mapped_cols]
    
    return df_standardized


def compute_score(gt_fields: set, pred_fields: set) -> float:
    """Jaccard Index between ground truth and predicted field sets."""
    intersection = len(gt_fields & pred_fields)
    union = len(gt_fields | pred_fields)
    return intersection / union if union > 0 else 0.0


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    check_api_key()
    
    print(f"Testing on {len(GLUE_DATASETS)} GLUE datasets: {[d['card_id'] for d in GLUE_DATASETS]}")
    
    results = []

    for exp in GLUE_DATASETS:
        card_id, hf_name, hf_config = exp["card_id"], exp["hf_name"], exp["hf_config"]
        print(f"\n{'='*40}\nProcessing: {card_id}")
        
        try:
            # Step A: LLM Processing
            llm_result = load_standardized_dataset(hf_name, config=hf_config)
            mapping = llm_result.get("mapping", {})
            
            # Get raw dataset and apply LLM standardization
            ds_raw = llm_result.get("dataset")
            if not ds_raw:
                raise ValueError("Agent failed to return a valid dataset object.")
            
            df_raw = pd.DataFrame(list(ds_raw.take(50)))
            df_llm = apply_llm_mapping(df_raw, mapping)
            
            # Extract LLM predicted fields
            llm_fields = {k for k in mapping.keys() if k != "task"}

            # Step B: Unitxt Ground Truth
            recipe = f"card=cards.{card_id}"
            gt_data = unitxt_load(recipe, split="train", streaming=True)
            df_gt_raw = pd.DataFrame(list(gt_data.take(50)))
            
            # Extract clean standardized data from task_data
            df_gt = extract_unitxt_standardized(df_gt_raw)
            gt_fields = extract_task_data_fields(df_gt_raw)

            # Step C: Evaluation
            score = compute_score(gt_fields, llm_fields)

            # Step D: Save artifacts
            save_path = f"{RESULTS_DIR}/{card_id}"
            os.makedirs(save_path, exist_ok=True)
            
            df_llm.to_csv(f"{save_path}/llm_standardized.csv", index=False)
            df_gt.to_csv(f"{save_path}/unitxt_standardized.csv", index=False)

            results.append({
                "dataset": card_id,
                "score": round(score, 3),
                "llm_fields": sorted(llm_fields),
                "gt_fields": sorted(gt_fields),
                "mapping": json.dumps(mapping),
                "eval_card": llm_result.get("code", ""),
                "error": None
            })
            
            print(f"✅ {card_id} | Score: {score:.3f}")
            print(f"   LLM: {sorted(llm_fields)}")
            print(f"   GT:  {sorted(gt_fields)}")

        except Exception as e:
            print(f"❌ {card_id} failed: {e}")
            results.append({
                "dataset": card_id,
                "score": 0.0,
                "llm_fields": [],
                "gt_fields": [],
                "mapping": "{}",
                "eval_card": "",
                "error": str(e)
            })
            continue

    # Save final results
    df_results = pd.DataFrame(results)
    output_path = f"{RESULTS_DIR}/evaluation_results.csv"
    df_results.to_csv(output_path, index=False)
    
    print(f"\n{'='*40}")
    print(f"🎉 Evaluation complete. Results saved to: {output_path}")
    print(df_results[["dataset", "score", "llm_fields", "gt_fields"]].to_string(index=False))


if __name__ == "__main__":
    main()