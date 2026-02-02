"""End-to-end evaluation pipeline for Unitxt LLM Agent."""
import os
import sys
import pandas as pd
import wandb
from unitxt import load_dataset as unitxt_load
from standardize import load_standardized_dataset

# ============================================================================
# CONFIGURATION
# ============================================================================
os.makedirs("results", exist_ok=True)
WANDB_PROJECT = "unitxt-llm-agent"

# Unitxt internal columns to ignore during evaluation
EXCLUDED_COLS = {
    'recipes', 'metrics', 'postprocessors', 'data_classification_policy', 
    'source', 'split', 'group', 'subset', 'task_data'
}

GLUE_DATASETS = [
    {"card_id": "sst2", "hf_name": "glue", "hf_config": "sst2"},
    {"card_id": "mrpc", "hf_name": "glue", "hf_config": "mrpc"},
    {"card_id": "qnli", "hf_name": "glue", "hf_config": "qnli"},
    {"card_id": "mnli", "hf_name": "glue", "hf_config": "mnli"},
    {"card_id": "wnli", "hf_name": "glue", "hf_config": "wnli"},
]

def check_api_key():
    """Ensure API key is set before starting expensive tests."""
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY is missing.")
        print("   Run: export OPENROUTER_API_KEY='your_key_here'")
        sys.exit(1)

def compute_score(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    """Compare columns overlap (Jaccard Index)."""
    # 1. Get columns
    gt_cols = set(gt_df.columns)
    pred_cols = set(pred_df.columns)
    
    # 2. Filter out Unitxt internal metadata (starting with _ or in blocklist)
    gt_cols = {c for c in gt_cols if not c.startswith("_") and c not in EXCLUDED_COLS}
    
    # 3. Compute Jaccard Index
    intersection = len(gt_cols & pred_cols)
    union = len(gt_cols | pred_cols)
    
    return intersection / union if union > 0 else 0.0

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    check_api_key()
    
    run = wandb.init(project=WANDB_PROJECT, job_type="eval", name="glue_eval_run_openrouter")
    print(f"Testing on {len(GLUE_DATASETS)} GLUE datasets: {[d['card_id'] for d in GLUE_DATASETS]}")
    
    table_data = []

    for exp in GLUE_DATASETS:
        card_id, hf_name, hf_config = exp["card_id"], exp["hf_name"], exp["hf_config"]
        print(f"\n{'='*40}\nProcessing: {card_id}")
        
        try:
            # -------------------------------------------------------
            # Step A: LLM Processing (Agent)
            # -------------------------------------------------------
            # This now calls the OpenRouter API via standardize.py
            llm_result = load_standardized_dataset(hf_name, config=hf_config)
            mapping = llm_result.get("mapping", {})
            
            # Optimization: Reuse the dataset object returned by the agent
            ds_raw = llm_result.get("dataset")
            if not ds_raw:
                raise ValueError("Agent failed to return a valid dataset object.")
                
            df_raw = pd.DataFrame(list(ds_raw.take(50)))
            
            # Apply LLM renaming logic
            rename_dict = {
                v: k for k, v in mapping.items() 
                if k != "task" and isinstance(v, str) and v in df_raw.columns
            }
            df_llm = df_raw.rename(columns=rename_dict)
            
            # Keep only the mapped columns for clean comparison
            mapped_cols = list(rename_dict.values())
            if mapped_cols:
                df_llm = df_llm[mapped_cols]

            # -------------------------------------------------------
            # Step B: Unitxt Ground Truth
            # -------------------------------------------------------
            # Load the official Unitxt card output
            recipe = f"card=cards.{card_id}"
            gt_data = unitxt_load(recipe, split="train", streaming=True)
            df_gt = pd.DataFrame(list(gt_data.take(50)))

            # -------------------------------------------------------
            # Step C: Evaluation
            # -------------------------------------------------------
            score = compute_score(df_gt, df_llm)

            # Save artifacts
            save_path = f"results/{card_id}"
            os.makedirs(save_path, exist_ok=True)
            df_llm.to_csv(f"{save_path}/llm_processed.csv", index=False)
            df_gt.to_csv(f"{save_path}/unitxt_ground_truth.csv", index=False)

            # Prepare log data
            llm_cols_str = str(list(df_llm.columns))
            gt_cols_str = str([c for c in df_gt.columns if c not in EXCLUDED_COLS and not c.startswith("_")])

            wandb.log({
                "dataset": card_id, 
                "score": score, 
                "llm_cols": llm_cols_str, 
                "gt_cols": gt_cols_str
            })
            
            table_data.append([card_id, score, llm_cols_str, gt_cols_str])
            print(f"✅ {card_id} | Score: {score:.3f}")

        except Exception as e:
            print(f"❌ {card_id} failed: {e}")
            wandb.log({"dataset": card_id, "score": 0.0, "error": str(e)})
            table_data.append([card_id, 0.0, "Error", str(e)])
            continue

    # Summary Table
    columns = ["Dataset", "Score", "LLM Predicted Columns", "GT Columns"]
    wandb.log({"evaluation_summary": wandb.Table(data=table_data, columns=columns)})
    
    run.finish()
    print("\n🎉 Evaluation complete. Check wandb for results.")

if __name__ == "__main__":
    main()