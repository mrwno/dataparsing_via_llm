"""
LLM-based dataset standardization for Unitxt using OpenRouter.
Replaces local inference with API calls to powerful models.
"""
import os
import json
from datasets import load_dataset, Dataset
from unitxt import get_from_catalog
from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_ID = "anthropic/claude-opus-4.5"

def _get_client():
    """Initialize OpenRouter client safely."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Error: OPENROUTER_API_KEY environment variable is not set.")
    
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

def _infer_mapping(features: dict, sample_rows: list, instruction: str = None) -> dict:
    """Use OpenRouter API to infer the mapping. Returns None if inference fails."""
    client = _get_client()
    column_names = list(features.keys())
    
    # SYSTEM: Define role but DO NOT give specific schema examples (User request)
    system_message = (
        "You are an expert Data Scientist specializing in the 'Unitxt' library for NLP. "
        "Your job is to inspect raw dataset samples and deduce the standard Unitxt fields."
    )
    
    # USER: Pure data + High-level goal. No spoon-feeding.
    user_prompt = f"""
    Analyze the following dataset samples and determine the underlying NLP task and column mapping.

    DATASET METADATA:
    - Available Columns: {column_names}
    - Data Types: {json.dumps({k: str(v) for k, v in features.items()})}
    - 10 Samples:
    {json.dumps(sample_rows[:10], indent=2, default=str)}
    {f'- Additional Context: {instruction}' if instruction else ''}

    YOUR MISSION:
    1. Deduce the NLP task type (e.g., classification, regression, translation, etc.) based solely on the data patterns.
    2. Map the raw column names to the canonical standard fields used in Unitxt cards for that specific task.
    3. Return a single valid JSON object.

    OUTPUT FORMAT:
    {{
        "task": "<detected_task_type>",
        "<standard_unitxt_field_1>": "<raw_column_name>",
        "<standard_unitxt_field_2>": "<raw_column_name>"
    }}
    """

    try:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}, 
            temperature=0.0 
        )
        
        response_content = completion.choices[0].message.content
        
        # Clean potential markdown wrapping
        if "```json" in response_content:
            response_content = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            response_content = response_content.split("```")[1].split("```")[0].strip()
            
        parsed = json.loads(response_content)
        
        if parsed and "task" in parsed:
            return parsed
            
    except Exception as e:
        print(f"API Error or JSON Parsing failed: {e}")
        pass
    
    return None


def _score_mapping(dataset: Dataset, mapping: dict, n: int = 10) -> float:
    """Score the validity of a mapping by checking N sample rows."""
    if not mapping:
        return 0.0
        
    try:
        samples = list(dataset.take(n)) if hasattr(dataset, 'take') else dataset[:n]
        if isinstance(samples, dict):
            samples = [dict(zip(samples.keys(), vals)) for vals in zip(*samples.values())]
        
        valid = 0
        required_fields = [v for k, v in mapping.items() if k != "task"]
        
        for row in samples:
            if all(col in row for col in required_fields):
                valid += 1
        
        return valid / n
    except Exception:
        return 0.0


def _generate_code(mapping: dict) -> str:
    """Generate Unitxt preprocess_steps code string."""
    if not mapping:
        return "# Mapping failed"
        
    steps = []
    for field_name, source_col in mapping.items():
        if field_name == "task" or not isinstance(source_col, str):
            continue
        if source_col != field_name:
            steps.append(f"Rename(field='{source_col}', to_field='{field_name}')")
    
    return ", ".join(steps) if steps else "# No rename steps needed"


def standardize(dataset, instruction: str = None) -> dict:
    """
    Standardize a HuggingFace dataset into Unitxt format.
    """
    if isinstance(dataset, str):
        ds = load_dataset(dataset, split="train", streaming=True)
    else:
        ds = dataset
    
    features = ds.features
    samples = list(ds.take(5))
    
    mapping = _infer_mapping(features, samples, instruction)
    
    if mapping is None:
        print(f"Warning: Failed to infer mapping for columns: {list(features.keys())}")
        return {
            "mapping": {},
            "code": "# Error: LLM/API failed",
            "score": 0.0,
            "dataset": ds,
        }

    score = _score_mapping(ds, mapping)
    print(f"Mapping: {mapping} (score: {score:.2f})")
    
    # Map inferred task to actual Unitxt catalog paths
    task_type = mapping.get("task", "classification").lower()
    
    # We still need this internal mapping to load the right Unitxt card
    # But the LLM generated the task_type purely from its own logic
    task_mapping = {
        "classification": "tasks.classification.binary",
        "nli": "tasks.classification.multi_class",
        "generation": "tasks.generation",
        "summarization": "tasks.summarization.abstractive",
        "translation": "tasks.translation",
        "regression": "tasks.regression"
    }
    
    selected_task = task_mapping.get(task_type, "tasks.classification.binary")
    
    try:
        get_from_catalog(selected_task)
    except Exception:
        pass

    return {
        "mapping": mapping,
        "code": _generate_code(mapping),
        "score": score,
        "dataset": ds,
    }


def load_standardized_dataset(dataset_name: str, config: str = None, instruction: str = None):
    """Convenience function."""
    if config:
        ds = load_dataset(dataset_name, config, split="train", streaming=True)
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True)
    
    return standardize(ds, instruction)