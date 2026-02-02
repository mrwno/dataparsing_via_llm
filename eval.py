"""
LLM-based dataset standardization for Unitxt.
Uses a small LLM to automatically map HuggingFace datasets to Unitxt format.
"""
import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from unitxt import get_from_catalog
from unitxt.card import TaskCard
from unitxt.loaders import LoadFromDictionary

# Model configuration
MODEL_ID = "google/gemma-3-1b-it"
_tokenizer = None
_model = None


def _load_model():
    """Lazy load the LLM model."""
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="auto"
        )
    return _tokenizer, _model


def _infer_mapping(features: dict, sample_rows: list, instruction: str = None) -> dict:
    """Use LLM to infer the mapping from dataset columns to Unitxt fields."""
    tokenizer, model = _load_model()
    column_names = list(features.keys())
    
    # Zero-shot prompt: test LLM's raw Unitxt knowledge
    prompt_template = f"""You are an expert in the 'Unitxt' library for NLP dataset standardization.

DATASET INFO:
- Columns: {column_names}
- Types: {json.dumps({k: str(v) for k, v in features.items()})}
- Samples: {json.dumps(sample_rows[:5], indent=2, default=str)}
{f'- User hint: {instruction}' if instruction else ''}

Analyze the dataset columns and map them to the corresponding Unitxt task fields based on your knowledge of the library.
Infer the NLP task type and create a mapping from raw columns to Unitxt fields.

Return a JSON object: {{"task": "<task_type>", "<unitxt_field>": "<column_name>", ...}}
Use EXACT column names from: {column_names}

Return ONLY valid JSON:"""
    
    # Direct prompt usage (Removed gepa optimization)
    prompt = prompt_template
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON from response
    try:
        json_start = response.rfind("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            parsed = json.loads(response[json_start:json_end])
            if parsed and "task" in parsed and len(parsed) > 1:
                return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Fallback: heuristic-based detection
    cols = set(features.keys())
    if cols & {"premise", "sentence1"}:
        return {"task": "nli", "text_a": "premise" if "premise" in cols else "sentence1",
                "text_b": "hypothesis" if "hypothesis" in cols else "sentence2", "label": "label"}
    if "label" in cols:
        text_col = next((c for c in cols if c in ["text", "sentence", "content"]), list(cols)[0])
        return {"task": "classification", "text": text_col, "label": "label"}
    return {"task": "generation", "input": list(cols)[0], "output": list(cols)[-1]}


def _score_mapping(dataset: Dataset, mapping: dict, n: int = 5) -> float:
    """Score the validity of a mapping by checking N sample rows."""
    try:
        samples = list(dataset.take(n)) if hasattr(dataset, 'take') else dataset[:n]
        if isinstance(samples, dict):
            samples = [dict(zip(samples.keys(), vals)) for vals in zip(*samples.values())]
        
        valid = 0
        required_fields = [v for k, v in mapping.items() if k != "task"]
        
        for row in samples:
            if all(field in row and row[field] is not None for field in required_fields):
                valid += 1
        
        return valid / n
    except Exception:
        return 0.0


def _generate_code(mapping: dict) -> str:
    """Generate Unitxt preprocess_steps code string directly from LLM output (no correction)."""
    steps = []
    for field_name, source_col in mapping.items():
        if field_name == "task" or not isinstance(source_col, str):
            continue
        # Output exactly what LLM suggested - no field name correction
        if source_col != field_name:
            steps.append(f"Rename(field='{source_col}', to_field='{field_name}')")
    
    return ", ".join(steps) if steps else "# No rename steps needed"


def standardize(dataset, instruction: str = None) -> TaskCard:
    """
    Standardize a HuggingFace dataset into Unitxt format.
    
    Args:
        dataset: HF dataset name (str) or Dataset object
        instruction: Optional instruction to guide the LLM mapping
    
    Returns:
        TaskCard ready for Unitxt processing
    """
    # Load dataset if string
    if isinstance(dataset, str):
        ds = load_dataset(dataset, split="train", streaming=True)
    else:
        ds = dataset
    
    # Get features and samples
    features = ds.features
    samples = list(ds.take(5))
    
    # Infer mapping with LLM
    mapping = _infer_mapping(features, samples, instruction)
    
    # Score the mapping
    score = _score_mapping(ds, mapping)
    print(f"Mapping: {mapping} (score: {score:.2f})")
    
    if score < 0.5:
        print("Warning: Low confidence mapping. Consider providing an instruction.")
    
    # Create Unitxt card based on task type
    task_type = mapping.get("task", "classification")
    
    # Map to Unitxt task (using correct catalog paths)
    task_mapping = {
        "classification": "tasks.classification.binary",
        "nli": "tasks.classification.multi_class",
        "generation": "tasks.generation",
    }
    
    task = get_from_catalog(task_mapping.get(task_type, "tasks.classification"))
    
    return {
        "mapping": mapping,
        "code": _generate_code(mapping),
        "score": score,
        "dataset": ds,
    }


def load_standardized_dataset(dataset_name: str, config: str = None, instruction: str = None):
    """
    Convenience function to load and standardize a dataset in one call.
    """
    if config:
        ds = load_dataset(dataset_name, config, split="train", streaming=True)
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True)
    
    return standardize(ds, instruction)