"""Local HuggingFace model-based dataset standardization for Unitxt."""
import json
import re
from datasets import load_dataset
from unitxt import get_from_catalog
from src.utils import extract_json, generate_code, score_mapping


_FALLBACK_SPLITS = ["train", "test", "validation"]


def _load_split(name: str, config: str | None) -> object:
    """Load a HuggingFace dataset trying train → test → validation."""
    last_err = None
    for split in _FALLBACK_SPLITS:
        try:
            return load_dataset(name, config, split=split, streaming=True, trust_remote_code=True) if config else \
                   load_dataset(name, split=split, streaming=True, trust_remote_code=True)
        except Exception as e:
            err_str = str(e)
            if config is None and "Config name is missing" in err_str:
                candidates = [c for c in re.findall(r"'([^']+)'", err_str) if c != name]
                if candidates:
                    try:
                        return load_dataset(name, candidates[0], split=split, streaming=True, trust_remote_code=True)
                    except Exception as e2:
                        last_err = e2
                else:
                    last_err = e
                    break
            else:
                last_err = e
    raise ValueError(f"No accessible split for {name}/{config}: {last_err}")


LOCAL_MODEL_ID = "Qwen/Qwen3-0.6B"
_local_pipeline = None

TASK_MAPPING = {
    "classification": "tasks.classification.binary",
    "nli": "tasks.classification.multi_class",
    "generation": "tasks.generation",
    "summarization": "tasks.summarization.abstractive",
    "translation": "tasks.translation",
    "regression": "tasks.regression",
}


def _get_local_pipeline(model_id: str = LOCAL_MODEL_ID):
    """
    Lazily load and cache the local HuggingFace text-generation pipeline.

    Args:
        model_id: HuggingFace model identifier.

    Returns:
        A transformers pipeline ready for text generation.
    """
    global _local_pipeline
    if _local_pipeline is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        print(f"Loading local model '{model_id}' (first call only)…")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto"
        )
        _local_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        print("Local model loaded.")
    return _local_pipeline


def _infer_mapping_local(
    features: dict, sample_rows: list, instruction: str | None = None, model_id: str = LOCAL_MODEL_ID
) -> dict | None:
    """
    Use a local HuggingFace model to infer the Unitxt field mapping.

    Args:
        features: Dictionary of dataset features with column names and types.
        sample_rows: List of sample data rows from the dataset.
        instruction: Optional additional context for the LLM.
        model_id: HuggingFace model identifier.

    Returns:
        Parsed mapping dict, or None if inference fails.
    """
    pipe = _get_local_pipeline(model_id)
    column_names = list(features.keys())

    system_message = (
        "You are an expert Data Scientist. "
        "Your job is to inspect raw dataset samples and deduce the NLP task and column mapping. "
        "Respond ONLY with a single valid JSON object — no explanation, no markdown."
    )

    user_prompt = f"""Analyze the following dataset samples and determine the NLP task and column mapping.

DATASET METADATA:
- Available Columns: {column_names}
- Data Types: {json.dumps({k: str(v) for k, v in features.items()})}
- 10 Samples:
{json.dumps(sample_rows[:10], indent=2, default=str)}
{f'- Additional Context: {instruction}' if instruction else ''}

YOUR MISSION:
1. Deduce the NLP task type (e.g. classification, nli, regression, translation, summarization).
2. Map raw column names to canonical fields.
3. Return ONLY a valid JSON object — no explanation, no markdown.

OUTPUT FORMAT — choose EXACTLY ONE of the two formats below:

If the task has ONE text column:
{{
    "task": "<detected_task_type>",
    "text": "<raw_column_name>",
    "label": "<raw_column_name_with_label>"
}}

If the task has TWO text columns (e.g. NLI, similarity):
{{
    "task": "<detected_task_type>",
    "text_a": "<raw_column_name_1>",
    "text_b": "<raw_column_name_2>",
    "label": "<raw_column_name_with_label>"
}}

RULES:
- Use ONLY raw column names as values — never class names or literal strings.
- NEVER mix both formats. If you use "text", do NOT add "text_a" or "text_b".
- "label" must be the raw column name containing the target label."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]

    try:
        output = pipe(messages, max_new_tokens=1024, do_sample=False)
        generated = output[0]["generated_text"]
        response_text = generated[-1].get("content", "") if isinstance(generated, list) else str(generated)
        #print(f"[local model raw output] {response_text!r}")
        parsed = extract_json(response_text)
        if parsed and "task" in parsed:
            return parsed
        print(f"[local model] JSON extraction failed or 'task' key missing. parsed={parsed}")
    except Exception as e:
        print(f"Local inference error: {e}")
    return None


def standardize_local(dataset, instruction: str | None = None, model_id: str = LOCAL_MODEL_ID) -> dict:
    """
    Standardize a HuggingFace dataset into Unitxt format using a local LLM.

    Args:
        dataset: Dataset name (str) or dataset object.
        instruction: Optional additional context for the LLM.
        model_id: HuggingFace model identifier for local inference.

    Returns:
        Dictionary with keys: mapping, code, score, dataset.
    """
    ds = _load_split(dataset, None) if isinstance(dataset, str) else dataset
    features = ds.features
    samples = list(ds.take(5))
    mapping = _infer_mapping_local(features, samples, instruction, model_id)

    if mapping is None:
        print(f"Warning: Local model failed to infer mapping for columns: {list(features.keys())}")
        return {"mapping": {}, "code": "# Error: local LLM failed", "score": 0.0, "dataset": ds}

    sc = score_mapping(ds, mapping)
    print(f"Mapping: {mapping} (score: {sc:.2f})")

    selected_task = TASK_MAPPING.get(mapping.get("task", "").lower(), "tasks.classification.binary")
    try:
        get_from_catalog(selected_task)
    except Exception:
        pass

    return {"mapping": mapping, "code": generate_code(mapping), "score": sc, "dataset": ds}


def load_standardized_dataset_local(
    dataset_name: str, config: str | None = None, instruction: str | None = None, model_id: str = LOCAL_MODEL_ID
) -> dict:
    """
    Load and standardize a dataset using the local HuggingFace model.

    Args:
        dataset_name: HuggingFace dataset name.
        config: Optional dataset configuration name.
        instruction: Optional additional context for the LLM.
        model_id: HuggingFace model identifier for local inference.

    Returns:
        Dictionary with keys: mapping, code, score, dataset.
    """
    ds = _load_split(dataset_name, config)
    return standardize_local(ds, instruction, model_id)
