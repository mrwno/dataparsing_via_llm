"""API-based dataset standardization for Unitxt using OpenRouter."""
import os
import re
import json
from datasets import load_dataset
from unitxt import get_from_catalog
from openai import OpenAI
from src.utils import generate_code, score_mapping


DEFAULT_MODEL_ID = "anthropic/claude-opus-4.6"

TASK_MAPPING = {
    "classification": "tasks.classification.binary",
    "nli": "tasks.classification.multi_class",
    "generation": "tasks.generation",
    "summarization": "tasks.summarization.abstractive",
    "translation": "tasks.translation",
    "regression": "tasks.regression",
}


def _get_client() -> OpenAI:
    """
    Initialize OpenRouter client safely.

    Raises:
        ValueError: If OPENROUTER_API_KEY environment variable is not set.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def _infer_mapping(features: dict, sample_rows: list, instruction: str = None,
                   model_id: str | None = None) -> dict | None:
    """
    Use OpenRouter API to infer the Unitxt field mapping.

    Args:
        features: Dictionary of dataset features with column names and types.
        sample_rows: List of sample data rows from the dataset.
        instruction: Optional additional context for the LLM.
        model_id: OpenRouter model ID. Defaults to DEFAULT_MODEL_ID.

    Returns:
        Parsed mapping dict, or None if inference fails.
    """
    client = _get_client()
    model_id = model_id or DEFAULT_MODEL_ID
    column_names = list(features.keys())

    system_message = (
        "You are an expert Data Scientist specializing in the 'Unitxt' library for NLP. "
        "Your job is to inspect raw dataset samples and deduce the standard Unitxt fields."
    )

    user_prompt = f"""
    Analyze the following dataset samples and determine the underlying NLP task and column mapping.

    DATASET METADATA:
    - Available Columns: {column_names}
    - Data Types: {json.dumps({k: str(v) for k, v in features.items()})}
    - 10 Samples:
    {json.dumps(sample_rows[:10], indent=2, default=str)}
    {f'- Additional Context: {instruction}' if instruction else ''}

    YOUR MISSION:
    1. Deduce the NLP task type (e.g. classification, nli, regression, translation, summarization) from the data patterns.
    2. Map the raw column names to the canonical standard fields for that task.
    3. Return a single valid JSON object — no explanation, no markdown.

    If the task has ONE text column:
    {{
        "task": "<detected_task_type>",
        "text": "<raw_column_name>",
        "label": "<raw_column_name_with_label>"
    }}

    If the task has TWO text columns (e.g. NLI, similarity, QA):
    {{
        "task": "<detected_task_type>",
        "text_a": "<raw_column_name_1>",
        "text_b": "<raw_column_name_2>",
        "label": "<raw_column_name_with_label>"
    }}

    RULES:
    - Use ONLY raw column names as values — never class names or literal strings.
    - "label" must be the raw column name containing the target label.
    - NEVER mix both formats: if you use "text", do NOT add "text_a" or "text_b".
    """

    # Only Claude and OpenAI models reliably support json_object format
    _JSON_FORMAT_PREFIXES = ("anthropic/", "openai/")
    use_json_format = any(model_id.startswith(p) for p in _JSON_FORMAT_PREFIXES)

    _messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]

    try:
        if use_json_format:
            completion = client.chat.completions.create(
                model=model_id,
                messages=_messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
        else:
            completion = client.chat.completions.create(
                model=model_id,
                messages=_messages,
                temperature=0.0,
            )
        response_content = completion.choices[0].message.content

        # Try progressively more lenient extraction
        parsed = None
        for attempt in (response_content,):
            # 1. Strip markdown fences
            if "```json" in attempt:
                attempt = attempt.split("```json")[1].split("```")[0].strip()
            elif "```" in attempt:
                attempt = attempt.split("```")[1].split("```")[0].strip()
            try:
                parsed = json.loads(attempt)
                break
            except json.JSONDecodeError:
                pass
            # 2. Extract first {...} block via regex
            m = re.search(r"\{.*\}", attempt, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                    break
                except json.JSONDecodeError:
                    pass

        if parsed and "task" in parsed:
            return parsed
    except Exception as e:
        print(f"API Error or JSON Parsing failed: {e}")
    return None


_FALLBACK_SPLITS = ["train", "test", "validation"]


def _load_split(name: str, config: str | None) -> object:
    """Load a HuggingFace dataset trying train → test → validation.

    If config is None and the dataset requires one, the first suggested
    config is extracted from the error message and retried automatically.
    """
    last_err = None
    for split in _FALLBACK_SPLITS:
        try:
            return load_dataset(name, config, split=split, streaming=True, trust_remote_code=True) if config else \
                   load_dataset(name, split=split, streaming=True, trust_remote_code=True)
        except Exception as e:
            err_str = str(e)
            if config is None and "Config name is missing" in err_str:
                # Extract suggested configs (quoted names in the error message)
                candidates = [c for c in re.findall(r"'([^']+)'", err_str) if c != name]
                if candidates:
                    try:
                        return load_dataset(name, candidates[0], split=split, streaming=True, trust_remote_code=True)
                    except Exception as e2:
                        last_err = e2
                else:
                    last_err = e
                    break  # No candidates — retrying splits won't help
            else:
                last_err = e
    raise ValueError(f"No accessible split for {name}/{config}: {last_err}")


def standardize(dataset, instruction: str = None, model_id: str | None = None) -> dict:
    """
    Standardize a HuggingFace dataset into Unitxt format using the OpenRouter API.

    Args:
        dataset: Dataset name (str) or dataset object.
        instruction: Optional additional context for the LLM.
        model_id: OpenRouter model ID. Defaults to DEFAULT_MODEL_ID.

    Returns:
        Dictionary with keys: mapping, code, score, dataset.
    """
    ds = load_dataset(dataset, split="train", streaming=True) if isinstance(dataset, str) else dataset
    features = ds.features
    samples = list(ds.take(5))
    # features can be None for some streaming datasets — infer from first row
    if features is None and samples:
        features = {k: type(v).__name__ for k, v in samples[0].items()}
    if not features:
        return {"mapping": {}, "code": "# Error: no features", "score": 0.0, "dataset": ds}
    mapping = _infer_mapping(features, samples, instruction, model_id=model_id)

    if mapping is None:
        print(f"Warning: Failed to infer mapping for columns: {list(features.keys())}")
        return {"mapping": {}, "code": "# Error: LLM/API failed", "score": 0.0, "dataset": ds}

    sc = score_mapping(ds, mapping)
    print(f"Mapping: {mapping} (score: {sc:.2f})")

    selected_task = TASK_MAPPING.get(mapping.get("task", "").lower(), "tasks.classification.binary")
    try:
        get_from_catalog(selected_task)
    except Exception:
        pass

    return {"mapping": mapping, "code": generate_code(mapping), "score": sc, "dataset": ds}


def load_standardized_dataset(dataset_name: str, config: str | None = None,
                              instruction: str | None = None,
                              model_id: str | None = None) -> dict:
    """
    Load and standardize a dataset using the OpenRouter API.

    Args:
        dataset_name: HuggingFace dataset name.
        config: Optional dataset configuration name.
        instruction: Optional additional context for the LLM.
        model_id: OpenRouter model ID. Defaults to DEFAULT_MODEL_ID.

    Returns:
        Dictionary with keys: mapping, code, score, dataset.
    """
    ds = _load_split(dataset_name, config)
    return standardize(ds, instruction, model_id=model_id)
