# dataset-preprocessing-agent

**Automated dataset standardization using LLM agents.**

Every HuggingFace dataset has a unique schema (`tweet_text`, `review_body`, `sentence1`, …), making it hard to reuse models across tasks without manual mapping work. This library automates that step: given a raw dataset, an LLM inspects a small sample and produces a JSON mapping of raw column names to a canonical schema, evaluated against [Unitxt](https://github.com/IBM/unitxt) and [tasksource](https://github.com/sileod/tasksource) ground truths.

---

## Installation

```bash
pip install dataset-preprocessing-agent
```

For notebook visualization support:

```bash
pip install "dataset-preprocessing-agent[notebook]"
```

**Python 3.10+ required.**

For the API backend, export your OpenRouter key:

```bash
export OPENROUTER_API_KEY="your_key_here"
```

---

## Quick Start

### API backend (OpenRouter)

Default model: `anthropic/claude-opus-4.6`

```python
from dataset_preprocessing_agent.standardize_api import load_standardized_dataset

# default model
result = load_standardized_dataset("glue", config="sst2")
# custom model
result = load_standardized_dataset("glue", config="sst2", model_id="mistralai/mistral-7b-instruct")
print(result["mapping"])
# {"task": "classification", "text": "sentence", "label": "label"}
```

### Local backend (HuggingFace model)

Default model: `Qwen/Qwen3-0.6B`

```python
from dataset_preprocessing_agent.standardize_local import load_standardized_dataset_local

# default model
result = load_standardized_dataset_local("glue", config="sst2")
# custom model
result = load_standardized_dataset_local("glue", config="sst2", model_id="Qwen/Qwen2.5-1.5B-Instruct")
print(result["mapping"])
# {"task": "classification", "text": "sentence", "label": "label"}
```

### Evaluate against Unitxt ground truth

```python
from dataset_preprocessing_agent.eval import evaluate

result = evaluate(hf_name="glue", hf_config="sst2", card_id="sst2")
print(result["score"])        
print(result["gt_cols"])      # e.g. ['label', 'sentence']
print(result["pred_cols"])    # e.g. ['label', 'sentence']
```

### Evaluate against tasksource ground truth

```python
from dataset_preprocessing_agent.eval_ts import evaluate_ts

result = evaluate_ts("glue", "rte")
print(result["score"])
print(result["ts_gt"])        # GT mapping from tasksource preprocessing
```

---

## Architecture

The pipeline runs in three stages:

1. **Standardization** — an LLM inspects 5–10 raw samples and outputs a JSON mapping of raw column names to canonical fields (`task`, `text` / `text_a` + `text_b`, `label`).
2. **Mapping application** — `apply_llm_mapping` renames columns and converts integer labels to class name strings.
3. **Evaluation** — the predicted raw column set is compared to the ground-truth column set using Jaccard similarity on raw HuggingFace column names.

### Backends

| Module | Backend |
|--------|---------|
| `standardize_api` | Cloud LLM via OpenRouter API |
| `standardize_local` | Local HuggingFace model |

### Baselines

| Baseline | Method |
|----------|--------|
| `baseline_keyword_match` | Synonym dictionary matching |
| `baseline_embedding_match` | Cosine similarity via `all-MiniLM-L6-v2` |

### Evaluation backends

| Module | Ground truth |
|--------|-------------|
| `eval` | Unitxt task cards |
| `eval_ts` | tasksource preprocessing objects |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `unitxt` | Ground-truth task cards |
| `tasksource` | Ground-truth preprocessing objects |
| `datasets` | HuggingFace dataset loading |
| `transformers` + `torch` + `accelerate` | Local model inference |
| `openai` | OpenRouter API client |
| `sentence-transformers` | Embedding baseline |
| `pandas` | Result DataFrames |
