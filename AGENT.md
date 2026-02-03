# Agent Context & Project Definition

@Context
**Role:** You are an expert AI Research Engineer assisting a Master 2 student with a thesis project.
**Domain:** NLP, Multi-task Learning, Data Engineering, LLM Agents.
**Core Reference:** "Datasets have different formats and need standardization" (Based on: https://aclanthology.org/2024.lrec-main.1361.pdf).

## 1. Project Overview
**Goal:** Build an automated system that uses Language Models (local inference) to "guess" the structure of any raw Hugging Face dataset and map it to a standardized format (Unitxt Card). The `load_standardized_dataset` function should be able to load the dataset from the Hugging Face Hub and apply the standardization automatically.

## 2. Technical Stack
* **Language:** Python 3.9+
* **Core Framework:** `unitxt` (IBM Research) -> Used for defining "Cards" (dataset mappings) and templates.
* **Data Handling:** `datasets` (Hugging Face), `pandas`.
* **LLM Inference:** `transformers` (Local: Gemma-3-270m-it) or API.
* **Evaluation & Tracking:** Save the result in a Dataframe format and then save it in a CSV file.
* **Baselines:** `sentence-transformers` (for semantic matching baselines).

## 3. Core Architecture
The pipeline consists of three stages:
1.  **Hypothesis Generation:** An Agent looks at `N=5` samples of a raw dataset and guesses the mapping (e.g., "The column 'tweet' is the 'text' input").
2.  **Card Generation:** The Agent generates valid `unitxt` Card code (specifically the `preprocess_steps`).
3.  **Verification:** The system runs the generated Card through `unitxt` to verify it compiles and produces sensible metrics.

## 4. Current Priority: The Evaluation Pipeline
We are adopting a "Test-First" approach. We need to build the evaluation loop and the baselines before the complex LLM agent.

**The Eval Loop Logic:**
1.  **Input:** A list of Golden datasets (e.g., SST2, WNLI, MRPC) where the Ground Truth Unitxt Card is known.
2.  **Process:**
    * Hide the Ground Truth Card.
    * Run the **Baselines** OR the **LLM Agent** to guess the mapping.
    * Generate the `eval_card` (the standardized mapping).
    * Load the dataset using the `eval_card`.
    * Compare the performance of the `eval_card` with the Ground Truth Card.
3.  **Logging (WandB):**
    Log results into a `wandb.Table` with these **exact columns**:
    * `dataset`: Name of the dataset.
    * `columns`: Column names of the raw dataset.
    * `method`: The method used ("baseline_keyword", "baseline_embedding", "llm_agent").
    * `card`: The Ground Truth Unitxt card code.
    * `eval_card`: The Generated card code.
    * `score_gap`: Performance difference (True Score - Generated Score).

## 5. Baselines for Scientific Validation
To prove the value of the LLM approach, we must implement `baseline_standardize(dataset)` functions:
1.  **Lexical Baseline (Keyword):** A simple rule-based function that maps columns based on exact string matches (e.g., if column name is "sentence", map to "text").
2.  **Semantic Baseline (Embedding):** A function using `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) to compute cosine similarity between column names and target fields (e.g., map column "review_body" to "text" because they are semantically close).

## 6. Coding Guidelines (@StyleGuide)
* **Conciseness:** Write ultra-concise, production-ready code. Avoid boilerplate.
* **No Reinventing:** Always check if there is an existing library that perform what i ask before writing custom logic.
* **Error Handling:** The pipeline must be robust. If an LLM/Baseline fails, log the error and skip to the next dataset.
* **Typing:** Use Python type hints.