# Automated Dataset Standardization with LLM Agents

## Project Overview

In the era of ML, preparing data is a massive bottleneck.
Every dataset has a unique schema (e.g., `tweet_text` vs. `review_body`, `target` vs. `label`), making it difficult to train a single model on multiple tasks without manual engineering.

**The Goal:**
This project explores an Agentic Approach to automate data preparation. Instead of writing manual rules for every new dataset, we use a small, efficient LLM (and later famous models) to:

1. **Analyze** the raw columns of a dataset.
2. **Infer** the underlying NLP task (Classification, NLI, QA, etc.).
3. **Map** the raw columns to the standard **Unitxt** schema (e.g., `text_a`, `label`).

We compare this Agentic method against traditional baselines (Keyword Matching and Semantic Embeddings) to evaluate its robustness and accuracy.

---

## Quick Start Guide

Follow these steps to set up the environment and run the experiments.

### 1. Create a Virtual Environment

It is recommended to use a clean Python environment (Python 3.9+).

### 3. Install Dependencies

Install the required libraries.

```bash
pip install -r requirements.txt
```

## Project Structure

* `standardize.py`: Contains the **LLM Agent** logic (Gemma-2B) for schema inference.
* `baselines.py`: Contains the Keyword and Embedding baseline algorithms.
* `eval.py`: The evaluation pipeline comparing predictions vs. Unitxt Ground Truth.
* `experiments.ipynb`: The main notebook for running the campaign and visualizing results.
