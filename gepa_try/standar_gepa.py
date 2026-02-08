"""
LLM-based dataset standardization for Unitxt.
Implements the GEPA (Genetic-Pareto) architecture for prompt optimization using Local Gemma-3.
"""
import os
import json
import torch
import random
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datasets import load_dataset, Dataset
from unitxt import get_from_catalog
from openai import OpenAI
from transformers import pipeline

# ============================================================================
# CONFIGURATION & GLOBAL RESOURCES
# ============================================================================

MODEL_ID = "anthropic/claude-opus-4.5"
LOCAL_MODEL_ID = "google/gemma-3-270m-it" # ou un autre modèle local performant

# Singleton pour le pipeline local
_LOCAL_PIPELINE = None

# Cache pour les prompts optimisés (évite de relancer GEPA à chaque run)
_GEPA_PROMPT_CACHE = {}

# ============================================================================
# GEPA: GROUND TRUTH DATA (GOLD SAMPLES)
# ============================================================================
# Ces exemples servent de "Fonction de Fitness" pour l'algorithme génétique.
GOLD_SAMPLES = [
    {
        "name": "Sentiment Analysis",
        "input": {"sentence": "I love this movie!", "label": 1},
        "columns": ["sentence", "label"],
        "expected": {"task": "classification", "text": "sentence", "label": "label"}
    },
    {
        "name": "NLI Task",
        "input": {"premise": "A man walks.", "hypothesis": "A person moves.", "label": 0},
        "columns": ["premise", "hypothesis", "label"],
        "expected": {"task": "nli", "text_a": "premise", "text_b": "hypothesis", "label": "label"}
    },
    {
        "name": "Translation",
        "input": {"en": "Hello", "fr": "Bonjour"},
        "columns": ["en", "fr"],
        "expected": {"task": "translation", "text": "en", "translation": "fr"}
    },
    {
        "name": "Summarization",
        "input": {"article": "Long text...", "summary": "Short text."},
        "columns": ["article", "summary"],
        "expected": {"task": "summarization", "text": "article", "summary": "summary"}
    }
]

# ============================================================================
# CLIENTS & UTILS
# ============================================================================

def _get_client():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Error: OPENROUTER_API_KEY environment variable is not set.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

def _get_local_pipeline():
    global _LOCAL_PIPELINE
    if _LOCAL_PIPELINE is None:
        print(f"⚙️ Loading local GEPA Engine: {LOCAL_MODEL_ID}...")
        _LOCAL_PIPELINE = pipeline(
            "text-generation",
            model=LOCAL_MODEL_ID,
            model_kwargs={"dtype": torch.bfloat16},
            device_map="auto",
        )
    return _LOCAL_PIPELINE

def _clean_json_output(text: str) -> dict:
    """Extract JSON reliably and flatten nested structures."""
    parsed = None
    try:
        # 1. Extraction basique
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        # Nettoyage brutal des caractères parasites
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        
        if start != -1 and end != 0:
            parsed = json.loads(text[start:end])
    except:
        return None

    if not parsed:
        return None

    # 2. APLATISSEMENT (Correction du bug que tu as eu)
    # Si le modèle répond {'task': '...', 'mapping': {'text': '...'}}
    # On remonte tout au premier niveau.
    if "mapping" in parsed and isinstance(parsed["mapping"], dict):
        parsed.update(parsed["mapping"])
        # On garde 'task' s'il est au niveau supérieur
    
    # Standardisation des clés en minuscules
    return {k.lower(): v for k, v in parsed.items()}

# ============================================================================
# GEPA ARCHITECTURE IMPLEMENTATION
# ============================================================================
TASK_ALIASES = {
    "classification": ["sentiment analysis", "sentiment", "classif", "binary classification"],
    "nli": ["entailment", "inference", "natural language inference"],
    "translation": ["translate", "traduction"],
    "summarization": ["summary", "abstract"]
}

@dataclass
class GEPAIndividual:
    """Represents a single prompt candidate in the population."""
    prompt_text: str
    score: float = 0.0
    trace: str = "" # Stores the 'Reflection' (why it failed/succeeded)

class GEPAOptimizer:
    """
    Implements the Genetic-Pareto optimization loop.
    Phases: Evaluation -> Reflection -> Mutation -> Selection
    """
    def __init__(self, pipeline, base_prompt: str, population_size: int = 3, generations: int = 2):
        self.pipeline = pipeline
        self.population = [GEPAIndividual(prompt_text=base_prompt)]
        self.population_size = population_size
        self.generations = generations
        self.gold_data = GOLD_SAMPLES

    def _evaluate(self, individual: GEPAIndividual):
        """Fitness Function: Runs the prompt against Gold Samples."""
        total_score = 0.0
        logs = []

        for sample in self.gold_data:
            # Prepare Prompt
            prompt = individual.prompt_text.replace("{sample}", json.dumps(sample["input"]))
            prompt = prompt.replace("{columns}", str(sample["columns"]))
            
            try:
                # Inference (gardez max_new_tokens=64 pour la vitesse CPU)
                out = self.pipeline(
                    [{"role": "user", "content": prompt}], 
                    max_new_tokens=64, return_full_text=False, temperature=0.1
                )
                
                # --- NOUVELLE LOGIQUE D'EXTRACTION ---
                # On gère le format liste vs string ici aussi par sécurité
                gen_text = out[0]["generated_text"]
                if isinstance(gen_text, list): gen_text = gen_text[-1]["content"]
                
                pred = _clean_json_output(gen_text)
                expected = sample["expected"]

                # --- SCORE PROGRESSIF ---
                step_score = 0.0
                
                if not pred:
                    logs.append(f"Sample '{sample['name']}': 0.00 (Invalid JSON)")
                else:
                    # NIVEAU 1 : JSON Valide (+0.1)
                    step_score += 0.1
                    
                    # NIVEAU 2 : La clé "task" existe (+0.1)
                    pred_task = pred.get("task", "").lower()
                    expected_task = expected.get("task").lower()
                    
                    if pred_task:
                        step_score += 0.1
                        
                    # NIVEAU 3 : La tâche est sémantiquement proche (+0.3)
                    # On vérifie si c'est exact OU si c'est dans les alias
                    is_exact = pred_task == expected_task
                    is_alias = pred_task in TASK_ALIASES.get(expected_task, [])
                    
                    if is_exact:
                        step_score += 0.3
                    elif is_alias:
                        step_score += 0.2 # Un peu moins de points, mais pas zéro !
                    
                    # NIVEAU 4 : Mapping des colonnes (+0.5)
                    # On vérifie si les VALEURS attendues (ex: "sentence") sont présentes dans les VALEURS prédites
                    # Peu importe la clé (ex: si modèle dit 'text':'sentence' et on veut 'inputs':'sentence', c'est bien)
                    expected_values = set(v for k, v in expected.items() if k != "task")
                    pred_values = set(v for k, v in pred.items() if k != "task" and isinstance(v, str))
                    
                    if expected_values:
                        overlap = len(expected_values & pred_values)
                        mapping_ratio = overlap / len(expected_values)
                        step_score += (mapping_ratio * 0.5)

                    logs.append(f"Sample '{sample['name']}': {step_score:.2f} (Task: {pred_task})")
                
                total_score += step_score

            except Exception as e:
                logs.append(f"Sample '{sample['name']}': Error {str(e)}")
        
        individual.score = total_score / len(self.gold_data)
        individual.trace = "\n".join(logs)

    def _reflect_and_mutate(self, parent: GEPAIndividual) -> GEPAIndividual:
        """
        The Core of GEPA:
        1. Reflect: Ask LLM to analyze the trace.
        2. Mutate: Ask LLM to rewrite the prompt based on reflection.
        """
        # --- Helper interne pour extraire le texte proprement ---
        def extract_content(pipe_output):
            # pipe_output est souvent: [{'generated_text': [{'role': 'user', ...}, {'role': 'assistant', 'content': '...'}]}]
            try:
                gen = pipe_output[0]["generated_text"]
                if isinstance(gen, list):
                    # C'est une conversation, on prend le dernier message (la réponse de l'assistant)
                    return gen[-1]["content"]
                elif isinstance(gen, str):
                    # C'est du texte brut
                    return gen
                return str(gen)
            except Exception as e:
                print(f"⚠️ Erreur extraction pipeline: {e}")
                return ""

        # 1. REFLECTION
        reflection_prompt = [
            {"role": "user", "content": f"""
            Analyze the performance trace of an AI Prompt.
            
            Current Prompt:
            "{parent.prompt_text}"
            
            Performance Trace:
            {parent.trace}
            
            Task: Identify ONE specific weakness (e.g., confused columns, wrong JSON format).
            Response (1 sentence):
            """}
        ]
        
        # On appelle le pipeline
        out_ref = self.pipeline(
            reflection_prompt, 
            max_new_tokens=100
        )
        reflection_text = extract_content(out_ref)

        # 2. MUTATION
        mutation_prompt = [
            {"role": "user", "content": f"""
            Act as an Expert Prompt Engineer. Improve the prompt below based on the critique.
            
            Original Prompt: "{parent.prompt_text}"
            Critique: "{reflection_text}"
            
            Constraint: Keep {{sample}} and {{columns}} placeholders.
            Return ONLY the rewritten prompt text.
            """}
        ]
        
        out_mut = self.pipeline(
            mutation_prompt, 
            max_new_tokens=512
        )
        mutation_text = extract_content(out_mut)
        
        # Nettoyage final
        mutation_text = mutation_text.replace('"', '').strip()
        
        return GEPAIndividual(prompt_text=mutation_text)

    def run(self) -> str:
        """Main Evolution Loop."""
        print(f"🧬 [GEPA] Initializing Population ({self.population_size} candidates)...")
        
        # Eval Gen 0
        self._evaluate(self.population[0])
        best_individual = self.population[0]
        print(f"🧬 [GEPA] Baseline Score: {best_individual.score:.2f}")

        if best_individual.score >= 0.95:
            return best_individual.prompt_text

        for gen in range(self.generations):
            print(f"🧬 [GEPA] --- Generation {gen + 1} ---")
            next_gen = []
            
            # Elitism: Keep the best
            next_gen.append(best_individual)
            
            # Create offspring through Reflection & Mutation
            while len(next_gen) < self.population_size:
                # Select parent (Tournament selection could go here, strictly taking best for now)
                parent = best_individual 
                child = self._reflect_and_mutate(parent)
                self._evaluate(child)
                next_gen.append(child)
                
                print(f"   > Child Score: {child.score:.2f}")
                if child.score > best_individual.score:
                    best_individual = child
                    print(f"   🚀 New Best found!")
            
            self.population = next_gen
            
            if best_individual.score >= 0.98:
                print("🧬 [GEPA] Converged to optimal solution.")
                break

        return best_individual.prompt_text

# ============================================================================
# MAIN INFERENCE LOGIC
# ============================================================================

def _infer_mapping_local(features: dict, sample_rows: list, instruction: str = None) -> dict:
    """
    Standardize using Local LLM + GEPA Optimization.
    """
    pipe = _get_local_pipeline()
    cols = list(features.keys())
    
    # 1. Generate a Cache Key based on dataset structure
    # Datasets with same columns usually require the same prompt strategy
    cache_key = "_".join(sorted(cols))
    
    if cache_key in _GEPA_PROMPT_CACHE:
        print("⚡ Using cached GEPA prompt.")
        optimized_prompt = _GEPA_PROMPT_CACHE[cache_key]
    else:
        # 2. Run GEPA Optimization
        print("🧬 [GEPA] No cache found. Starting optimization process...")
        
        base_template = """
        Your task is to map dataset columns to standard Unitxt fields.
        
        Input Columns: {columns}
        Sample Data: {sample}
        
        Return a JSON object with:
        - "task": The NLP task type (classification, nli, summarization, translation).
        - The mapping of columns (e.g. "text": "sentence").
        """
        
        optimizer = GEPAOptimizer(pipe, base_template, population_size=3, generations=2)
        optimized_prompt = optimizer.run()
        
        # Save to cache
        _GEPA_PROMPT_CACHE[cache_key] = optimized_prompt

    # 3. Final Inference on the ACTUAL dataset samples
    # We replace the placeholders with the real current data
    final_prompt = optimized_prompt.replace("{sample}", json.dumps(sample_rows[0]))
    final_prompt = final_prompt.replace("{columns}", str(cols))
    
    # Optional: Add user instruction if provided
    if instruction:
        final_prompt += f"\nContext: {instruction}"

    try:
        out = pipe(
            [{"role": "user", "content": final_prompt}],
            max_new_tokens=512, return_full_text=False, temperature=0.1
        )
        return _clean_json_output(out[0]["generated_text"])
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return None

# ============================================================================
# STANDARD FUNCTIONS (API FALLBACK & UTILS)
# ============================================================================

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
    # ... (Code existant identique) ...
    if not mapping: return "# Mapping failed"
    steps = []
    for field, source in mapping.items():
        if field != "task" and source != field:
            steps.append(f"Rename(field='{source}', to_field='{field}')")
    return ", ".join(steps) if steps else "Pass()"

def standardize(dataset, instruction: str = None, use_local_llm: bool = False) -> dict:
    """
    Main entry point.
    """
    if isinstance(dataset, str):
        ds = load_dataset(dataset, split="train", streaming=True)
    else:
        ds = dataset
    
    features = ds.features
    samples = list(ds.take(5))
    
    # Branching Logic
    if use_local_llm:
        mapping = _infer_mapping_local(features, samples, instruction)
    else:
        # Fallback to API logic (you need to uncomment/paste your old _infer_mapping here)
        # For now, assumes you pasted _infer_mapping_api as _infer_mapping
        pass 
        # mapping = _infer_mapping_api(features, samples, instruction) 
        # (Note: Assurez-vous d'avoir la fonction API dans le fichier si use_local_llm=False)
        mapping = {} # Placeholder si API function pas copiée

    if not mapping:
        return {"mapping": {}, "code": "Error", "score": 0.0, "dataset": ds}

    score = _score_mapping(ds, mapping)
    print(f"Mapping: {mapping} (score: {score:.2f})")
    
    return {
        "mapping": mapping,
        "code": _generate_code(mapping),
        "score": score,
        "dataset": ds,
    }

def load_standardized_dataset(dataset_name: str, config: str = None, instruction: str = None, use_local_llm: bool = False):
    if config:
        ds = load_dataset(dataset_name, config, split="train", streaming=True)
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True)
    return standardize(ds, instruction, use_local_llm=use_local_llm)