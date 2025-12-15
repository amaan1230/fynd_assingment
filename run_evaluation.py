import os
import json
import time
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core import exceptions as google_exceptions

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    # Try getting from user environment if not in .env
    pass 

if not api_key:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    # In a real scenario we might exit, but for now let's hope it's there or we handle it
    
genai.configure(api_key=api_key)

# Model check - simple fallback
model_name = "gemini-2.5-flash"
model = genai.GenerativeModel(model_name)

# Sampling and rate-limit friendly defaults (tunable via env vars)
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "50"))  # set to 200 if you have enough quota
WAIT_BETWEEN_CALLS = float(os.getenv("WAIT_BETWEEN_CALLS", "13"))  # seconds; free tier is 5 rpm
MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
DATA_PATH = os.getenv("DATA_PATH", "data/yelp_reviews.csv")  # expected Kaggle CSV path

def get_gemini_response(prompt, max_retries: int = MAX_RETRIES):
    """Call Gemini with basic retry/backoff to handle 429s."""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            # Throttle to respect free-tier rate limits (5 requests/minute)
            time.sleep(WAIT_BETWEEN_CALLS)
            return response.text
        except google_exceptions.ResourceExhausted as e:
            retry_delay = getattr(e, "retry_delay", None)
            wait_seconds = getattr(retry_delay, "seconds", None) or WAIT_BETWEEN_CALLS + attempt * 5
            print(f"Rate limit hit; waiting {wait_seconds}s before retry (attempt {attempt+1}/{max_retries}).")
            time.sleep(wait_seconds)
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            break
    return None

def extract_json(response_text):
    try:
        # Simple cleanup to handle markdown code blocks if present
        text = response_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return data
    except Exception as e:
        # print(f"JSON Decode Error: {e} | Text: {response_text}")
        return None

def normalize_prediction(json_obj):
    """Return (pred, valid_bool) enforcing integer 1-5 if possible."""
    if not isinstance(json_obj, dict):
        return None, False
    pred = json_obj.get("predicted_stars")
    if isinstance(pred, (int, float)):
        pred_int = int(pred)
        if 1 <= pred_int <= 5:
            return pred_int, True
    return None, False

def load_reviews(sample_size: int):
    """
    Prefer Kaggle CSV (DATA_PATH). Expected columns include one of:
    text/review/review_text/comment and stars/rating/label.
    Falls back to HuggingFace yelp_review_full test split if file missing.
    """
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        text_col = next((c for c in ["text", "review", "review_text", "comment"] if c in df.columns), None)
        star_col = next((c for c in ["stars", "rating", "label"] if c in df.columns), None)
        if not text_col or not star_col:
            raise ValueError(f"Dataset at {DATA_PATH} must have text and star columns (found: {df.columns}).")
        df = df[[text_col, star_col]].rename(columns={text_col: "text", star_col: "stars"})
        df["stars"] = df["stars"].astype(int)
        df = df[(df["stars"] >= 1) & (df["stars"] <= 5)]
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f"Loaded Kaggle CSV from {DATA_PATH} with {len(df)} rows.")
        return df

    print(f"Local Kaggle dataset not found at {DATA_PATH}; falling back to HuggingFace 'yelp_review_full' (test split).")
    dataset = load_dataset("yelp_review_full", split="test")
    dataset = dataset.shuffle(seed=42).select(range(sample_size))
    df = pd.DataFrame(dataset)
    df["stars"] = df["label"] + 1  # dataset labels are 0-4
    df = df[["text", "stars"]]
    print(f"Loaded {len(df)} samples from HuggingFace fallback.")
    return df

# --- Prompt Strategies ---

def prompt_strategy_1(review_text):
    """Zero-Shot, schema-first."""
    return f"""
    You are rating Yelp reviews from 1 (terrible) to 5 (excellent).
    Output ONLY JSON. Do not add prose. Follow the schema exactly.
    Review: "{review_text}"
    JSON schema:
    {{
      "predicted_stars": <1-5 integer>,
      "explanation": "<brief reason under 20 words>"
    }}
    """

def prompt_strategy_2(review_text):
    """Reasoning with rubric & guardrails."""
    return f"""
    Rate the Yelp review on a 1-5 star scale using this rubric:
    1=awful/angry, 2=poor/major issues, 3=mixed/ok, 4=good/minor issues, 5=great/enthusiastic.
    - Briefly justify the score with key evidence (no summaries).
    - Stay consistent with the rubric.
    Output ONLY JSON in the schema:
    {{
      "predicted_stars": <1-5 integer>,
      "explanation": "<reason referencing 1-2 points>"
    }}
    Review: "{review_text}"
    """

def prompt_strategy_3(review_text, examples=[]):
    """Few-shot with grounded exemplars."""
    example_text = ""
    for ex in examples:
        example_text += f'Review: "{ex["text"]}"\nLabel JSON: {{"predicted_stars": {ex["stars"]}, "explanation": "{ex["reason"]}"}}\n\n'
    return f"""
    Use these labeled examples as a guide:
    {example_text}
    Now rate the next review strictly in JSON (no prose outside JSON):
    Review: "{review_text}"
    {{
      "predicted_stars": <1-5 integer>,
      "explanation": "<reason under 20 words>"
    }}
    """

# --- Prompt registry & evaluation ---

PROMPTS = [
    {
        "id": "p1",
        "name": "Zero-shot JSON-first",
        "builder": prompt_strategy_1,
        "rationale": "Keeps instructions minimal but schema-first to test baseline validity."
    },
    {
        "id": "p2",
        "name": "Rubric reasoning",
        "builder": prompt_strategy_2,
        "rationale": "Adds explicit rubric and evidence request to improve calibration."
    },
    {
        "id": "p3",
        "name": "Few-shot grounded",
        "builder": None,  # filled later with closure to include examples
        "rationale": "Provides grounded exemplars to stabilize outputs and format."
    },
]

def build_few_shot_examples(df, k=3):
    """Sample small set for few-shot; drop them from evaluation."""
    sample = df.sample(n=min(k, len(df)), random_state=123)
    examples = []
    for _, row in sample.iterrows():
        examples.append({
            "text": row["text"][:400].replace("\n", " "),
            "stars": int(row["stars"]),
            "reason": f"Tone consistent with {row['stars']}-star review"
        })
    remaining = df.drop(sample.index)
    return examples, remaining

def compute_consistency(df, pred_cols, valid_cols):
    """Cross-prompt consistency metrics."""
    if not pred_cols:
        return {"all_agree": 0, "within_one": 0, "two_agree": 0}
    valid_mask = df[valid_cols].all(axis=1)
    if valid_mask.sum() == 0:
        return {"all_agree": 0, "within_one": 0, "two_agree": 0}
    preds = df.loc[valid_mask, pred_cols]
    spread = preds.max(axis=1) - preds.min(axis=1)
    within_one = spread.le(1).mean()
    all_agree = (preds.nunique(axis=1) == 1).mean()
    def has_two_agree(row):
        vals = row.tolist()
        return any(vals.count(v) >= 2 for v in set(vals))
    two_agree = preds.apply(has_two_agree, axis=1).mean()
    return {"all_agree": all_agree, "within_one": within_one, "two_agree": two_agree}

def main():
    print(f"Loading dataset (target {SAMPLE_SIZE} samples)...")
    df = load_reviews(SAMPLE_SIZE + 3)  # grab a few extra for few-shot
    few_shot_examples, eval_df = build_few_shot_examples(df, k=3)
    eval_df = eval_df.head(SAMPLE_SIZE)
    print(f"Using {len(eval_df)} rows for evaluation and {len(few_shot_examples)} for few-shot examples.")

    # Wire few-shot builder now that examples exist
    for prompt in PROMPTS:
        if prompt["id"] == "p3":
            prompt["builder"] = lambda text, ex=few_shot_examples: prompt_strategy_3(text, examples=ex)

    results = []
    start_time = time.time()
    print(f"Starting evaluation on {len(eval_df)} samples...")

    for i, row in enumerate(eval_df.itertuples(index=False)):
        text = row.text[:1500]
        actual_stars = int(row.stars)
        row_result = {
            "idx": i,
            "text_snippet": text[:120],
            "actual_stars": actual_stars,
        }

        for prompt in PROMPTS:
            prompt_text = prompt["builder"](text)
            resp_text = get_gemini_response(prompt_text)
            resp_json = extract_json(resp_text) if resp_text else None
            pred, valid = normalize_prediction(resp_json)
            row_result[f"{prompt['id']}_pred"] = pred
            row_result[f"{prompt['id']}_valid"] = bool(valid)

        results.append(row_result)

        if (i + 1) % 5 == 0 or (i + 1) == len(eval_df):
            print(f"Processed {i+1}/{len(eval_df)} samples...")

    df_results = pd.DataFrame(results)

    # Per-prompt metrics
    summary_rows = []
    pred_cols = []
    valid_cols = []
    for prompt in PROMPTS:
        pred_col = f"{prompt['id']}_pred"
        valid_col = f"{prompt['id']}_valid"
        pred_cols.append(pred_col)
        valid_cols.append(valid_col)

        valid_df = df_results[df_results[valid_col]]
        acc = accuracy_score(valid_df["actual_stars"], valid_df[pred_col]) if not valid_df.empty else 0
        validity = df_results[valid_col].mean() if len(df_results) else 0
        summary_rows.append({
            "prompt": prompt["name"],
            "accuracy": round(acc, 3),
            "json_validity": round(validity, 3),
            "rationale": prompt["rationale"],
        })

    consistency = compute_consistency(df_results, pred_cols, valid_cols)

    summary_df = pd.DataFrame(summary_rows)
    print("\n--- PER-PROMPT METRICS ---")
    print(summary_df)
    print("\n--- CONSISTENCY METRICS (where all prompts were valid) ---")
    print(consistency)

    # Save artifacts
    df_results.to_csv("evaluation_results.csv", index=False)
    summary_df.to_markdown("prompt_comparison.md", index=False)
    with open("prompt_definitions.md", "w", encoding="utf-8") as f:
        for prompt in PROMPTS:
            f.write(f"### {prompt['name']}\n\n")
            f.write(f"Rationale: {prompt['rationale']}\n\n")
            sample_prompt = prompt["builder"]("Example review text...")
            f.write("Prompt text:\n")
            f.write("```\n")
            f.write(sample_prompt.strip())
            f.write("\n```\n\n")
    with open("consistency_metrics.json", "w", encoding="utf-8") as f:
        json.dump(consistency, f, indent=2)

    print("\nArtifacts written: evaluation_results.csv, prompt_comparison.md, prompt_definitions.md, consistency_metrics.json")

if __name__ == "__main__":
    main()
