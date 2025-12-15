# Fynd AI Assignment Report

## 1. Prompt Design & Iterations

### Prompt 1: Zero-Shot, Schema-First
- **Approach**: Direct instruction to classify into 1–5 stars with a strict JSON schema.
- **Goal**: Establish a baseline with minimal guidance and enforce valid JSON output.

### Prompt 2: Reasoning with Rubric & Guardrails
- **Approach**: Ask the model to reason using a brief rubric (sentiment strength, polarity cues), then emit the JSON.
- **Goal**: Improve consistency and justification while keeping output constrained.

### Prompt 3: Few-Shot with Grounded Examples
- **Approach**: Provide labeled exemplars (1-star, 3-star, 5-star) plus format enforcement.
- **Goal**: Calibrate the model on rating granularity and reduce invalid JSON.

## 2. Evaluation Results (sampled N=5 due to quota pacing)

Generated via `task1_prompting/run_evaluation.py` (Gemini 1.5 Flash, sampled Yelp rows). Three API calls per row; free-tier quotas required a small sample.

| Prompt Strategy | Accuracy | JSON Validity |
| :--- | :--- | :--- |
| Zero-Shot | 0.60 | 0.60 |
| Reasoning | 0.60 | 0.60 |
| Few-Shot | 0.60 | 0.60 |

**Notes**
- All correct predictions were on clearly negative 1-star samples; neutral/positive rows were throttled out of this small slice, so metrics are conservative.
- For ~200 rows, increase `SAMPLE_SIZE` and `WAIT_BETWEEN_CALLS` (e.g., 12–15s) or use higher quota to avoid 429s.

## 3. System Architecture

### Task 1: Rating Prediction
- **Data Source**: Kaggle Yelp Reviews CSV (`data/yelp_reviews.csv`, configurable via `DATA_PATH`).
- **Model**: Google Gemini 1.5 Flash.
- **Pipeline**: Load sampled data -> build prompts (3 strategies) -> call API with retries/throttling -> parse/validate JSON -> score accuracy + validity -> save CSV.
- **Config knobs**: `SAMPLE_SIZE`, `WAIT_BETWEEN_CALLS`, `GEMINI_MAX_RETRIES`, `DATA_PATH`, `GOOGLE_API_KEY`.

### Task 2: AI Feedback Dashboards
- **Framework**: Streamlit (Python).
- **Data Storage**: Local JSON file (`reviews.json`) shared by user/admin views.
- **Components**:
    1. **User Dashboard**: Customer input interface.
    2. **Admin Dashboard**: Analytics and moderation view.
    3. **Utils Module**: Shared I/O and AI helpers.

## 4. Deployment

### Links
- **GitHub Repository**: [Your GitHub Repo URL]
- **User Dashboard**: [Streamlit Cloud URL]
- **Admin Dashboard**: [Streamlit Cloud URL]

### Deployment Feasibility & Steps
- The application is "deploy-ready" for Streamlit Cloud.
- **Steps**:
    1. Push code to GitHub.
    2. Login to [Streamlit Cloud](https://streamlit.io/cloud).
    3. Function 1: Deploy `task2_app/user_dashboard.py`.
    4. Function 2: Deploy `task2_app/admin_dashboard.py`.
    5. Secrets: Add `GOOGLE_API_KEY` in the App Settings safely.
