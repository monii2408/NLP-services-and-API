# Project Report — Extractive QA as a Service

<!--
INSTRUCTIONS (delete this block in your submission)
- Copy this file to the root of your repository as REPORT.md.
- It must be committed in the commit tagged `submission`.
- Keep every heading below, in this order. Fill every section.
- A report with a missing or empty section counts as NOT delivered (final grade × 0.5).
- Length: 1–2 pages of text (about 1000 words max), not counting tables and figures.
- Language: English.
- Write about YOUR implementation. Generic text that could describe any team's project is not acceptable.
-->

## 0. Team

| Field | Value |
|---|---|
| Team id | `team-XX` |
| Members | Name 1, Name 2, Name 3, Name 4, (Name 5) |
| Repository URL | |
| `submission` commit hash | |
| Public service URL used at the dry run | |
| AI tools used (which, and for what) | |

## 1. Model choice and justification

- **Base checkpoint** (exact Hugging Face id):
- **Alternatives considered** (at least one) and why you did not choose them:
- **Trade-off:** how the choice balances quality, model size, and CPU-only serving on your laptop.

## 2. Training

| Item | Value |
|---|---|
| Training data (full SQuAD 2.0 train or subset; how many examples) | |
| Validation data (what you validated on, how many examples) | |
| Epochs | |
| Learning rate and schedule (warmup?) | |
| Batch size | |
| `max_seq_length` / `doc_stride` | |
| Hardware and training time | |

- **Loss curve:** link to the file in the repository (e.g. `logs/loss.png`).
- **Validation results:** EM and F1 on your validation data, separately for answerable and unanswerable questions if you have them.

## 3. No-answer threshold

- **Decision rule.** For each window of the context the model gives start and end logits. The no-answer score of a window is `start_logit[CLS] + end_logit[CLS]`; the service keeps the minimum over all windows as `null_logit`. The span score is `start_logit[s] + end_logit[e]` for any span of at most 30 tokens inside the context, and the best one over all windows is `span_logit`. The service returns `null` when `null_logit - span_logit > threshold`. These are raw logits; the `score` field of the response is a separate probability.
- **How the value was chosen.** We ran the fine-tuned model once over the full SQuAD 2.0 validation set (11,873 questions: 5,928 answerable, 5,945 unanswerable) and swept the threshold from -8 to +8 in steps of 0.05 (`evaluation/evaluate_model.py`, results in `evaluation/threshold_sweep.json`). The hidden test set is about 25% unanswerable, while validation is about 50%, so we maximized `0.75 * HasAns F1 + 0.25 * NoAns accuracy` instead of the plain F1. The maximum is 71.3 at +2.5, and the curve is flat between +1.5 and +3.0 (within 0.3 points). **Final value: +2.0** (`QA_NULL_THRESHOLD`), the middle of that flat region, which also stays within 0.3 points of the best at a 30% unanswerable share.

| Threshold | HasAns F1 | NoAns accuracy | F1 (all) | Weighted (25% NoAns) |
|---|---|---|---|---|
| -2.0 | 61.4 | 75.5 | 68.4 | 64.9 |
| 0.0 | 71.3 | 64.5 | 67.9 | 69.6 |
| **+2.0** | 77.5 | 51.6 | 64.5 | 71.0 |
| +5.0 | 82.0 | 33.1 | 57.5 | 69.7 |

- **Too low or too high.** A low threshold makes the service answer `null` too often: at -2.0 it loses 16 points of HasAns F1 compared with +2.0. A high threshold makes it always answer: at +8.0 NoAns accuracy falls to 18.1%, so almost every unanswerable item fails. Validation is Wikipedia text and the hidden set is written by the instructor, so the best value may shift; on the public sample, `sample-04` (unanswerable) has a gap of -1.06 and is answered at any threshold above -1.06, and lowering the threshold enough to catch it would cost about 11 points of HasAns F1.

## 4. Serving architecture

- **Framework and server:** FastAPI served by uvicorn with a single worker process. The model lives in that process, `/predict` is a synchronous function that FastAPI runs in its thread pool, and inference runs on CPU in fp32.
- **Model loading:** at startup, FastAPI's lifespan hook starts a background thread that loads the tokenizer and the model from `models/distilbert-full/` and builds the `/info` payload (the parameter count is computed from the loaded model, the team name comes from `QA_TEAM`). The model is assigned last, so `/health` answers 503 `{"status": "loading"}` until everything is ready and 200 afterwards (about 4 s on the development laptop).
- **Request flow:** the body is validated with Pydantic (`id` 1-64 characters, `context` 1-5000, `question` 1-500, no extra fields); an invalid body gives 400 `bad_request`. The question and the context are tokenized together into windows of at most 384 tokens that overlap by 128 tokens, so long contexts become several windows. Each window goes through the model. We scan every start/end pair of up to 30 context tokens and keep the highest `start_logit + end_logit` over all windows. The character offsets given by the tokenizer produce `start` and `answer = context[start:end]`, so the answer is always a verbatim span of the context at the reported offset. If the no-answer evidence beats the best span by more than the threshold (section 3), the service returns `answer: null, start: null`. Unexpected exceptions become 500 `internal_error`.
- **Access control:** a FastAPI dependency on `/info` and `/predict` reads `QA_API_TOKEN` from the environment on every request and compares the whole `Authorization` header with `Bearer <token>` using `secrets.compare_digest` (exact and case-sensitive). A missing, wrong or unset token gives 401 `{"error": "unauthorized"}`. `/health` has no dependency and stays public. The token is never written in code or committed.
- **Tunnel:** *(Rafa: tool and exact command.)*
- **How to run** from a clean machine (commands and environment variables):

```bash
python -m venv .venv && source .venv/bin/activate    # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# download distilbert-squad2-full.zip from the model-v1 release and unzip it into models/
export QA_API_TOKEN=...      # PowerShell: $env:QA_API_TOKEN = "..."
export QA_TEAM=team-XX
uvicorn service.app:app --host 0.0.0.0 --port 8000
```

## 5. Results on the public sample set

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|---|
| sample-01 | | | | |
| sample-02 | | | | |
| sample-03 | `null` | | | |
| sample-04 | `null` | | | |
| sample-05 | | | | |

Comment on any wrong answer: why do you think it failed?

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | |
| Size on disk | |
| Mean latency per `/predict` request on your laptop (CPU) | |
| Laptop CPU and RAM | |

## 7. Problems encountered and how you fixed them

At least two concrete problems. For each: what happened, the cause, the fix, and the commit where it was fixed.

1. **Problem:** … **Cause:** … **Fix:** … **Commit:** `abc1234`
2. **Problem:** … **Cause:** … **Fix:** … **Commit:** `def5678`

## 8. Decoder investigation

How could a decoder (GPT-style) or an encoder–decoder architecture address specific problems **your team** encountered (Section 7 or Section 5)? Examples: answers that are not a contiguous span, yes/no questions, long contexts.

For each point: the problem, how the alternative architecture would handle it, and what it would cost (latency, model size, evaluation).

## 9. Known limitations

What your service does **not** do or handle, and what would happen if it received such input.
