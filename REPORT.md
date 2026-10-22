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

- How your service decides to return `null` (which scores are compared).
- How you chose the threshold value, and the final value.
- What happens to your results if the threshold is too low or too high.

## 4. Serving architecture

- **Framework and server** (e.g. FastAPI + uvicorn), number of workers.
- **Model loading:** when and where the model is loaded.
- **Request flow:** from the incoming `/predict` request to the returned span, including how long contexts are handled.
- **Access control:** where the token is checked and where the token value comes from.
- **Tunnel:** which tool you use and the command you run.
- **How to run** from a clean machine (commands and environment variables):

```bash
# example
pip install -r requirements.txt
export QA_API_TOKEN=...
uvicorn app:app --host 0.0.0.0 --port 8000
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
