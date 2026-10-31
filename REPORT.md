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

- **Base checkpoint** (exact Hugging Face id): `distilbert-base-uncased` (66,364,418 parameters), not previously fine-tuned on SQuAD; we fine-tuned it ourselves on SQuAD 2.0.
- **Alternatives considered** (at least one) and why you did not choose them: *(TODO team: name the alternative(s) we really discussed, e.g. `bert-base-uncased`, and the reason.)*
- **Trade-off:** DistilBERT has about 40% fewer parameters than BERT-base, so the 254 MB model loads in about 4 s and runs on CPU in fp32 without a GPU. We accept a lower F1 (67.7 on SQuAD 2.0 dev, section 2) in exchange for a service that fits a laptop.

## 2. Training

| Item | Value |
|---|---|
| Training data (full SQuAD 2.0 train or subset; how many examples) | Full SQuAD 2.0 train: 130,319 examples (43,498 unanswerable), 131,754 windows |
| Validation data (what you validated on, how many examples) | Full SQuAD 2.0 dev: 11,873 examples (5,945 unanswerable), 12,134 windows |
| Epochs | 2 (16,470 steps) |
| Learning rate and schedule (warmup?) | 3e-5, linear decay, warmup ratio 0.1, weight decay 0.01 |
| Batch size | 16 (eval 64) |
| `max_seq_length` / `doc_stride` | 384 / 128 |
| Hardware and training time | NVIDIA GeForce RTX 4070 Laptop GPU, bf16, 26.6 minutes |

- **Loss curve:** [`logs/distilbert-full/loss.png`](logs/distilbert-full/loss.png) (final validation loss 1.304; full log in `logs/distilbert-full/train.log`).
- **Validation results** (no-answer threshold 0, from `logs/distilbert-full/metrics.json`):

| Subset | Examples | EM | F1 |
|---|---|---|---|
| All | 11,873 | 63.61 | 67.74 |
| Answerable | 5,928 | 62.65 | 70.92 |
| Unanswerable | 5,945 | 64.58 | 64.58 |

## 3. No-answer threshold

- How your service decides to return `null` (which scores are compared).
- How you chose the threshold value, and the final value.
- What happens to your results if the threshold is too low or too high.

## 4. Serving architecture

- **Framework and server:** FastAPI on uvicorn, one worker process. The model is loaded once in that process; `/predict` is a synchronous function that FastAPI runs in its thread pool, on CPU in fp32.
- **Model loading:** a background thread started by FastAPI's lifespan hook loads the tokenizer and the model from `models/distilbert-full/`. The model is assigned last, so `/health` answers 503 `{"status": "loading"}` until it is ready (about 4 s on the development laptop) and 200 afterwards.
- **Request flow:** Pydantic validates the body (`id` 1-64, `context` 1-5000 and `question` 1-500 characters, no extra fields; otherwise 400 `bad_request`). The question and the context are tokenized into windows of at most 384 tokens with a 128-token overlap, so a long context becomes several windows. In every window we scan the start/end pairs of up to 30 context tokens and keep the best `start_logit + end_logit`. The tokenizer's character offsets give `start` and `answer = context[start:end]`, so the answer is always a verbatim span at the reported offset. If the no-answer evidence beats the best span by more than the threshold (section 3), the answer is `null`.
- **Access control:** a dependency on `/info` and `/predict` reads `QA_API_TOKEN` from the environment and compares the whole `Authorization` header with `Bearer <token>` using `secrets.compare_digest`. A missing, wrong or unset token gives 401 `{"error": "unauthorized"}`; `/health` stays public. The token is never in code or in git.
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
| sample-01 | `16` | `16` | 383 | Yes |
| sample-02 | `3.5 days` | `3.5 days` | 437 | Yes |
| sample-03 | `null` | `null` | `null` | Yes |
| sample-04 | `null` | `3` | 310 | **No** |
| sample-05 | `50,257` | `50,257` | 3157 | Yes |

The service gets 4 of 5 right, including `sample-05`, whose 3,171-character context needs several windows. `sample-04` fails because the question asks about BERT-base while the paragraph only gives the "3 times" GPU-memory reduction for GPT-3 175B: the model matches the pattern of the question and ignores the entity mismatch. Its no-answer evidence is only 1.06 logits below the best span, so returning `null` here would need a threshold that makes the service answer `null` too often on answerable questions (section 3).

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 66,364,418 |
| Size on disk | 254 MB (`models/distilbert-full/`) |
| Mean latency per `/predict` request on your laptop (CPU) | |
| Laptop CPU and RAM | |

## 7. Problems encountered and how you fixed them

At least two concrete problems. For each: what happened, the cause, the fix, and the commit where it was fixed.

1. **Problem:** FastAPI's default validation error does not match the contract. **Cause:** a bad body returns 422 with `{"detail": [...]}`, but the contract requires `{"error": "bad_request"}`. **Fix:** custom exception handlers return 400 `bad_request` for invalid bodies, the contract body for HTTP errors such as 401, and 500 `internal_error` for anything else. **Commit:** `fefe1d2`
2. **Problem:** with `QA_TEAM` unset, the model never became ready and `/health` stayed 503 with no visible reason. **Cause:** the loading thread builds the `/info` payload from `QA_TEAM`, and an exception inside a background thread does not stop the server (we reproduced it: `KeyError: 'QA_TEAM'` in the console, `/health` still 503). **Fix:** the loader logs "model failed to load" with the traceback, and the README documents the variable. **Commit:** `fefe1d2`, `4f8e70f`
3. **Problem:** the service could feed the model different input from the one used in training. **Cause:** `train.py` strips leading whitespace from every question and the service did not; we found it by comparing the two files. **Fix:** the service calls `question.lstrip()` before tokenizing. **Commit:** `fefe1d2`

## 8. Decoder investigation

How could a decoder (GPT-style) or an encoder–decoder architecture address specific problems **your team** encountered (Section 7 or Section 5)? Examples: answers that are not a contiguous span, yes/no questions, long contexts.

For each point: the problem, how the alternative architecture would handle it, and what it would cost (latency, model size, evaluation).

1. **Unanswerable questions with a similar-looking fact (`sample-04`).** Our encoder scores spans independently of the question's entity, so it returned `3` for BERT-base when the paragraph only describes GPT-3 175B. An encoder–decoder (e.g. T5) or a decoder (GPT-style) reads the question and the context in one generation pass and can output "unanswerable" as text, which lets it compare the entities instead of thresholding a logit difference. **Cost:** a model several times larger than our 66M-parameter DistilBERT, slower CPU generation token by token, and a risk of invented answers, so we would have to check that each answer is a substring of the context.
2. **Answers that are not one contiguous span (section 9).** Our service can only return one span of at most 30 tokens, so a yes/no answer or an answer built from two sentences is impossible. A generative model can write the answer freely. **Cost:** the contract requires `start` as a character offset into the context; a free-text answer has no reliable offset, so we would need an extra step to find the text in the context, and EM/F1 would no longer measure exactly what the model is asked to do.
3. **Long contexts (`sample-05`, 3,171 characters).** We split the context into 384-token windows with a 128-token overlap and keep the best-scoring span, so the model never sees the whole paragraph at once. A long-context decoder reads it in one pass and can use evidence from distant sentences. **Cost:** attention memory and latency grow with the context length, which is hard to afford on a CPU-only laptop, and for contexts of at most 5000 characters our windowed encoder already answers `sample-05` correctly.

## 9. Known limitations

- **Short extractive answers only.** The service returns one contiguous span of at most 30 tokens. Yes/no answers, computed numbers or answers spread over several sentences cannot be produced; the model returns the closest span or `null`.
- **Weak at spotting unanswerable questions.** `sample-04` shows the typical failure: a paragraph with the right kind of fact about a different entity gets answered. Section 3 gives the accuracy on unanswerable validation questions.
- **English only.** The base model is uncased DistilBERT; other languages give unreliable spans.
- **Oversized questions fail.** A question that alone needs more than 384 tokens is valid for the contract (we tested 500 `!` characters) but makes the tokenizer fail, and the service answers 500 `internal_error` instead of truncating the question.
- **Invalid input.** Bodies outside the contract (missing or extra field, context over 5000 characters, question over 500) get 400 `bad_request`; a missing or wrong token gets 401.
- **One request at a time.** One worker and no batching, so concurrent requests would compete for the CPU. If the model cannot load, `/health` stays 503 and the reason only appears in the console.
