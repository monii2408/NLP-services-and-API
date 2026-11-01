# Project Report — Extractive QA as a Service

## 0. Team

| Field | Value |
|---|---|
| Team id | `Los sofas` |
| Members | Diego Montano, Mónica Alvarenga, Mauricio Chávez, Rafael Santos |
| Repository URL | https://github.com/monii2408/NLP-services-and-API |
| `submission` commit hash | see the git tag submission |
| Public service URL used at the dry run | https://sweat-omnivore-manpower.ngrok-free.dev |
| AI tools used (which, and for what) | Claude Code (Anthropic): implementation of the FastAPI service, `evaluation/evaluate_model.py` and the threshold sweep, and a first draft of this report, all reviewed by the team. Claude (Anthropic, chat): deploy and tunnel setup, latency and sample-check scripts, and drafts of report sections 4, 6 and 7, reviewed by Rafael. |

## 1. Model choice and justification

- **Base checkpoint** (exact Hugging Face id): `distilbert-base-uncased` (66,364,418 parameters), not previously fine-tuned on SQuAD; we fine-tuned it ourselves on SQuAD 2.0.
- **Alternatives considered** (at least one) and why you did not choose them: `bert-base-uncased` (about 110M parameters) usually scores a few F1 points higher, but needs about 1.7 times the CPU compute per window. With CPU-only serving, a 5 s limit per request and a 150 s cap for 50 items, we chose the smaller model.
- **Trade-off:** the 254 MiB model loads in about 4 s and runs in fp32 on a laptop CPU, at 0.7 s per request through the tunnel (section 6). We accept a lower F1 (67.7 on SQuAD 2.0 dev) for a service that fits a laptop.

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

- **Rule.** The service takes the best span over all windows and the smallest `[CLS]` score (`null_logit`) over windows. It returns `null` when `null_logit - best_span_logit > threshold`; a higher threshold means more answers.
- **Choice.** `evaluation/evaluate_model.py` runs the model once on the 11,873 validation questions with the service's own scoring code and scores 321 thresholds from -8 to 8 ([`evaluation/threshold_sweep.json`](evaluation/threshold_sweep.json)). Validation is about 50% unanswerable but the hidden test set about 25%, so we select by an F1 that weights unanswerable questions by 0.25. The sweep peaks at 2.5 (weighted F1 71.29); the service uses **2.0** (`QA_NULL_THRESHOLD`), 0.25 points lower (71.03).
- **Too low or too high:** a low threshold discards correct answers (at -8, `null` for four of five answerable questions); a high one answers everything, so unanswerable questions are missed (at 4 only 39.5% are detected). `sample-04` stays wrong because its gap is -1.06; returning `null` there needs a threshold below -1.06, which costs about 8 points of answerable EM.

| Threshold | Answerable EM | Unanswerable accuracy | Weighted F1 |
|---|---|---|---|
| -8 | 19.7 | 93.8 | 39.3 |
| 0 | 64.6 | 64.5 | 69.6 |
| **2.0 (service)** | 69.9 | 51.6 | 71.0 |
| 2.5 (sweep optimum) | 70.8 | 49.0 | 71.3 |
| 4 | 72.9 | 39.5 | 70.8 |

## 4. Serving architecture

- **Framework and server:** FastAPI on uvicorn, one worker, port 8000, bound to `127.0.0.1`.
- **Model loading:** a background thread started at application startup loads the tokenizer, the model from `models/distilbert-full/` and `metadata.json`. The model is assigned last, so `/health` answers 503 `{"status":"loading"}` until ready and then 200. If loading fails, `/health` stays 503 and the error is printed to the console.
- **Request flow:** pydantic validates the body (extra fields rejected, contract length limits). Question and context are tokenized into 384-token windows with stride 128, so long contexts become several windows. The best span is the token pair with the highest start+end logit (at most 30 tokens) over all windows, and the answer is sliced from the original context with the tokenizer's offsets, so it is always a verbatim span at `start`.
- **Access control:** a dependency on `/info` and `/predict` compares the `Authorization` header with `Bearer <QA_API_TOKEN>` using `secrets.compare_digest`. The token comes only from the environment; if unset, every request is rejected with 401 `{"error":"unauthorized"}`. `/health` is public. Invalid bodies give 400 `bad_request`, other failures 500 `internal_error`.
- **Tunnel:** ngrok, `ngrok http --url=sweat-omnivore-manpower.ngrok-free.dev 127.0.0.1:8000`; backup Cloudflare Quick Tunnel, `cloudflared tunnel --url http://127.0.0.1:8000`. Both use `127.0.0.1`, not `localhost` (section 7, problem 1). Dry-run screenshots of tests 1-4 through the public URL are in `deploy/dry_run/`.
- **How to run** from a clean machine (Windows PowerShell):

```powershell
git clone https://github.com/monii2408/NLP-services-and-API
cd NLP-services-and-API
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# download distilbert-squad2-full.zip from the model-v1 release and unzip it into models/
$env:QA_API_TOKEN = "<token>"
$env:QA_TEAM = "Los sofas"
uvicorn service.app:app --host 127.0.0.1 --port 8000
# in another terminal
ngrok http 127.0.0.1:8000
```

## 5. Results on the public sample set

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|---|
| sample-01 | `16` | `16` | 383 | Yes |
| sample-02 | `3.5 days` | `3.5 days` | 437 | Yes |
| sample-03 | `null` | `null` | `null` | Yes |
| sample-04 | `null` | `3` | 310 | **No** |
| sample-05 | `50,257` | `50,257` | 3157 | Yes |

The service gets 4 of 5 right, including `sample-05`, whose 3,171-character context needs several windows. `sample-04` asks about BERT-base, but the paragraph gives the "3 times" memory reduction for GPT-3 175B: the model matches the question's pattern and ignores the entity mismatch. Its no-answer evidence is only 1.06 logits below the best span (section 3).

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 66,364,418 |
| Size on disk | 254 MiB (266 MB), `models/distilbert-full/` |
| Mean latency per `/predict` request on your laptop (CPU) | 0.22 s mean over `127.0.0.1` (p50 0.08 s, p95 0.66 s, max 0.68 s); 0.70 s through the ngrok tunnel (p50 0.58 s, p95 1.08 s, max 1.16 s). 30 requests per URL: the 5 public samples plus one synthetic 5000-character context. Estimated 35 s for 50 items through the tunnel (cap: 150 s). |
| Laptop CPU and RAM | Intel Core i7-11800H (8 cores, 16 threads), 31.7 GB RAM |

## 7. Problems encountered and how you fixed them

1. **Problem:** every request took about 2.1 s locally. **Cause:** on Windows `localhost` resolves to IPv6 first and uvicorn listens only on `127.0.0.1`, so each connection waited about 2 s. **Fix:** use `127.0.0.1` in clients and in the tunnel command; mean latency fell from 2.16 s to about 0.2 s. **Commit:** `ac636c8`
2. **Problem:** `/health` stayed 503. **Cause:** `transformers>=5.0` needs PyTorch 2.6 or newer, but `requirements.txt` allowed `torch>=2.2`, so the model could not be imported. **Fix:** require `torch>=2.6`. **Commit:** `90847de`
3. **Problem:** after the upgrade the import failed with `operator torchvision::nms does not exist`. **Cause:** an old `torchvision` not matching the new torch. **Fix:** uninstall `torchvision` and `torchaudio`, which the service does not use. **Commit:** none (environment fix)
4. **Problem:** the ngrok URL answered, but not with our service. **Cause:** `ngrok http` was started without a port and forwarded to `localhost:80`. **Fix:** start it with the explicit target `127.0.0.1:8000`. **Commit:** none (operational fix)

## 8. Decoder investigation

1. **Unanswerable questions with a similar fact (`sample-04`).** Our encoder returned `3` for BERT-base although the paragraph only describes GPT-3 175B. An encoder–decoder (T5) or decoder reads question and context together and can generate "unanswerable" by comparing the entities, instead of thresholding a logit gap. **Cost:** a model several times larger than our 66M parameters, slower token-by-token CPU generation, and invented answers, so each answer must be checked as a substring of the context.
2. **Answers that are not one span (section 9).** Our service returns one span of at most 30 tokens, so yes/no answers or answers built from two sentences are impossible. A generative model can write them freely. **Cost:** the contract needs `start` as a character offset, so free text needs an extra matching step, and EM/F1 no longer measure the same thing.
3. **Long contexts (`sample-05`, 3,171 characters).** Windows of 384 tokens mean the model never sees the whole paragraph at once; a long-context decoder would. **Cost:** attention memory and latency grow with context length, hard on a CPU-only laptop, and our windows already answer `sample-05` correctly.

## 9. Known limitations

- **Short extractive answers only:** one contiguous span of at most 30 tokens; no yes/no, computed or multi-sentence answers.
- **Weak on unanswerable questions:** `sample-04` shows it; only 51.6% of unanswerable validation questions are detected at threshold 2.0.
- **English only:** the base model is uncased DistilBERT.
- **Oversized questions fail:** a question that alone needs more than 384 tokens is valid for the contract (we tested 500 `!` characters) but the tokenizer fails and the service answers 500 `internal_error`.
- **One request at a time:** one worker and no batching. If the model cannot load, `/health` stays 503 and the reason only appears in the console.
