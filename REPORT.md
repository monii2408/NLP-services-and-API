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
| Team id | `team-XX` *(TODO: the id given in the course)* |
| Members | Diego Montaño, Mónica Alvarenga, Mauricio Chávez, Rafa *(TODO: surname)* |
| Repository URL | https://github.com/monii2408/NLP-services-and-API |
| `submission` commit hash | *(TODO: fill in after tagging `submission`)* |
| Public service URL used at the dry run | *(TODO: Rafa, after the dry run)* |
| AI tools used (which, and for what) | Claude Code (Anthropic): implementation of the FastAPI service, `evaluation/evaluate_model.py` and the threshold sweep, and a first draft of this report, all reviewed by the team. *(TODO: each member adds the tools they used.)* |

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

- **Decision rule.** For every question the service finds the best span (highest `start_logit + end_logit` inside the context) over all windows, and takes the no-answer score from the `[CLS]` token (`null_logit = start_logit[CLS] + end_logit[CLS]`, best window). It returns `null` when `null_logit - best_span_logit > threshold`; a higher threshold makes the service answer more often.
- **How we chose it.** `evaluation/evaluate_model.py` runs the model once over the full SQuAD 2.0 validation set (11,873 questions) with the same scoring code as the service, stores the gap `null_logit - best_span_logit` of each question and scores 321 thresholds from -8 to 8 in steps of 0.05 ([`evaluation/threshold_sweep.json`](evaluation/threshold_sweep.json)). Validation has about 50% unanswerable questions, but the hidden test set has about 25%, so we select by a weighted F1 that counts unanswerable questions with weight 0.25. The sweep peaks at 2.5 (weighted F1 71.29, plateau 2.45-2.8); the service runs with **2.0** (`QA_NULL_THRESHOLD`, default in `service/app.py`), whose weighted F1 is 71.03, 0.25 points lower. Threshold 0 gives 69.60.
- **Too low or too high** (validation, from the sweep):

| Threshold | Answerable EM | Unanswerable accuracy | Weighted F1 |
|---|---|---|---|
| -8 (almost always `null`) | 19.7 | 93.8 | 39.3 |
| 0 | 64.6 | 64.5 | 69.6 |
| **2.0 (service)** | 69.9 | 51.6 | 71.0 |
| 2.5 (sweep optimum) | 70.8 | 49.0 | 71.3 |
| 4 | 72.9 | 39.5 | 70.8 |

A threshold that is too low discards correct answers: at -8 the service says `null` for four of five answerable questions. A threshold that is too high answers every question, so unanswerable questions are wrong (at 4 only 39.5% are detected). This is also why `sample-04` stays wrong: its gap is -1.06, and only a threshold below -1.06 would return `null`, which costs about 8 points of answerable EM compared with threshold 0 (56.1 at -2).

## 4. Serving architecture

Framework and server: FastAPI served by uvicorn, a single worker (one process holds one copy of the model). Port 8000, bound to 127.0.0.1.
Model loading: service/app.py starts a background thread at application startup that loads the tokenizer and the model from models/distilbert-full/ (QA_MODEL_DIR overrides the path) and reads metadata.json (base model id, max_seq_length, doc_stride). The model object is assigned last, so /health answers 503 {"status":"loading"} until everything is ready and then 200 {"status":"ok"}. If loading fails, /health stays at 503 and the error is printed to the console.
Request flow: /predict validates the body with a pydantic model (id, context, question, extra fields rejected, length limits from the contract). The question and context are tokenized together into windows of 384 tokens with a stride of 128 (truncation="only_second"), so long contexts (up to 5000 characters) become several windows. Each window goes through DistilBERT on CPU. The best span is the context-token pair with the highest start+end logit (at most 30 tokens long) across all windows. The no-answer score is the minimum, over windows, of the [CLS] start+end logits. The service returns null when null_score - best_span_score > 2.0 (QA_NULL_THRESHOLD). The answer text is sliced from the original context using the tokenizer's character offsets, so context[start:start+len(answer)] == answer always holds.
Access control: a FastAPI dependency (require_token) protects /info and /predict. It compares the Authorization header with Bearer <QA_API_TOKEN> using secrets.compare_digest. The token value comes only from the environment variable QA_API_TOKEN; if it is unset, every request is rejected. Failures return 401 {"error":"unauthorized"} with WWW-Authenticate: Bearer. /health is public. Validation errors return 400 bad_request and unhandled exceptions return 500 internal_error, always in the {"error", "detail"} format.
Tunnel: ngrok (main): <exact command you run, e.g. ngrok http --url=sweat-omnivore-manpower.ngrok-free.dev 127.0.0.1:8000>. Backup: Cloudflare Quick Tunnel, cloudflared tunnel --url http://127.0.0.1:8000. Both point to 127.0.0.1, not localhost (see Section 7, problem 1).
How to run from a clean machine (Windows PowerShell):
powershell
git clone https://github.com/monii2408/NLP-services-and-API
cd NLP-services-and-API
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# download distilbert-squad2-full.zip from the model-v1 release and unzip it into models/
$env:QA_API_TOKEN = "<token>"
$env:QA_TEAM = "<team-id>"
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

The service gets 4 of 5 right, including `sample-05`, whose 3,171-character context needs several windows. `sample-04` fails because the question asks about BERT-base while the paragraph only gives the "3 times" GPU-memory reduction for GPT-3 175B: the model matches the pattern of the question and ignores the entity mismatch. Its no-answer evidence is only 1.06 logits below the best span, so returning `null` here would need a threshold that makes the service answer `null` too often on answerable questions (section 3).

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 66,364,418 |
| Size on disk | 254 MB (`models/distilbert-full/`) |
| Mean latency per `/predict` request on your laptop (CPU) | 118 ms mean, 58 ms median, 398 ms max (5 public samples × 10 requests after one warm-up, over `127.0.0.1`); about 60-100 ms for the four short samples and 305 ms for `sample-05`, which needs several windows |
| Laptop CPU and RAM | AMD Ryzen 5 8645HS (6 cores, 12 threads), 23.3 GB RAM |

## 7. Problems encountered and how you fixed them

Problem: every request took about 2.1 s locally, regardless of context length. Cause: on Windows, localhost resolved to IPv6 (::1) first; uvicorn listens only on 127.0.0.1, so each connection waited about 2 s before falling back to IPv4. Fix: use 127.0.0.1 in clients and in the tunnel command; mean latency dropped from 2.16 s to 0.16 s. Commit: <hash>
Problem: the service started but /health stayed at 503. Cause: transformers>=5.0 needs PyTorch 2.6 or newer, and requirements.txt allowed torch>=2.2 (the laptop had 2.5.1), so AutoModelForQuestionAnswering could not be imported. Fix: require torch>=2.6 and install the CPU build. Commit: <hash>
Problem: after upgrading torch, the model import failed with operator torchvision::nms does not exist. Cause: an old torchvision build, not matching the new torch, was imported by transformers. The service does not use it. Fix: uninstall torchvision and torchaudio. Commit: <hash>
Problem: the ngrok URL answered but not with our service. Cause: ngrok http was started without a port and forwarded to localhost:80. Fix: start it with the explicit target 127.0.0.1:8000 and check that the Forwarding line ends in 127.0.0.1:8000. Commit: <hash or "operational, no code change">

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
