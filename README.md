# Extractive QA as a Service

Fine-tuned Transformer encoder for extractive question answering (SQuAD 2.0 style),
served behind a token-protected API (`/health`, `/info`, `/predict`) and exposed through a public tunnel.

The API contract is defined in `qa_api_contract.yaml`. The public sample set is `public_sample.json`.
The written report is `REPORT.md`.

## Structure

| Folder | Owner | Contents |
|---|---|---|
| `training/` | Montaño | Data preparation, fine-tuning script, loss curves |
| `service/` | Mónica | FastAPI app implementing the contract, auth, long-context windows |
| `evaluation/` | Mauricio | EM/F1 evaluation, no-answer threshold, public sample checks |
| `deploy/` | Rafa | Tunnel setup, run scripts, latency measurements |
| `tests/` | Mauricio | Contract tests (the five curl checks as code) |
| `logs/` | Montaño | Training logs and loss curves (committed) |

Every member reviews the code of the other folders. Everyone must be able to explain all of it at Q&A.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The access token is read from the environment variable `QA_API_TOKEN`. Copy `.env.example` only as a reference.
Never commit the real token or the model weights.

## Training

Fine-tunes `distilbert-base-uncased` on SQuAD 2.0. Needs a GPU for a reasonable time (CPU works, slowly).
Install a CUDA build of PyTorch first if `pip install -r requirements.txt` gave you the CPU one, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu128`.

```bash
# quick smoke test (a few minutes)
python training/train.py --run_name smoke --train_samples 2000 --eval_samples 500 --epochs 1
# full run (all of SQuAD 2.0 train, validated on the full SQuAD 2.0 dev set)
python training/train.py --run_name distilbert-full
```

- Weights and `metadata.json` (model id, params, `max_seq_length`, `doc_stride` for `/info`) go to `models/<run_name>/` (not committed).
- Logs go to `logs/<run_name>/` (committed): `train.log`, `config.json`, `log_history.json`, `loss.png`,
  `metrics.json` (EM/F1 with no-answer threshold 0) and `val_score_diffs.json`
  (best span score minus null score per validation id, input for tuning the threshold in `evaluation/`).

## Run (local)

```bash
export QA_API_TOKEN=...            # never write the value in a file tracked by git
uvicorn service.app:app --host 0.0.0.0 --port 8000
```
