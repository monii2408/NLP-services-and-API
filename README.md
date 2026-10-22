# Extractive QA as a Service

Fine-tuned Transformer encoder for extractive question answering (SQuAD 2.0 style),
served behind a token-protected API (`/health`, `/info`, `/predict`) and exposed through a public tunnel.

The API contract is defined in `qa_api_contract.yaml`. The public sample set is `public_sample.json`.
The written report is `REPORT.md`.

## Structure

| Folder | Owner | Contents |
|---|---|---|
| `training/` | Mauricio | Data preparation, fine-tuning script, loss curves |
| `service/` | Montaño | FastAPI app implementing the contract, auth, long-context windows |
| `evaluation/` | Mónica | EM/F1 evaluation, no-answer threshold, public sample checks |
| `deploy/` | Rafa | Tunnel setup, run scripts, latency measurements |
| `tests/` | Mónica | Contract tests (the five curl checks as code) |
| `logs/` | Mauricio | Training logs and loss curves (committed) |

Every member reviews the code of the other folders. Everyone must be able to explain all of it at Q&A.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The access token is read from the environment variable `QA_API_TOKEN`. Copy `.env.example` only as a reference.
Never commit the real token or the model weights.

## Run (local)

```bash
export QA_API_TOKEN=...            # never write the value in a file tracked by git
uvicorn service.app:app --host 0.0.0.0 --port 8000
```
