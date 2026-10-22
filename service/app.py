"""FastAPI service implementing qa_api_contract.yaml.

Owner: Montaño. TODO:
- Load the fine-tuned model once at startup; /health returns 503 {"status": "loading"} until then.
- GET /health (public) -> 200 {"status": "ok"}.
- GET /info (protected) -> team, model, params, max_seq_length, doc_stride.
- POST /predict (protected) -> id, answer, start, score. Split long contexts into windows (doc_stride).
- Token: read from QA_API_TOKEN; return 401 {"error": "unauthorized"} on missing or wrong token.
- Errors use {"error": <code>, "detail": <optional>} with codes unauthorized, bad_request, internal_error.
"""

from fastapi import FastAPI

app = FastAPI(title="Extractive QA Service")
