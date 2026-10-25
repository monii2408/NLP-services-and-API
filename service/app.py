"""FastAPI service implementing qa_api_contract.yaml.

Owner: Mónica. TODO:
- Load the fine-tuned model once at startup; /health returns 503 {"status": "loading"} until then.
- GET /health (public) -> 200 {"status": "ok"}.
- GET /info (protected) -> team, model, params, max_seq_length, doc_stride.
- POST /predict (protected) -> id, answer, start, score. Split long contexts into windows (doc_stride).
- Token: read from QA_API_TOKEN; return 401 {"error": "unauthorized"} on missing or wrong token.
- Errors use {"error": <code>, "detail": <optional>} with codes unauthorized, bad_request, internal_error.
"""

import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

logger = logging.getLogger("qa_service")

MODEL_DIR = Path(os.environ.get("QA_MODEL_DIR", "models/distilbert-full"))
BASE_MODEL = "distilbert-base-uncased"
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
MAX_ANSWER_TOKENS = 30
# Predict "no answer" when null_logit - best_span_logit exceeds this value.
# 2.0 was the best value of a sweep over the SQuAD 2.0 validation set, weighting unanswerable
# questions at 25%; Mauricio owns the calibration (evaluation/). QA_NULL_THRESHOLD overrides it.
NULL_THRESHOLD = float(os.environ.get("QA_NULL_THRESHOLD", "2.0"))

_tokenizer = None
_model = None
_info: dict | None = None


def _load_model() -> None:
    global _tokenizer, _model, _info
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR)
        model.eval()
        _info = {
            "team": os.environ["QA_TEAM"],
            "model": BASE_MODEL,
            "params": sum(p.numel() for p in model.parameters()),
            "max_seq_length": MAX_SEQ_LENGTH,
            "doc_stride": DOC_STRIDE,
        }
        # Assign the model last, so /health only turns "ok" once everything is set.
        _tokenizer, _model = tokenizer, model
        logger.info("model loaded from %s", MODEL_DIR)
    except Exception:
        logger.exception("model failed to load from %s", MODEL_DIR)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_load_model, daemon=True).start()
    yield


app = FastAPI(title="Extractive QA Service", lifespan=lifespan)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    context: str = Field(min_length=1, max_length=5000)
    question: str = Field(min_length=1, max_length=500)


def require_token(request: Request) -> None:
    expected = os.environ.get("QA_API_TOKEN", "")
    header = request.headers.get("authorization", "")
    # An unset token fails closed: nothing is authorized.
    if not expected or not secrets.compare_digest(header.encode(), f"Bearer {expected}".encode()):
        raise HTTPException(
            status_code=401,
            detail={"error": "unauthorized"},
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.exception_handler(StarletteHTTPException)
async def http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    content = exc.detail if isinstance(exc.detail, dict) else {"error": "internal_error", "detail": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first["loc"][1:])
    return JSONResponse(
        status_code=400,
        content={"error": "bad_request", "detail": f"field '{location}': {first['msg']}"},
    )


@app.exception_handler(Exception)
async def internal_error(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error", exc_info=exc)
    return JSONResponse(status_code=500, content={"error": "internal_error"})


def _scan(context: str, question: str) -> tuple[str | None, int | None, float, float, float, float]:
    """Best span and no-answer evidence over all windows of one context.

    Returns (span_text, start, span_logit, span_prob, null_logit, null_prob).
    span_text is None when no span was found.
    """
    encoding = _tokenizer(
        question.lstrip(),
        context,
        truncation="only_second",
        max_length=MAX_SEQ_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    best_logit = float("-inf")
    best_span = None
    best_prob = 0.0
    null_logit = float("inf")
    null_prob = 0.0

    for window in range(len(encoding["input_ids"])):
        inputs = {
            key: torch.tensor([encoding[key][window]])
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if key in encoding
        }
        with torch.no_grad():
            output = _model(**inputs)

        start_logits = output.start_logits[0]
        end_logits = output.end_logits[0]
        start_probs = torch.softmax(start_logits, dim=0).tolist()
        end_probs = torch.softmax(end_logits, dim=0).tolist()
        starts = start_logits.tolist()
        ends = end_logits.tolist()

        # Index 0 is [CLS]; its logits are the no-answer evidence for this window.
        window_null = starts[0] + ends[0]
        if window_null < null_logit:
            null_logit = window_null
            null_prob = start_probs[0] * end_probs[0]

        sequence_ids = encoding.sequence_ids(window)
        context_tokens = [i for i, seq in enumerate(sequence_ids) if seq == 1]
        offsets = encoding["offset_mapping"][window]

        for s in context_tokens:
            for e in range(s, min(s + MAX_ANSWER_TOKENS, len(sequence_ids))):
                if sequence_ids[e] != 1:
                    break
                logit = starts[s] + ends[e]
                if logit > best_logit:
                    best_logit = logit
                    best_span = (offsets[s][0], offsets[e][1])
                    best_prob = start_probs[s] * end_probs[e]

    if best_span is None:
        return None, None, best_logit, best_prob, null_logit, null_prob

    char_start, char_end = best_span
    return context[char_start:char_end], char_start, best_logit, best_prob, null_logit, null_prob


def _best_answer(context: str, question: str) -> tuple[str | None, int | None, float]:
    text, start, best_logit, best_prob, null_logit, null_prob = _scan(context, question)
    if text is None or null_logit - best_logit > NULL_THRESHOLD:
        return None, None, null_prob
    return text, start, best_prob


@app.get("/health")
def health():
    if _model is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok"}


@app.get("/info", dependencies=[Depends(require_token)])
def info():
    if _info is None:
        raise HTTPException(status_code=500, detail={"error": "internal_error", "detail": "model not loaded"})
    return _info


@app.post("/predict", dependencies=[Depends(require_token)])
def predict(body: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=500, detail={"error": "internal_error", "detail": "model not loaded"})
    answer, start, score = _best_answer(body.context, body.question)
    return {"id": body.id, "answer": answer, "start": start, "score": round(min(max(score, 0.0), 1.0), 6)}
