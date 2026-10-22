"""Contract tests: the five curl checks from the brief, as code.

Owner: Mónica. TODO:
- /health without token -> 200 {"status": "ok"} (503 while loading).
- /info without token -> 401.
- /info with wrong token -> 401.
- /info with valid token -> 200 with team, model, params.
- /predict with valid token -> 200 with id, answer, start, score.
Run against BASE_URL (default http://localhost:8000) with QA_API_TOKEN set.
"""
