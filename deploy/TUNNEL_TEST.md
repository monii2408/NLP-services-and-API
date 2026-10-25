# Test the service through the public tunnel

Windows PowerShell commands, run from the repository root. Never commit the tokens.
The dry run must go through the tunnel, not through localhost.

## 1. Start the service

```powershell
.\.venv\Scripts\Activate.ps1
$env:QA_API_TOKEN = "<development token from the instructor>"
$env:QA_TEAM = "team-XX"          # the name registered with the instructor, no spaces
uvicorn service.app:app --host 0.0.0.0 --port 8000
```

- `QA_TEAM` is required. Without it the model fails to load and `/health` answers 503
  `{"status": "loading"}` forever. The reason is printed in this console ("model failed to load").
- The model must be in `models/distilbert-full/` (release `model-v1`, see the root README).
- Optional: `QA_NULL_THRESHOLD` (default `2.0`, chosen in REPORT.md section 3).

## 2. Check it locally (second terminal)

Use `127.0.0.1`. The first request to `localhost` from Windows PowerShell takes about 2 s because
it tries IPv6 first (measured: 2036 ms against 2 ms).

```powershell
curl.exe -i http://127.0.0.1:8000/health      # 200 {"status":"ok"} after a few seconds
```

## 3. Start the tunnel (third terminal)

Use one tool and install a second one as a backup. On the machine used for development:
ngrok and cloudflared are **not installed**; Tailscale is.

```powershell
ngrok http 127.0.0.1:8000                           # once: ngrok config add-authtoken <your ngrok token>
cloudflared tunnel --url http://127.0.0.1:8000      # URL changes at every restart
tailscale funnel 8000                               # needs MagicDNS and HTTPS in the tailnet; a new URL can take 10 min
```

Copy the public HTTPS URL and set it in the terminal used for testing:

```powershell
$env:BASE_URL = "https://<public-url>"
$env:QA_API_TOKEN = "<same token as the service>"
```

## 4. The five checks (from another network, e.g. phone hotspot)

Create `request.json` outside the repository, without a BOM (a BOM makes the JSON invalid):

```powershell
$i = (Get-Content public_sample.json -Raw -Encoding UTF8 | ConvertFrom-Json).items[0]
$json = @{ id = $i.id; context = $i.context; question = $i.question } | ConvertTo-Json -Compress
[IO.File]::WriteAllText("$env:TEMP\request.json", $json, (New-Object Text.UTF8Encoding $false))
```

```powershell
curl.exe -i "$env:BASE_URL/health"
curl.exe -i "$env:BASE_URL/info"
curl.exe -i -H "Authorization: Bearer wrong" "$env:BASE_URL/info"
curl.exe -i -H "Authorization: Bearer $env:QA_API_TOKEN" "$env:BASE_URL/info"
curl.exe -i -X POST "$env:BASE_URL/predict" -H "Authorization: Bearer $env:QA_API_TOKEN" -H "Content-Type: application/json" -d "@$env:TEMP\request.json"
```

| # | Check | Expected |
|---|---|---|
| 1 | `/health`, no token | 200 `{"status":"ok"}` |
| 2 | `/info`, no token | 401 `{"error":"unauthorized"}` |
| 3 | `/info`, wrong token | 401 |
| 4 | `/info`, valid token | 200 with `team`, `model`, `params` |
| 5 | `/predict`, valid token | 200 with `id`, `answer`, `start`, `score` (sample-01: `16`, start 383) |

## 5. Public sample and latency through the tunnel

The 5 s timeout applies to the full round trip, tunnel included.

```powershell
$items = (Get-Content public_sample.json -Raw -Encoding UTF8 | ConvertFrom-Json).items
$ms = @()
foreach ($i in $items) {
  $body = @{ id = $i.id; context = $i.context; question = $i.question } | ConvertTo-Json -Compress
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $r = Invoke-RestMethod -Uri "$env:BASE_URL/predict" -Method Post -Headers @{ Authorization = "Bearer $env:QA_API_TOKEN" } -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body))
  $sw.Stop(); $ms += $sw.ElapsedMilliseconds
  "{0}  answer={1}  start={2}  {3} ms" -f $r.id, $(if ($null -eq $r.answer) { "null" } else { $r.answer }), $(if ($null -eq $r.start) { "null" } else { $r.start }), $sw.ElapsedMilliseconds
}
"mean {0} ms, max {1} ms" -f [int]($ms | Measure-Object -Average).Average, ($ms | Measure-Object -Maximum).Maximum
```

Expected on the public sample (locally): sample-01 `16` at 383, sample-02 `3.5 days` at 437,
sample-03 `null`, sample-05 `50,257` at 3157. sample-04 is unanswerable but the model answers `3`
(a known limitation, see REPORT.md section 3). Paste the answers and the mean latency into
REPORT.md sections 5 and 6.

## 6. Before sending the URL to the instructor

- Laptop plugged in, sleep disabled; service and tunnel running for the whole session.
- Checks 1-5 pass **through the tunnel**, from another network.
- Max latency far below 5 s. If every request takes about 2 s, the tunnel tool is resolving
  `localhost` to IPv6: point it at `127.0.0.1` as in step 3.
- On presentation day the token changes: restart the service with the new `QA_API_TOKEN`.
- If the tunnel URL changes (Cloudflare restarts), send the new one.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `/health` stays 503 | `QA_TEAM` not set, or `models/distilbert-full/` missing; read the service console |
| 401 with the right token | token set in a different terminal, or extra spaces; set it in the same terminal as the service |
| 400 on `/predict` from `request.json` | the file has a BOM; recreate it with the snippet in step 4 |
| ngrok shows a warning page | only in browsers; the harness is not affected |
| Port 8000 already in use | an old service is still running; close it |
