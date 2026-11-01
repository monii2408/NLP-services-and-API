PART 2 — deploy/README.md
Deploy and tunnel

Owner: Rafa. Never commit the ngrok authtoken or the value of QA_API_TOKEN.

Run the service (Windows PowerShell)
powershell
.venv\Scripts\Activate.ps1
$env:QA_API_TOKEN = "<token>"      # development token now, presentation-day token later
$env:QA_TEAM = "<team-id>"         # registered with the instructor, no spaces
uvicorn service.app:app --host 127.0.0.1 --port 8000

The weights are not in git: download distilbert-squad2-full.zip from the model-v1 release and unzip it into models/ so that models/distilbert-full/ exists. Wait until curl.exe -i http://127.0.0.1:8000/health returns 200.

Tunnel
Main (ngrok, once: ngrok config add-authtoken <your ngrok authtoken>): <exact command> (public URL: https://sweat-omnivore-manpower.ngrok-free.dev)
Backup (Cloudflare Quick Tunnel, URL changes on every start): cloudflared tunnel --url http://127.0.0.1:8000

Use 127.0.0.1, not localhost: on Windows localhost tries IPv6 first and adds about 2 s per request.

Measure latency
powershell
$env:QA_API_TOKEN = "<token>"
python deploy\measure_latency.py --url http://127.0.0.1:8000 --url https://<public-url> --sample public_sample.json --repeat 5

It checks the four access-control cases, validates every response against the contract, and prints mean, p50, p95 and max latency, the tunnel overhead, and an estimate for 50 items against the 150 s cap.

Before the dry run and the presentation
Real token and team id set in the terminal where uvicorn runs; restart uvicorn after changing them.
Service and tunnel running; /health returns 200 through the public URL.
Tested from another network (phone hotspot).
Public URL sent to the instructor (again if it changes).
Laptop plugged in, sleep disabled (powercfg /change standby-timeout-ac 0).