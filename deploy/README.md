# Deploy and tunnel

Owner: Rafa.

TODO:
- Tunnel: ngrok (main) and Cloudflare Quick Tunnel or Tailscale Funnel (backup). Record the exact command used.
- Laptop: plugged in, sleep disabled, service and tunnel running for the whole session.
- Measure latency per /predict through the tunnel (5 s timeout applies to the full round trip).
- Record model size on disk, CPU and RAM for REPORT.md section 6.
- Send the public URL to the instructor before the dry run and before the presentation.

Never commit the ngrok auth token or the QA_API_TOKEN value.
