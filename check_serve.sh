#!/usr/bin/env bash
# Smoke-test the SGLang-Omni Breeze server on this node: log tail, /health, one TTS request.
set -uo pipefail
LOG=/mnt/data/users/minhth21/breeze-tts/logs/serve.log
OUT=/mnt/data/users/minhth21/breeze-tts/logs/test_plain.wav
PY=/mnt/data/users/minhth21/conda_env/breeze312/bin/python
echo "--- serve.log tail ---"
tail -n 20 "$LOG"
echo "--- health ---"
curl -s --max-time 5 -w " http=%{http_code}\n" http://127.0.0.1:8000/health
echo "--- speech ---"
curl -s --max-time 120 -o "$OUT" -w "http=%{http_code} bytes=%{size_download} time=%{time_total}s\n" \
  http://127.0.0.1:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"Xin chào, đây là bài kiểm tra giọng nói.","response_format":"wav"}'
"$PY" -c 'import soundfile as sf, sys; i = sf.info(sys.argv[1]); print("wav:", i.samplerate, "Hz,", round(i.duration, 2), "s")' "$OUT" 2>/dev/null \
  || { echo "not a wav, body:"; head -c 500 "$OUT"; echo; }
