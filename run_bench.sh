#!/usr/bin/env bash
# Run bench_matrix.py on worker-5 against the live server.
set -uo pipefail
B=/mnt/data/users/minhth21/breeze-tts
PY=/mnt/data/users/minhth21/conda_env/breeze312/bin/python
OUT=$B/logs/bench
mkdir -p "$OUT"
REF_TEXT="Xin chào các bạn, hôm nay tôi sẽ đọc một đoạn văn ngắn để làm giọng mẫu, mong các bạn lắng nghe và góp ý giúp tôi nhé."
# Reuse the existing reference so results stay comparable with logs/bench/matrix_before.json / matrix_after.json.
if [ ! -s "$OUT/ref.wav" ]; then
  curl -s --max-time 120 -o "$OUT/ref.wav" http://127.0.0.1:8000/v1/audio/speech -H 'Content-Type: application/json' \
    -d "{\"input\":\"$REF_TEXT\",\"response_format\":\"wav\",\"seed\":7}"
fi
"$PY" -c 'import soundfile as sf,sys; print("ref.wav:", round(sf.info(sys.argv[1]).duration,2), "s")' "$OUT/ref.wav"
ext_before=$(grep -c 'POST /v1/audio/speech' "$B/logs/serve.log" | tr -d ' ')
loc_before=$(grep -c '127.0.0.1:[0-9]* - "POST /v1/audio/speech' "$B/logs/serve.log" | tr -d ' ')
"$PY" "$B/bench_matrix.py" --ref-audio "$OUT/ref.wav" --ref-text "$REF_TEXT" --out "$OUT/matrix_run.json" "$@"
ext_after=$(grep -c 'POST /v1/audio/speech' "$B/logs/serve.log" | tr -d ' ')
loc_after=$(grep -c '127.0.0.1:[0-9]* - "POST /v1/audio/speech' "$B/logs/serve.log" | tr -d ' ')
echo "speech POSTs from other clients during run: $(( (ext_after - ext_before) - (loc_after - loc_before) ))"
