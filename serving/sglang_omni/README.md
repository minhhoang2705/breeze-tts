# Breeze TTS 2 on SGLang-Omni

## Setup

By default, the scripts create and use an isolated Python 3.12 `.venv`:

```bash
./bootstrap.sh
./serve.sh
```

To install and run from an activated Conda environment instead (recommended on
H100 nodes already managed with Conda), create/activate a dedicated serving
environment first:

```bash
conda create -n breeze-serving python=3.12 -y  # omit if it already exists
conda activate breeze-serving
./bootstrap.sh
./serve.sh
```

The scripts detect the active Conda environment and install into/use its Python.
You can explicitly select an interpreter with `PYTHON=/path/to/python` (or
`PYTHON=python` after activation). This avoids hard-coding Python 3.12
`site-packages` paths and keeps the repository-local `.venv` as the default
when no Conda environment is active. Use a dedicated serving environment rather
than mixing SGLang-Omni dependencies into a training environment.

The script does not download model weights. By default the server uses the local
checkpoint at `../../../breeze-tts-2`; download or place the checkpoint there
before serving, or set `MODEL=/absolute/path/to/checkpoint`.

The bootstrap installs `qwen-tts==0.1.1` with `--no-deps` so its pinned
Transformers version cannot replace the SGLang-Omni runtime. The tokenizer is
available as `qwen_tts.Qwen3TTSTokenizer` from the environment's
`site-packages/qwen_tts/`; import it after applying Omni's Qwen Transformers
compatibility patch.

A `site-packages/breeze_omni_register.pth` hook imports the out-of-tree model
registration when `BREEZE_OMNI_REGISTER=1`. `serve.sh` sets that flag and adds
this directory to `PYTHONPATH`, so the registration runs in both the CLI and
spawn-created Python processes.

## Run

```bash
./serve.sh            # http://0.0.0.0:8000, ready in ~25 s
```

Defaults (1x RTX 4070 Ti, 12 GB): Triton attention, `MEM_FRACTION=0.56`,
`MAX_RUNNING=12`, `CUDA_GRAPH_MAX_BS=$MAX_RUNNING`. Override them, or `PORT`,
`HOST`, `ATTENTION`, `MODEL`, through the environment; extra `sgl-omni serve`
arguments are passed through. `MAX_RUNNING` counts SGLang *lanes*: a CFG
request uses two. Keep `MAX_RUNNING * 2048` within the KV pool printed at start
(`#tokens: ...`), so SGLang never retracts a request whose audio has already
been streamed. `serve.sh` refuses to start if `PORT` is taken, because sgl-omni
would otherwise move to a random port silently.

## API

`POST /v1/audio/speech` (OpenAI-compatible). Output is mono 24 kHz.

| Field | Meaning |
|---|---|
| `input` | Text to speak. |
| `instructions` | Voice design or voice direction (Breeze `instruction`). |
| `ref_audio`, `ref_text` | Voice cloning. Both must be given: an audio data URI plus its exact transcript. |
| `stage_params.tts_engine.cfg_scale` | `1` (default) = no CFG. `0` = use the negative prompt only. Any other value = CFG; the template must have an instruction. Same rules as `infer.py`, and README suggests `4`. |
| `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_new_tokens`, `seed` | Backbone (codebook 0) sampling. Defaults are 0.9 / 50 / 1.0 / 1.1 / 1500. |
| `stage_params.tts_engine.depth_temperature`, `depth_top_k`, `depth_top_p` | Depth-decoder sampling (codebooks 1-15). Defaults are 0.9 / 50 / 1.0. |
| `stream: true` + `response_format: "pcm"` | Streaming 16-bit PCM. The first chunk covers 4 frames (320 ms), later chunks 12 frames. |

```bash
curl http://127.0.0.1:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"Hello there.","response_format":"wav"}' --output speech.wav

curl -N http://127.0.0.1:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"Speak slowly.","instructions":"A calm, warm voice.",
       "stage_params":{"tts_engine":{"cfg_scale":4}},"stream":true,"response_format":"pcm"}' \
  | ffplay -nodisp -autoexit -f s16le -ar 24000 -ac 1 -
```

Security note: sgl-omni passes a bare local filesystem path in `ref_audio`
straight through to the model stage, which reads it; the same holds for the
upstream Higgs and Qwen3-TTS integrations. Prefer data URIs, and do not expose
the port to untrusted clients (or set `HOST=127.0.0.1`).

## How it maps onto SGLang-Omni

Pipeline: `preprocessing` (prompt templates, reference encoding with the bundled
12 Hz codec) -> `tts_engine` (SGLang) -> `vocoder` (incremental 12 Hz decode).

- **Backbone**: SGLang `Qwen3ForCausalLM` layers with paged KV. Prefill embeddings
  come from the T5Gemma2 text encoder plus codec-frame embeddings
  (`components/text_encoder.py`). Each decode step feeds back the summed
  embedding of the previous 16-code frame. The frame table is tied to
  `depth_decoder.model.embed_tokens` (`tie_codebooks_embeddings`).
- **Depth decoder** (`components/depth_decoder.py`): codebooks 1-15 for every
  live frame in one CUDA graph over `MAX_RUNNING` fixed lanes, captured at
  startup. Per-shape graphs cost ~65 MB each, the single graph ~0.2 GB.
- **CFG**: paired SGLang requests `<rid>` / `<rid>_uncond` (negative prompt),
  the same pattern as sglang-omni `minimax_music3`. `BreezeScheduler` queues and
  admits each pair atomically and aborts both together. `BreezeTTSModelRunner`
  guides the code-0 logits as `uncond + s*(cond - uncond)` before the repetition
  penalty and the suppression of reserved ids. The depth decoder guides every
  step. The uncond lane takes the cond lane's frame as its next input.

## Measured (base checkpoint, RTX 4070 Ti)

`bench.py` cycles plain EN, plain VI, voice design + CFG 4, clone, and clone +
instruction + CFG 4, streaming PCM, 24 requests per row, `MAX_RUNNING=12`:

| concurrency | req/s | audio s / wall s | RTF mean | TTFA p50 (s) | TTFA p95 (s) |
|---|---|---|---|---|---|
| 1 | 0.31 | 1.73 | 0.58 | 0.24 | 0.39 |
| 2 | 0.43 | 3.09 | 0.61 | 0.38 | 0.56 |
| 4 | 0.94 | 5.44 | 0.65 | 0.41 | 0.54 |
| 8 | 1.72 | 9.90 | 0.72 | 0.47 | 0.69 |
| 12 | 1.82 | 10.03 | 0.95 | 1.30 | 3.57 |

With 12 concurrent requests the lanes are full (CFG requests take two each), so
requests start queueing. `bench.py` writes `c<N>/results.json` per row, and
`tools/plot_bench.py` needs matplotlib from the repository-root inference
environment, not the serving environment. From this directory, regenerate the
chart with:
`../../.venv/bin/python tools/plot_bench.py ../../outputs/sglang_omni/bench docs/assets/bench.svg` writes [`docs/assets/bench.svg`](docs/assets/bench.svg).

For the full story (architecture diagrams, CFG pairing, the memory budget, the
bugs found during the port and the verification), see
[`docs/PORTING.md`](docs/PORTING.md).

Quality parity: the same 30 requests (10 cases x 3 seeds) were generated with
`infer.py` and with this server, then scored with Whisper large-v3 (WER against
the input text): 0.236 vs 0.233 overall. Per kind the two match within noise:
EN plain 0.02/0.01, design+CFG 0/0, clone 0.01/0.02, edit+CFG 0.01/0.02. Vietnamese
is ~1.1 for **both**, because the base checkpoint does not speak Vietnamese;
serve a Vietnamese fine-tune for that.

## Tests

```bash
PYTHON_BIN="${PYTHON:-${CONDA_PREFIX:-$PWD/.venv}/bin/python}"
PYTHONPATH=. flock /tmp/breeze_gpu.lock "$PYTHON_BIN" -m pytest -q tests/
```

The component tests compare against `goldens/` (generate them with
`tools/make_goldens.py` in the repository-root Breeze inference environment; see
`goldens/README.md`): exact prompt ids, prefill embeddings at cosine > 0.999,
teacher-forced depth codes, and graph == eager. `test_integration_cfg.py` covers
CFG pairing and guidance order, and `test_scheduler_pairs.py` covers pair admission and deferral.
`test_live_streaming.py` runs only when a server is up (`./serve.sh` first). For
plain, design+CFG, clone and edit+CFG, it sends the same greedy request once as
streamed PCM and once as a buffered WAV. It then requires an identical sample
count and correlation > 0.99, which covers chunking, the final flush and the
incremental codec.
