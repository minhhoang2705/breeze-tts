# Breeze golden traces

Generate the five reference traces from the Breeze checkpoint and this repository's inference implementation:

```bash
cd /home/minhtranh/works/SLP/TTS/research/breeze-tts
flock /tmp/breeze_gpu.lock .venv/bin/python serving/sglang_omni/tools/make_goldens.py
```

The script requires the local checkpoint at `/home/minhtranh/works/SLP/TTS/research/breeze-tts-2`, the bundled `audio_tokenizer`, and the repository's `.venv`. It uses eager attention, greedy code selection, and saves CPU tensors in `plain_vi.pt`, `plain_en.pt`, `instruction_en.pt`, `ref_clone.pt`, and `ref_edit.pt`; matching 24 kHz WAV previews are written beside them. `--checkpoint` and `--output-dir` can override the defaults. Generation includes eight frames per trace and scale-4 CFG traces for the instruction and reference-edit prompts.

The reference recording is `data/vivoice_scale3/audio/vivoice_s0_0001491.wav`; its transcript, `Rồi rồi, tới công chuyện rồi.`, is in `data/eval_holdout.jsonl`. It is encoded by the bundled tokenizer and included in the reference prompt/code tensor fields.

Backbone code-0 and depth-decoder outputs in the files are raw pre-suppression logits. Greedy selection suppresses code IDs 2048–2050, matching the fast path. The script asserts that code-0 and depth codes equal the argmax of the corresponding logits after this suppression. CFG branch logits are recorded separately; guided selection is `uncond + 4.0 * (cond - uncond)` at both backbone and depth steps. `meta` records the instantiated rotary configuration, compute dtype, checkpoint, attention mode, codec rate, and source git SHA.

For comparisons against SGLang-Omni BF16 execution, use cosine similarity greater than 0.999 for floating-point tensors and top-1 agreement for logits. This generator is the eager reference: it uses full-sequence depth forwards rather than the fast path's cached/CUDA-graph depth decoder. WAV previews decode the eight greedy frames and are intentionally short.
