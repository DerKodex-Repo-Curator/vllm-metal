# Qwen3-TTS with vLLM-Omni on MPS

The optional Omni integration serves Qwen3-TTS through vLLM-Omni's existing
HTTP API, scheduler, model runners, and streaming codec pipeline. Requests
can arrive while other requests are generating. Both stages execute with
PyTorch on the Apple GPU.

## Development dependency

This draft requires the companion vLLM-Omni extension interfaces: external
operator registration, eager predictor caching, platform deployment defaults,
and shared-memory storage adaptation. The `omni` extra pins their public parent;
advance it to the merged upstream revision before merging this draft. Installing
the parent alone does not provide these interfaces.

For local development, install the extra and then the paired Omni checkout:

```bash
VLLM_OMNI_TARGET_DEVICE=cpu uv pip install -e '.[omni]' 'torch==2.13.0'
VLLM_OMNI_TARGET_DEVICE=cpu uv pip install --no-deps -e ../vllm-omni
```

The installation target selects portable dependencies. Inference uses MPS.
Follow the [contributor setup](CONTRIBUTING.md) to build the native artifacts.

## Serve

```bash
VLLM_METAL_BACKEND=mps vllm-omni serve \
  Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --omni \
  --host 127.0.0.1 --port 8091
```

Omni's MPS deployment defaults select eager execution, up to eight requests
per stage, and a 512 MiB talker KV budget. They retain the upstream two-stage
pipeline and sampling settings. The talker produces codec frames; the second
stage converts those frames to audio chunks. An explicit Omni deployment file
can change the capacity and memory settings.

Generate a completed WAV through the upstream API:

```bash
curl http://127.0.0.1:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello, have a wonderful day.","voice":"Ryan","language":"English","response_format":"wav","seed":171}' \
  --output speech.wav
```

For audio streaming, set `"stream": true`, `"stream_format": "audio"`, and
`"response_format": "pcm"`. With `stream=true` and no `stream_format`, Omni
returns `speech.audio.delta` and `speech.audio.done` SSE events instead.
Buffered WAV and streamed PCM use the model's native 24 kHz sample rate.

The Base checkpoint accepts `task_type="Base"`, `ref_audio` (an audio data URL),
and `ref_text`. VoiceDesign accepts `task_type="VoiceDesign"` and `instructions`.
Each request can have a separate reference clip and seed. Use the matching
checkpoint for the requested task.

## Code ownership

Omni owns HTTP handling, continuous batching, request cancellation, stage
transfer, prompt construction, both neural models, and waveform decoding.
Metal supplies the MPS platform and worker adapters, macOS shared-memory
storage, predictor operator implementations, the existing MPS attention backend, and
native Torch replacements for the vLLM 0.31 operations that otherwise launch
Triton. The upstream sampler retains request state and processing order.

Predictor normalization calls the existing MPS `F.rms_norm` implementation,
and rotary operators reuse upstream cached tables through operator registration.
The predictor selects Omni's explicit `"eager"` cache mode. This integration has no
fused-RoPE dependency and does not include the early-submission experiment.
Ordinary MPS and MLX serving retain their existing request contracts.

The validated setup uses one Apple GPU, unquantized BF16 models, and the
standard Qwen3-TTS pipeline. The supported decoding controls are greedy or seeded sampling, temperature,
top-k/top-p, repetition penalties, and minimum output length. Arbitrary logit
bias, token allowlists, min-p, frequency/presence penalties, logprobs, structured
output, bad-word filtering, speculative decoding, and distributed execution
are not supported by this integration.

## Validation

Run the device and attention regressions:

```bash
pytest tests/test_mps_backend.py tests/test_mps_attention.py -q
```

The optional HTTP test launches the real Omni server and checks streams at
concurrency 1, 4, and 8 using a downloaded CustomVoice checkpoint:

```bash
QWEN3_TTS_TEST_MODEL=/path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  pytest tests/test_mps_omni_serving.py -q
```

Serving latency and throughput must be measured on this engine path. The
older standalone-runner benchmark is not a serving benchmark for this change.
