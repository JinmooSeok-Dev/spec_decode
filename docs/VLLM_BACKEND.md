# vLLM Backend — Setup & Usage

Phase 2 swaps the HuggingFace verifier for one backed by [vLLM](https://github.com/vllm-project/vllm).
Target execution inherits PagedAttention, continuous batching, and multi-GPU
support from the vLLM runtime; the verifier only owns the rejection logic.

This document walks through the full setup and three ways to exercise the
vLLM backend. For the algorithmic background see [VERIFICATION](./VERIFICATION.md)
and [ROADMAP](./ROADMAP.md).

## Prerequisites

- A CUDA GPU whose compute capability is supported by the installed PyTorch
  (e.g. sm_80 / sm_86 / sm_89). Hopper / Blackwell (sm_90+) may require
  a matching PyTorch nightly.
- Python 3.10 or newer.
- Enough VRAM to hold the target model plus some KV cache headroom
  (`gpt2` ≈ 500 MB, `Llama-3.2-3B-Instruct` ≈ 6 GB in fp16).

## 1. Install

### Option A — new virtualenv (recommended)

```bash
cd /path/to/spec_decode
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[vllm,dev]"
```

This pulls the latest vLLM wheel from PyPI along with the project's dev
tooling. Depending on your network and CUDA version, the vLLM wheel itself
can be several hundred megabytes.

### Option B — reuse an existing environment that already has vLLM

If a colleague handed you a venv with vLLM preinstalled, just add `distspec`
to it as an editable install:

```bash
source /path/to/existing/venv/bin/activate
pip install -e ".[dev]"
```

`distspec-server` and the `distspec` import path will then be available in
that environment.

## 2. Select the right GPU

If the host has multiple GPUs and not all of them are supported by your
PyTorch build, pin the one you want to use **before** you run anything:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0        # or the index of the compatible GPU
```

Skipping this on a mixed-GPU host can surface as
`CUDA error: no kernel image is available for execution on the device`.

## 3. Three ways to run

Pick whichever matches what you want to verify. Smallest-blast-radius first.

### 3.1 One-shot Python script

The fastest smoke test — it loads a target model, feeds it a draft that
matches its own greedy continuation, and checks that the verifier accepts
all of them.

```python
# try_vllm.py
from distspec.common import SamplingParams
from distspec.server import VllmVerifier
from vllm import SamplingParams as VllmSP

v = VllmVerifier(
    model_name="gpt2",             # swap for your target model
    gpu_memory_utilization=0.5,
    max_model_len=512,
    enforce_eager=True,            # skip CUDA-graph build for a faster start
    dtype="float16",
)

prompt = v.tokenizer.encode("Hello world, today")
print("prompt token ids:", prompt)

# Take the target's own greedy continuation as the draft — every token
# should be accepted.
gen = v.llm.generate(
    prompt_token_ids=[prompt],
    sampling_params=VllmSP(temperature=0, max_tokens=3, detokenize=False),
    use_tqdm=False,
)
target_tokens = list(gen[0].outputs[0].token_ids)
print("target greedy:", target_tokens)

out = v.verify(
    draft_tokens=target_tokens,
    draft_probs=None,
    context_tokens=prompt,
    sampling_params=SamplingParams(temperature=0.0),
)
print("accepted :", out.accepted_tokens)
print("bonus    :", out.bonus_token)
```

```bash
python try_vllm.py
```

Expected: `accepted == target_tokens`, and `bonus` is the next greedy token.

### 3.2 Target server + streaming client

Production-shape: one process serves draft-verify requests, another drives
generation.

```bash
# Terminal 1 — target server
distspec-server \
    --backend vllm \
    --model gpt2 \
    --listen-address 127.0.0.1:8000 \
    --gpu-memory-utilization 0.5
```

```bash
# Terminal 2 — client
python examples/client_server.py \
    --prompt "The quick brown fox jumps" \
    --server 127.0.0.1:8000 \
    --max-tokens 20
```

The client prints generated tokens as they stream in; the server terminal
shows one verify request per step plus acceptance-rate / latency metrics.

### 3.3 Pytest suite

```bash
# CI-safe subset (no GPU, no model downloads) — finishes in seconds:
pytest -m "not slow and not vllm"

# vLLM-backed tests only (GPU + a compatible vLLM build required):
pytest -m vllm -v

# Everything, including the HuggingFace slow suite:
pytest
```

The `vllm` marker is defined in `pyproject.toml` and keeps vLLM tests out
of the default CI job.

## 4. Tuning knobs worth knowing

| Option | Default | When to tune |
|---|---|---|
| `--backend vllm` | `hf` | Must be set to select the vLLM verifier. |
| `--tensor-parallel-size N` | `1` | Shard the target across N GPUs. Requires N compatible cards and matching `CUDA_VISIBLE_DEVICES`. |
| `--gpu-memory-utilization X` | `0.9` | Raise to give vLLM more KV cache headroom; lower if other processes share the GPU. |
| `--max-model-len N` / `max_model_len` | `None` (use model's own) | Cap the maximum sequence length. Leave unset to inherit the model's `max_position_embeddings`. |
| `enforce_eager=True` | `False` | Skip CUDA-graph build at startup — useful for short runs and development. |
| `dtype="float16"` / `"bfloat16"` | `"auto"` | Force a dtype if `auto` picks something your GPU doesn't support well. |

## 5. Current limitations (Phase 2 S1)

- **Greedy only.** `SamplingParams(temperature > 0)` raises `NotImplementedError`.
  Random-mode rejection against the target's full distribution is tracked
  as Phase 2 S2 in [ROADMAP](./ROADMAP.md).
- **Draft stays on the client.** The vLLM backend verifies drafts produced by
  `distspec.client` proposers; vLLM's in-engine speculative decoding is not
  wired to this pipeline.
- **One verifier per process.** `ServerConfig.backend="vllm"` currently
  constructs a single `LLMEngine`; horizontal scaling is tracked as S6.

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA error: no kernel image is available` | PyTorch doesn't support your GPU's compute capability | Export `CUDA_VISIBLE_DEVICES` to a supported GPU, or upgrade PyTorch / use a nightly matching your GPU. |
| `Error in memory profiling` at startup | Older vLLM versions mis-profile very small models on roomy GPUs | Use a slightly larger target (e.g. `gpt2` → `gpt2-medium`) or raise `--gpu-memory-utilization`. |
| `ModuleNotFoundError: vllm` | Active venv doesn't have vLLM | `pip install -e ".[vllm]"` or activate a venv that already has it. |
| `RuntimeError: VllmVerifier requires the 'vllm' package` | Same as above — raised when you try to instantiate the verifier | Install vLLM. |
| `NotImplementedError: greedy (temperature=0) only` | Client sent `temperature > 0` | Use `SamplingParams(temperature=0.0)` for now (Phase 2 S1). |
| `ValidationError: max_model_len ... is greater than the derived max_model_len` | Explicit `--max-model-len` (or `ServerConfig.max_model_len`) exceeds the target model's `max_position_embeddings` | Either drop the flag (default is `None`, letting vLLM pick the model's native limit), or pass a value ≤ the model's own. gpt2-family models cap at 1024, OPT-125M at 2048, most modern Llama-family models ≥ 4096. |
| Client hangs waiting for a reply | Draft message shape mismatch or server stalled | Check the server log; the DEALER ↔ ROUTER protocol uses `send_multipart([b"", payload])` on both sides. |

## 7. Where to look next

- `src/distspec/server/vllm_verifier.py` — the entire implementation
  (roughly 200 lines, documented inline).
- `tests/test_vllm_verifier_unit.py` — the algorithm checked against mock
  vLLM outputs; runs without vLLM installed.
- `tests/test_vllm_verifier.py` — the live-engine tests behind
  `@pytest.mark.vllm`.
- [ROADMAP](./ROADMAP.md) — remaining Phase 2 work (S2 / S5 / S6).
