# Installation

AsyncWebRL targets **Python 3.12** on Linux with NVIDIA GPUs. It does **not**
assume any particular environment — pip, uv, conda, a container, or a Slurm
allocation are all fine. What matters is ending up with the package versions
below; *how* you get there is up to your setup.

## Overview

At a high level, a working setup comes down to three things:

1. **A Python 3.12 environment** on a Linux machine with NVIDIA GPUs.
2. **The verified package versions** (see the table below), plus the AsyncWebRL
   package itself.
3. **Runtime access** — a WebGym environment token and a local copy of the base
   model checkpoint.

The browser environment itself (the Omniboxes cluster that AsyncWebRL drives
over HTTP) is a **separate service** and is not installed from this repo. Its
code lives at
[`microsoft/webgym` (`webgym` branch)](https://github.com/microsoft/webgym/tree/webgym);
set it up with the
[WebGym server quickstart](https://webgym.readthedocs.io/en/latest/server/quickstart_server.html).

The rest of this page details each one.

## Package versions

The verified versions are pinned in
[`requirements.txt`](https://github.com/BiEchi/asyncwebrl/blob/main/requirements.txt)
(verified end to end on H100 — sm90, CUDA 12.9 — with Qwen3-VL-8B-Instruct). The
correctness-critical ones — get these wrong and the model silently emits garbage
or crashes:

| Package | Version | Why it matters |
|---|---|---|
| `transformers` | `5.2.0` | 5.3+ breaks Qwen3-VL `get_rope_index`; 5.6 returns `_no_split_modules` as a `set` (FSDP wrap crash). |
| `sglang` | `0.5.12.post1` | 0.5.7 / 0.5.8 emit garbage (`)\n;\n}`) on Qwen3-VL (sm90). |
| `flash-attn` | `2.8.1` | Built from source against your torch / CUDA ABI. |
| `torch` (+ vision/audio) | `2.11.0` | The stack sglang and flash-attn are built against. |
| `flashinfer-python` | `0.6.11.post1` | Matches the sglang build. |
| `nvidia-cudnn-cu12` | `>= 9.15` | Below 9.15 silently garbles Qwen3-VL vision features. |
| `numpy` | `< 2.3` | numba constraint. |

Install this verified set together with the AsyncWebRL package itself, using
whatever tool your environment uses. The exact mechanics — picking the CUDA
wheel index, building flash-attn from source, resolving the rest of the
dependency tree — are environment-specific and left to you. The one thing to
get right is that the versions in the table above are what you actually end up
with at runtime.

## WebGym environment access

Training requires the WebGym CPU cluster for browser-session rollouts:

```bash
export CPU_CLUSTER_TOKEN='your_token_here'  # use single quotes (contains !)
```

For the cluster URL, token issuance, and access policies, see the
[WebGym server quickstart](https://webgym.readthedocs.io/en/latest/server/quickstart_server.html).

## Local base-model checkpoint

SGLang's `update_weights_from_disk` fails silently with Hugging Face Hub paths
on shared-GPU worker nodes. Pre-save the base model to a local path once:

```python
from transformers import Qwen3VLForConditionalGeneration
m = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", torch_dtype="auto",
)
m.save_pretrained("/path/to/base_model_checkpoints/Qwen3-VL-8B-Instruct")
```

Point `test_rollout.base_model_checkpoint` in your YAML to this path.
