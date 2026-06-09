# AsyncWebRL / WebGym — self-contained training image.
#
# Reproduces the ONLY environment verified to produce coherent Qwen3-VL output
# on H100 (sm90) under the AReaL multi-step RL pipeline (see
# docs/working_setup_oss.md). Build on a GPU node (the flash-attn build needs
# nvcc + the base image's torch headers):
#
#   DOCKER_BUILDKIT=1 docker build -t asyncwebrl:latest .
#
# Then launch training inside it, e.g.:
#
#   docker run --gpus all --rm -it \
#     -e CPU_CLUSTER_TOKEN -e WANDB_API_KEY -e HF_TOKEN \
#     -v /path/to/tasks:/data asyncwebrl:latest \
#     python webgym/train.py --config scripts/configs/config_8x8h100_grpo_inst_nokl.yaml
#
# The base image is authoritative for the CUDA stack (python 3.12, torch
# 2.11+cu129, sglang 0.5.12.post1, CUDA toolkit 12.9). We only LAYER the few
# packages it does not ship correctly for this pipeline (transformers 5.2.0,
# flash_attn 2.8.1) plus the pure-python deps AReaL/WebGym need, installed into
# a side venv that is tail-appended to PYTHONPATH so the base CUDA stack stays
# in front.
FROM lmsysorg/sglang:v0.5.12.post1-cu129

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /AReaL

# ── System packages (browser server + build/runtime tooling) ────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git curl rsync redis-server \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip uv
ENV CUDA_HOME=/usr/local/cuda

# ── Side venv for the pure-python deps the base image does not ship ─────────
# Created against the BASE image python so the venv site-packages are ABI
# compatible and can be tail-appended to PYTHONPATH (base stays authoritative).
ENV AREAL_VENV=/opt/areal-venv
RUN uv venv --python "$(command -v python)" "$AREAL_VENV"

# Install the project's declared deps (NOT the project itself) + the webgym
# extra into the side venv. This pulls a full dependency closure including a
# torch/transformers/sglang set we immediately strip below.
COPY pyproject.toml uv.lock ./
RUN uv pip install --python "$AREAL_VENV" -r pyproject.toml --extra webgym --extra omnibox

# Strip every package that overlaps the base image's CUDA stack, so the
# container's torch/sglang/transformers/flash_attn/nvidia-* remain the only
# copy on PYTHONPATH (mixing ABIs breaks sgl_kernel / libnvJitLink / flash_attn
# loading — see docs/working_setup_oss.md "Repo .venv").
RUN uv pip uninstall --python "$AREAL_VENV" \
        torch torchvision torchaudio triton \
        flashinfer-python flashinfer-cubin \
        sglang sgl-kernel \
        transformers tokenizers huggingface_hub \
        flash_attn cuda-bindings cuda-python 2>/dev/null || true && \
    NV=$(uv pip list --python "$AREAL_VENV" 2>/dev/null | awk '/^nvidia-/ {print $1}') && \
    if [ -n "$NV" ]; then uv pip uninstall --python "$AREAL_VENV" $NV || true; fi

# Re-add the torch-adjacent helpers the base image does NOT ship, with
# --no-deps so they do not drag torch back into the venv.
RUN uv pip install --python "$AREAL_VENV" --no-deps \
        torchdata torchao "torch_memory_saver==0.0.9.post1" torchcodec || true

# ── AReaL source (editable, no deps — deps are the venv + base image) ───────
COPY . /AReaL
RUN uv pip install --python "$AREAL_VENV" --no-deps -e /AReaL

# Patch venv CLI shebangs to the env interpreter so `ray`/`wandb`/`playwright`
# run on the base image python (the venv has no python of its own on PATH).
RUN sed -i '1s|^#!.*python.*|#!/usr/bin/env python|' "$AREAL_VENV"/bin/* 2>/dev/null || true

# Base python first; venv tail-appended.
ENV PATH="$PATH:/opt/areal-venv/bin"
ENV PYTHONPATH="/AReaL:/opt/areal-venv/lib/python3.12/site-packages"

# ── Layer transformers 5.2.0 + flash_attn 2.8.1 + playwright + redis ────────
# Single source of truth shared with the Slurm path. FA2 is built from source
# here (no cached wheel inside the image).
RUN bash scripts/setup_env.sh

# ── Build-time smoke test: surface ABI/version breakage now, not at runtime ──
RUN python - <<'PY'
import torch, sglang, transformers, flash_attn
from areal.workflow.webgym_workflow import WebGymWorkflow  # noqa: F401
print("torch", torch.__version__, "| sglang", sglang.__version__,
      "| transformers", transformers.__version__, "| flash_attn", flash_attn.__version__)
assert transformers.__version__ == "5.2.0", transformers.__version__
print("[image OK] AsyncWebRL training environment ready")
PY
