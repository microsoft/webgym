#!/bin/bash
# Canonical environment layering for AsyncWebRL / WebGym.
#
# This is the SINGLE SOURCE OF TRUTH for turning the verified SGLang base
# container (`lmsysorg/sglang:v0.5.12.post1-cu129`) into a working WebGym +
# Qwen3-VL training environment. It is invoked from the Dockerfile (baked at
# image-build time, the recommended path), and can also be run at job startup
# against an ephemeral container on your scheduler.
#
# It layers the few packages that the container does NOT ship correctly for
# this pipeline on top of the container's authoritative torch/sglang stack:
#
#   1. transformers 5.2.0  (container ships 5.6.0; 5.3+ breaks get_rope_index)
#   2. flash_attn 2.8.1    (container ships FA4 beta; transformers' FA2 path
#                           needs real FlashAttention-2)
#   3. playwright chromium (+ system libs) for the Omniboxes browser server
#   4. redis-server        for the per-node Omniboxes registry
#
# It is idempotent: re-running is safe.
#
# Environment knobs:
#   FA2_WHEEL   Path to a prebuilt flash_attn 2.8.1 wheel. If set and present,
#               it is installed (fast). Otherwise FA2 is built from source
#               (needs nvcc + the container's torch headers; ~10-30 min).
#   PY          Python interpreter to layer onto. Default: `python` (the
#               container interpreter, which must stay authoritative).
#   SKIP_BROWSER_SERVER   Set to 1 to skip playwright/redis (e.g. trainer-only
#               nodes that never host browsers).
set -euxo pipefail

PY="${PY:-python}"

# Why these exact versions: see docs/working_setup_oss.md. Do not bump without
# re-verifying coherent Qwen3-VL output on H100 (sm90) end to end.
TRANSFORMERS_VERSION="5.2.0"
FLASH_ATTN_VERSION="2.8.1"

echo "[setup_env] layering onto: $($PY -c 'import sys; print(sys.executable)')"

# ── 1. transformers 5.2.0 (override container's 5.6.0) ──────────────────────
# Highest 5.x that keeps get_rope_index's old 4-arg signature AND imports under
# sglang 0.5.12.post1. --no-deps so it does not drag the container's torch /
# tokenizers / huggingface-hub.
pip install --no-deps --quiet "transformers==${TRANSFORMERS_VERSION}"
$PY -c "import transformers; assert transformers.__version__ == '${TRANSFORMERS_VERSION}', transformers.__version__; print('[ok] transformers', transformers.__version__)"

# ── 2. flash_attn 2.8.1 (replace the container's FA4 beta) ──────────────────
# Remove the FA4 beta dist-info first so transformers resolves our FA2 build.
rm -rf /usr/local/lib/python3.12/dist-packages/flash_attn \
       /usr/local/lib/python3.12/dist-packages/flash_attn_4-* 2>/dev/null || true

if [ -n "${FA2_WHEEL:-}" ] && [ -f "${FA2_WHEEL}" ]; then
  echo "[setup_env] installing prebuilt flash_attn wheel: ${FA2_WHEEL}"
  pip install --no-deps --quiet "${FA2_WHEEL}"
else
  echo "[setup_env] no FA2_WHEEL given; building flash_attn ${FLASH_ATTN_VERSION} from source"
  # uv's PEP-517 build sandbox drops CUDA_HOME, so use pip directly with the
  # build backend allowed to see the toolkit. Needs nvcc (present in the
  # sglang container) + the container's torch headers.
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  pip install --no-build-isolation --no-cache-dir "flash-attn==${FLASH_ATTN_VERSION}"
fi
$PY -c "from flash_attn import flash_attn_func; print('[ok] flash_attn loaded:', flash_attn_func.__module__)"

# ── 3. wandb sanity (0.24.0 silently drops uploads; need >=0.25) ────────────
$PY - <<'PYEOF'
import sys
try:
    import wandb
except ImportError:
    print("[warn] wandb not importable on this interpreter; skipping check")
    sys.exit(0)
v = tuple(int(x) for x in wandb.__version__.split(".")[:2])
if v < (0, 25):
    print(f"FATAL: wandb {wandb.__version__} has the silent-upload bug; need >=0.25.0", file=sys.stderr)
    sys.exit(1)
print(f"[ok] wandb {wandb.__version__}")
PYEOF

# ── 4. Omniboxes browser server (playwright + redis) ────────────────────────
if [ "${SKIP_BROWSER_SERVER:-0}" != "1" ]; then
  $PY -m playwright install chromium --with-deps
  if ! command -v redis-server >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq redis-server
  fi
  echo "[ok] playwright chromium + redis-server ready"
else
  echo "[setup_env] SKIP_BROWSER_SERVER=1, skipping playwright/redis"
fi

echo "[setup_env] done."
