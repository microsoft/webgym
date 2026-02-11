#!/bin/bash
# vLLM server management utilities
# Handles starting, stopping, and health checking the vLLM inference server

# Required global variables:
#   VLLM_PORT - Port for vLLM server
#   MODEL_TO_SERVE - Model path or HuggingFace model name
#   MODEL_TYPE - Model type (qwen-think, qwen-instruct)
#   HOST_DATA_PATH - Data directory for logs
#   MULTINODE_RANK - (optional) Rank for multinode mode

# Global state
VLLM_PID=""

# ========================
# Health Checks
# ========================

wait_for_vllm() {
    local timeout_iterations=${1:-90}  # Default 90 iterations = 3 minutes
    echo "Waiting for vLLM server to be ready on port ${VLLM_PORT} (timeout: $((timeout_iterations * 2 / 60)) min)..."
    for i in $(seq 1 ${timeout_iterations}); do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null; then
            echo "✅ vLLM server is ready at port ${VLLM_PORT}!"
            return 0
        fi
        echo "Waiting for vLLM server... (${i}/${timeout_iterations})"
        sleep 2
    done
    echo "❌ ERROR: vLLM server did not become ready in time (waited $((timeout_iterations * 2 / 60)) minutes)."
    return 1
}

is_vllm_healthy() {
    # Check if vLLM is already running and healthy on the expected port
    # Returns 0 if healthy, 1 otherwise
    if lsof -i :${VLLM_PORT} -t >/dev/null 2>&1; then
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null; then
            return 0
        fi
    fi
    return 1
}

# ========================
# Server Management
# ========================

ensure_vllm_running() {
    # Check if vLLM is already running and healthy BEFORE clearing GPU memory
    echo "🔍 Checking if vLLM server is already running on port ${VLLM_PORT}..."
    if is_vllm_healthy; then
        echo "✅ vLLM server is already running and healthy on port ${VLLM_PORT}"
        echo "   Reusing existing vLLM server (skipping GPU memory clear)"

        # Get the PID of the existing vLLM process
        VLLM_PID=$(lsof -i :${VLLM_PORT} -t | head -1)
        if [ -n "${VLLM_PID}" ]; then
            echo "   Existing vLLM PID: ${VLLM_PID}"
        fi
        return 0
    fi

    echo "   vLLM not running or not healthy, starting fresh..."
    wait_for_gpu_memory_clear
    start_vllm
}

start_vllm() {
    local NUM_GPUS
    NUM_GPUS="$(detect_num_gpus)"

    local MAX_RETRIES=3
    local RETRY_COUNT=0
    local TIMEOUT_ITERATIONS=300  # 300 × 2 sec = 600 seconds = 10 minutes per attempt

    # Check if vLLM is already running and healthy
    echo "🔍 Checking if vLLM server is already running on port ${VLLM_PORT}..."
    if lsof -i :${VLLM_PORT} -t >/dev/null 2>&1; then
        echo "   Port ${VLLM_PORT} is in use, checking health status..."
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null; then
            echo "✅ vLLM server is already running and healthy on port ${VLLM_PORT}"
            echo "   Reusing existing vLLM server"

            # Get the PID of the existing vLLM process
            VLLM_PID=$(lsof -i :${VLLM_PORT} -t | head -1)
            if [ -n "${VLLM_PID}" ]; then
                echo "   Existing vLLM PID: ${VLLM_PID}"
            fi
            return 0
        else
            echo "   Port is in use but vLLM is not healthy, will restart..."
            # Kill the unhealthy vLLM process
            lsof -i :${VLLM_PORT} -t | xargs -r kill -9 2>/dev/null || true
            pkill -9 -f "vllm serve" 2>/dev/null || true
            sleep 3
        fi
    else
        echo "   Port ${VLLM_PORT} is free, will start new vLLM server"
    fi

    # Enable flash attention for all machines (H100 and B200)
    echo "✅ Enabling flash attention"
    export VLLM_USE_TRITON_FLASH_ATTN=1

    # Determine log file name (rank-specific for multinode, default for single-node)
    local VLLM_LOG_FILE
    if [ -n "${MULTINODE_RANK}" ]; then
        VLLM_LOG_FILE="${HOST_DATA_PATH}/vllm_rank_${MULTINODE_RANK}.log"
    else
        VLLM_LOG_FILE="${HOST_DATA_PATH}/vllm.log"
    fi

    # Add reasoning flags for thinking models
    REASONING_FLAGS=""
    if [ "${MODEL_TYPE}" = "qwen-think" ]; then
        REASONING_FLAGS="--reasoning-parser deepseek_r1"
        echo "  Enabling reasoning with parser: deepseek_r1 (for Qwen thinking model)"
    fi

    while [ ${RETRY_COUNT} -lt ${MAX_RETRIES} ]; do
        RETRY_COUNT=$((RETRY_COUNT + 1))

        if [ -n "${MULTINODE_RANK}" ]; then
            echo "🚀 Starting vLLM server (attempt ${RETRY_COUNT}/${MAX_RETRIES}, rank ${MULTINODE_RANK})..."
        else
            echo "🚀 Starting vLLM server (attempt ${RETRY_COUNT}/${MAX_RETRIES})..."
        fi
        echo "   Model: ${MODEL_TO_SERVE}"
        echo "   Port: ${VLLM_PORT}"
        echo "   Log: ${VLLM_LOG_FILE}"

        # Start vLLM server
        vllm serve "${MODEL_TO_SERVE}" \
            --host 0.0.0.0 \
            --port "${VLLM_PORT}" \
            --max-num-seqs 512 \
            --gpu-memory-utilization 0.95 \
            --max-model-len 32768 \
            --tensor-parallel-size 1 \
            --data-parallel-size ${NUM_GPUS} \
            --limit-mm-per-prompt '{"video": 0}' \
            --allowed-local-media-path "${HOST_DATA_PATH}" \
            ${REASONING_FLAGS} \
            >"${VLLM_LOG_FILE}" 2>&1 &

        VLLM_PID=$!
        echo "   vLLM server started with PID: ${VLLM_PID}"

        # Wait for vLLM to become ready
        if wait_for_vllm ${TIMEOUT_ITERATIONS}; then
            echo "✅ vLLM server successfully started on attempt ${RETRY_COUNT}"
            return 0
        fi

        # If we get here, vLLM failed to start
        echo "⚠️  vLLM startup attempt ${RETRY_COUNT}/${MAX_RETRIES} failed (timeout after 10 minutes)"

        if [ ${RETRY_COUNT} -lt ${MAX_RETRIES} ]; then
            echo "🔄 Killing vLLM processes and retrying..."

            # Kill the vLLM server and all child processes
            if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
                echo "   Killing vLLM PID ${VLLM_PID}..."
                kill -9 "${VLLM_PID}" 2>/dev/null || true
            fi

            # Comprehensive cleanup of all vLLM processes
            echo "   Cleaning up all vLLM processes..."
            pkill -9 -f "vllm serve" 2>/dev/null || true
            pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
            sleep 2

            # Check if port is free
            if lsof -i :${VLLM_PORT} -t >/dev/null 2>&1; then
                echo "   Port ${VLLM_PORT} still in use, force killing..."
                lsof -i :${VLLM_PORT} -t | xargs -r kill -9 2>/dev/null || true
                sleep 2
            fi

            echo "   Waiting 5 seconds for GPU memory to clear..."
            sleep 5

            echo "   Retrying vLLM startup..."
        fi
    done

    # All retries exhausted
    echo "❌ FATAL: vLLM failed to start after ${MAX_RETRIES} attempts"
    echo "   Printing last 100 log lines from ${VLLM_LOG_FILE}:"
    tail -n 100 "${VLLM_LOG_FILE}" 2>/dev/null || echo "   (log file not found or empty)"
    exit 1
}

stop_vllm() {
    echo "🛑 Stopping vLLM server (comprehensive cleanup)..."

    # Step 1: Try graceful shutdown via PID if available
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "   Sending SIGTERM to PID ${VLLM_PID}..."
        kill "${VLLM_PID}" 2>/dev/null || true

        # Wait up to 10s for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
                echo "   ✅ PID ${VLLM_PID} stopped gracefully"
                break
            fi
            sleep 1
        done

        # Force kill if still running
        if kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "   Force killing PID ${VLLM_PID}..."
            kill -9 "${VLLM_PID}" 2>/dev/null || true
            sleep 1
        fi
    fi

    # Step 2: Kill all processes on vLLM port (catches orphaned servers)
    echo "   Checking port ${VLLM_PORT}..."
    if lsof -ti:${VLLM_PORT} >/dev/null 2>&1 || fuser ${VLLM_PORT}/tcp 2>/dev/null; then
        echo "   Killing processes on port ${VLLM_PORT}..."
        lsof -ti:${VLLM_PORT} 2>/dev/null | xargs -r kill -9 2>/dev/null || true
        fuser -k -9 ${VLLM_PORT}/tcp 2>/dev/null || true
        sleep 1
    fi

    # Step 3: Kill all vLLM processes by name (catches worker processes)
    echo "   Checking for vllm processes..."
    # Kill both old-style and new-style vLLM process names
    vllm_pids=$(pgrep -f "VLLM::|vllm.entrypoints.openai.api_server|vllm serve" 2>/dev/null || true)
    if [ -n "${vllm_pids}" ]; then
        echo "   Found vLLM processes: ${vllm_pids}"
        echo "   Killing all vLLM processes..."
        # Kill data parallel coordinator and engine cores
        pkill -9 -f "VLLM::DPCoordinator" 2>/dev/null || true
        pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
        # Kill old-style process names
        pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
        pkill -9 -f "vllm serve" 2>/dev/null || true
        sleep 2
    fi

    # Step 3b: Kill all processes using GPU (fallback for stubborn processes)
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "   Checking for GPU processes..."
        gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
        if [ -n "${gpu_pids}" ]; then
            echo "   Found GPU processes: ${gpu_pids}"
            echo "   Force killing GPU processes..."
            echo "${gpu_pids}" | xargs -r kill -9 2>/dev/null || true
            sleep 5  # GPU memory takes time to be released
        fi
    fi

    # Step 4: Verify port is free
    local max_port_checks=5
    for i in $(seq 1 ${max_port_checks}); do
        if ! lsof -ti:${VLLM_PORT} >/dev/null 2>&1 && ! fuser ${VLLM_PORT}/tcp 2>/dev/null; then
            echo "   ✅ Port ${VLLM_PORT} is free"
            VLLM_PID=""
            return 0
        fi
        echo "   Port ${VLLM_PORT} still in use, waiting... (${i}/${max_port_checks})"
        sleep 1
    done

    VLLM_PID=""
    echo "   ⚠️  Port ${VLLM_PORT} may still be in use, but continuing anyway"
}
