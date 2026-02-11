#!/bin/bash
# GPU memory management utilities
# Handles GPU memory cleanup and DeepSpeed process cleanup

# ========================
# DeepSpeed Cleanup
# ========================

cleanup_deepspeed_processes() {
    echo "🧹 Cleaning up lingering DeepSpeed/Training processes..."

    # Kill DeepSpeed training processes
    pkill -9 -f "deepspeed" 2>/dev/null && echo "   ✓ Killed deepspeed processes" || true
    pkill -9 -f "train.py" 2>/dev/null && echo "   ✓ Killed train.py processes" || true
    pkill -9 -f "llamafactory" 2>/dev/null && echo "   ✓ Killed llamafactory processes" || true
    pkill -9 -f "torch.distributed" 2>/dev/null && echo "   ✓ Killed torch.distributed processes" || true

    # Kill ALL Python processes that might be using GPU (more aggressive)
    echo "   Killing all Python GPU processes..."
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
        if [ -n "${gpu_pids}" ]; then
            echo "   Found GPU processes: ${gpu_pids}"
            echo "${gpu_pids}" | xargs -r kill -9 2>/dev/null || true
            echo "   ✓ Killed GPU processes"
        fi
    fi

    # Wait for processes to die
    sleep 5

    # Force GPU memory cleanup with Python
    python3 -c "
import torch
import gc
import sys
try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        print('   ✓ Cleared CUDA cache', file=sys.stderr)
except Exception as e:
    print(f'   ⚠ CUDA cache clear failed: {e}', file=sys.stderr)
" 2>&1 || echo "   ⚠ Could not clear CUDA cache (python/torch not available)"

    # Clean up shared memory segments
    echo "   Cleaning up IPC shared memory..."
    rm -rf /dev/shm/nccl-* 2>/dev/null && echo "   ✓ Removed NCCL shared memory" || true
    rm -rf /dev/shm/torch-* 2>/dev/null && echo "   ✓ Removed Torch shared memory" || true
    rm -rf /dev/shm/pymp-* 2>/dev/null && echo "   ✓ Removed PyMP shared memory" || true

    echo "✅ DeepSpeed cleanup complete"
    echo "   Waiting additional 10 seconds for GPU memory to fully release..."
    sleep 10
}

# ========================
# GPU Memory Management
# ========================

wait_for_gpu_memory_clear() {
    echo "Waiting for GPU memory to be cleared..."
    local max_attempts=180  # Wait up to 6 minutes before force-killing (increased from 2 min)
    local threshold_mb=10000  # Consider memory "cleared" if usage is below 10GB per GPU (increased from 2GB for safety)
    local kill_attempts=0
    local max_kill_attempts=5

    # IMMEDIATE CHECK: If GPU processes exist, kill them right away (don't wait)
    if command -v nvidia-smi >/dev/null 2>&1; then
        local initial_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{max=0}{if($1>max)max=$1}END{print max}')

        if [ -n "${initial_usage}" ] && [ "${initial_usage}" -ge "${threshold_mb}" ]; then
            echo "⚠️  Detected GPU memory in use: ${initial_usage} MB (threshold: ${threshold_mb} MB)"
            echo "🔍 Checking for lingering GPU processes..."

            local gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
            if [ -n "${gpu_pids}" ]; then
                echo "🔪 Found lingering GPU processes: ${gpu_pids}"
                echo "   Listing processes:"
                nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
                echo "   Force-killing immediately..."

                # Kill all vLLM processes first
                pkill -9 -f "VLLM::" 2>/dev/null || true

                # Kill GPU processes by PID
                echo "${gpu_pids}" | xargs -r kill -9 2>/dev/null || true

                echo "⏳ Waiting 5 seconds for GPU memory to be released..."
                sleep 5
            fi
        fi
    fi

    while true; do
        for i in $(seq 1 ${max_attempts}); do
            # Get GPU memory usage in MB for all GPUs
            local max_usage=0
            if command -v nvidia-smi >/dev/null 2>&1; then
                # Get memory used in MB for each GPU, find the maximum
                max_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{max=0}{if($1>max)max=$1}END{print max}')

                if [ -z "${max_usage}" ]; then
                    echo "Warning: Could not query GPU memory, assuming cleared"
                    return 0
                fi

                # Print progress every 30 attempts
                if [ $((i % 30)) -eq 0 ]; then
                    echo "Attempt ${i}/${max_attempts}: Max GPU memory usage: ${max_usage} MB"
                fi

                if [ "${max_usage}" -lt "${threshold_mb}" ]; then
                    echo ""
                    echo "✅ GPU memory cleared (below ${threshold_mb} MB threshold)"
                    return 0
                fi
            else
                echo "Warning: nvidia-smi not available, skipping GPU memory check"
                return 0
            fi

            sleep 2
        done

        # Memory still not cleared after max_attempts
        kill_attempts=$((kill_attempts + 1))

        if [ ${kill_attempts} -gt ${max_kill_attempts} ]; then
            echo "❌ ERROR: GPU memory did not clear after ${max_kill_attempts} force-kill attempts"
            echo "❌ This likely indicates a critical system issue"
            return 1
        fi

        echo "⚠️  GPU memory did not clear within timeout (${max_usage} MB still in use)"
        echo "🔪 Force-killing all GPU processes (attempt ${kill_attempts}/${max_kill_attempts})..."

        # Get all GPU process PIDs and force kill them
        local gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
        if [ -n "${gpu_pids}" ]; then
            echo "Killing GPU processes: ${gpu_pids}"
            echo "${gpu_pids}" | xargs -r kill -9 2>/dev/null || true
        else
            echo "No GPU processes found, trying pkill on vLLM and python..."
            pkill -9 -f "VLLM::" || true
            pkill -9 python || true
        fi

        # Clean up shared memory again
        echo "Cleaning up shared memory..."
        rm -rf /dev/shm/nccl-* 2>/dev/null || true
        rm -rf /dev/shm/torch-* 2>/dev/null || true
        rm -rf /dev/shm/pymp-* 2>/dev/null || true

        echo "⏳ Waiting 10 seconds for GPU memory to be released..."
        sleep 10

        # Check if memory is now cleared
        max_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{max=0}{if($1>max)max=$1}END{print max}')
        echo "After force-kill: Max GPU memory usage: ${max_usage} MB"

        if [ "${max_usage}" -lt "${threshold_mb}" ]; then
            echo "✅ GPU memory cleared after force-killing GPU processes"
            return 0
        fi

        echo "🔄 Memory still not cleared, will try again..."
    done
}
