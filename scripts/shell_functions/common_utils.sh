#!/bin/bash
# Common utility functions for run.sh
# These are general-purpose helpers used across multiple modules

# ========================
# Port Management
# ========================

kill_port_process() {
    local port=$1
    echo "Checking for process on port ${port}..."

    # Kill any process using the port
    fuser -k ${port}/tcp 2>/dev/null || true

    # Wait for port to be actually released (up to 5 seconds)
    local wait_count=0
    while [ $wait_count -lt 10 ]; do
        if ! lsof -ti:${port} > /dev/null 2>&1 && ! fuser ${port}/tcp 2>/dev/null; then
            echo "✅ Port ${port} is now free"
            return 0
        fi
        echo "⏳ Waiting for port ${port} to be released..."
        sleep 0.5
        wait_count=$((wait_count + 1))
    done

    # Force kill if still not released
    echo "⚠️ Port ${port} still in use, force killing..."
    lsof -ti:${port} 2>/dev/null | xargs kill -9 2>/dev/null || true
    fuser -k -9 ${port}/tcp 2>/dev/null || true
    sleep 1
    echo "✅ Process on port ${port} killed (if any)"
}

# ========================
# Model Resolution
# ========================

resolve_model_to_serve() {
    echo "DEBUG: MODEL_PATH=${MODEL_PATH}" >&2
    echo "DEBUG: MODEL_TYPE=${MODEL_TYPE}" >&2
    echo "DEBUG: Checking if ${MODEL_PATH} exists as directory..." >&2

    # Determine default model based on MODEL_TYPE
    local default_model
    if [ "${MODEL_TYPE}" = "qwen-think" ]; then
        default_model="Qwen/Qwen3-VL-8B-Thinking"
    elif [ "${MODEL_TYPE}" = "qwen-instruct" ]; then
        default_model="Qwen/Qwen3-VL-8B-Instruct"
    else
        echo "ERROR: Invalid MODEL_TYPE '${MODEL_TYPE}'. Must be 'qwen-think' or 'qwen-instruct'" >&2
        exit 1
    fi

    if [ -d "${MODEL_PATH}" ] && [ "$(ls -A "${MODEL_PATH}")" ]; then
        echo "DEBUG: Using local model: ${MODEL_PATH}" >&2
        echo "${MODEL_PATH}"
    else
        echo "DEBUG: Using default HuggingFace model: ${default_model}" >&2
        echo "${default_model}"
    fi
}

# ========================
# GPU Detection
# ========================

detect_num_gpus() {
    # First check if CUDA_VISIBLE_DEVICES is set
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        # Count comma-separated GPU IDs
        IFS=',' read -ra GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
        echo "${#GPU_ARRAY[@]}"
    elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name --format=csv,noheader | wc -l
    elif command -v rocm-smi >/dev/null 2>&1; then
        # Count unique GPU indices on ROCm
        rocm-smi -i | awk -F'[][]' '/^GPU\[/{print $2}' | wc -l
    else
        echo 1
    fi
}

# ========================
# Memory Usage Display
# ========================

print_memory_usage() {
    local phase_name="${1:-Unknown Phase}"
    echo "=================================================="
    echo "📊 CPU Memory Usage After ${phase_name}"
    echo "=================================================="

    # Print memory info from /proc/meminfo
    if [ -f /proc/meminfo ]; then
        local mem_total=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        local mem_available=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
        local mem_free=$(grep MemFree /proc/meminfo | awk '{print $2}')
        local mem_used=$((mem_total - mem_available))
        local mem_used_percent=$((mem_used * 100 / mem_total))

        echo "Total Memory:     $((mem_total / 1024 / 1024)) GB"
        echo "Used Memory:      $((mem_used / 1024 / 1024)) GB (${mem_used_percent}%)"
        echo "Available Memory: $((mem_available / 1024 / 1024)) GB"
        echo "Free Memory:      $((mem_free / 1024 / 1024)) GB"
    fi

    # Print swap info
    if [ -f /proc/meminfo ]; then
        local swap_total=$(grep SwapTotal /proc/meminfo | awk '{print $2}')
        local swap_free=$(grep SwapFree /proc/meminfo | awk '{print $2}')
        if [ "${swap_total}" -gt 0 ]; then
            local swap_used=$((swap_total - swap_free))
            local swap_used_percent=$((swap_used * 100 / swap_total))
            echo "Swap Total:       $((swap_total / 1024 / 1024)) GB"
            echo "Swap Used:        $((swap_used / 1024 / 1024)) GB (${swap_used_percent}%)"
        else
            echo "Swap:             Not configured"
        fi
    fi

    echo "=================================================="
}

# ========================
# Iteration Tracking
# ========================

calculate_iteration_from_trajectories() {
    local traj_dir=$1

    if [ ! -d "${traj_dir}" ]; then
        echo "0"
        return 1
    fi

    # Extract max iteration number directly from filenames (much faster than loading files)
    local max_iteration=$(python -c "
import sys
import os
import re

traj_dir = '${traj_dir}'
try:
    if not os.path.exists(traj_dir):
        print(0)
        sys.exit(0)

    # Pattern: train_trajectories.pt.iterationN (exclude rank-specific files like _rank_0)
    pattern = re.compile(r'.*_trajectories\.pt\.iteration(\d+)$')
    max_iter = 0

    for filename in os.listdir(traj_dir):
        match = pattern.match(filename)
        if match:
            iteration_num = int(match.group(1))
            max_iter = max(max_iter, iteration_num)

    print(max_iter)
except Exception as e:
    print(0, file=sys.stderr)
    print(0)
" 2>/dev/null)

    echo "${max_iteration}"
    return 0
}

get_next_iteration() {
    local train_traj_dir="${HOST_DATA_PATH}/train_trajectories"
    local current_iteration=$(calculate_iteration_from_trajectories "${train_traj_dir}")
    local next_iteration=$((current_iteration + 1))
    echo "${next_iteration}"
}

should_run_test_rollout() {
    local current_iteration=$1  # Pass current iteration as argument
    local train_traj_dir="${HOST_DATA_PATH}/train_trajectories"

    # If no argument provided, calculate from files (for backward compatibility)
    if [ -z "${current_iteration}" ]; then
        if [ ! -d "${train_traj_dir}" ]; then
            echo "false"
            return 1
        fi
        current_iteration=$(calculate_iteration_from_trajectories "${train_traj_dir}")
    fi

    local remainder=$((current_iteration % TEST_ROLLOUT_INTERVAL))

    # Count iteration files for logging (much faster than loading all trajectories)
    local num_files=$(python -c "
import os
import re
traj_dir = '${train_traj_dir}'
pattern = re.compile(r'.*_trajectories\.pt\.iteration(\d+)$')
count = 0
if os.path.exists(traj_dir):
    for f in os.listdir(traj_dir):
        if pattern.match(f):
            count += 1
print(count)
" 2>/dev/null || echo "unknown")

    echo "Current iteration: ${current_iteration} (checking for test rollout)" >&2
    echo "Test rollout check: iteration=${current_iteration}, interval=${TEST_ROLLOUT_INTERVAL}, remainder=${remainder}" >&2

    if [ "${remainder}" -eq 1 ]; then
        echo "true"
        return 0
    else
        echo "false"
        return 1
    fi
}

# ========================
# Checkpoint Management
# ========================

find_last_checkpoint() {
    local model_dir=$1

    # Find all checkpoint directories (checkpoint-XXX format)
    local checkpoints=($(find "${model_dir}" -maxdepth 1 -type d -name "checkpoint-*" | sort -V))

    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo ""
        return 1
    fi

    # Return the last checkpoint (highest step number)
    echo "${checkpoints[-1]}"
    return 0
}

# ========================
# Single-node Sync
# ========================

wait_for_sync() {
    local process_name=$1
    local iteration=$2
    echo "[$process_name] Iteration ${iteration}: Checking for sync flag at ${SYNC_FLAG}..."

    if [ -f "${SYNC_FLAG}" ]; then
        echo "[$process_name] Found sync flag, other process finished first. Deleting flag and continuing..."
        rm -f "${SYNC_FLAG}"
    else
        echo "[$process_name] No sync flag found, this process finished first. Creating flag and waiting..."
        echo "${process_name}:${iteration}:$(date +%s)" > "${SYNC_FLAG}"

        echo "[$process_name] Waiting for other process to delete the sync flag..."
        while [ -f "${SYNC_FLAG}" ]; do
            sleep 2
        done
        echo "[$process_name] Sync flag deleted by other process, continuing to next iteration..."
    fi
}
