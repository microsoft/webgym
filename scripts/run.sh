#!/bin/bash
echo "=========================================="
echo "Network Configuration"
echo "=========================================="
echo "Public IP (for logging only): $([curl -s --max-time 2 https://ipinfo.io/ip] 2>/dev/null || echo 'N/A')"
echo "Internal cluster IP (used for multi-node training): $(hostname -I | awk '{print $1}')"
echo "Hostname: $(hostname)"
echo "=========================================="
echo ""

# ========================
# Argument Parsing
# ========================

print_usage() {
    echo "Usage: $0 --data-path <path> --log-path <path> --rl-phase <phase> [options] [hydra_overrides...]"
    echo ""
    echo "Required Arguments:"
    echo "  --data-path <path>          Path to shared data (HuggingFace cache, etc.)"
    echo "                              This is read-only and shared across experiments"
    echo ""
    echo "  --log-path <path>           Path to experiment logs (trajectories, checkpoints)"
    echo "                              This is experiment-specific and can differ per run"
    echo ""
    echo "  --rl-phase <phase>          RL phase to execute:"
    echo "                              - rollout: Data collection only (train and/or eval)"
    echo "                              - update: Training updates only"
    echo "                              - both: Complete RL loop (train->eval->update)"
    echo ""
    echo "Options:"
    echo "  --eval-interval <N>         How often to run eval (required when --rl-phase both)"
    echo "                              Eval runs when iteration % N == 1"
    echo ""
    echo "  --rollout-split <split>     Restrict rollout (required when --rl-phase rollout):"
    echo "                              - train-only: Only collect train trajectories"
    echo "                              - eval-only: Only collect eval trajectories"
    echo ""
    echo "  --num-nodes <N>             Number of nodes for distributed execution (default: 1)"
    echo "                              Works with any --rl-phase"
    echo ""
    echo "  --rank-weights <w1,w2,...>  Comma-separated load weights for each node (required when --num-nodes > 1)"
    echo "                              Controls rollout task distribution proportionally"
    echo "                              Example: \"1,1,1\" for 3 nodes with equal load"
    echo "                              Example: \"2,1,1\" to give node 0 twice the load"
    echo ""
    echo "  --master                    Designate this node as master (rank 0)"
    echo "                              Alternative to setting RANK=0 environment variable"
    echo ""
    echo "  --worker <rank>             Designate this node as worker with specified rank (1, 2, 3, ...)"
    echo "                              Alternative to setting RANK=<rank> environment variable"
    echo ""
    echo "  --debug-mode                Skip vLLM start/stop (assume user manages vLLM)"
    echo ""
    echo "  --resume                    Resume from an existing checkpoint file."
    echo "                              Requires --rollout-split to be specified."
    echo "                              If no checkpoint exists for the split, exits immediately."
    echo "                              NOTE: If --resume is NOT specified, the script will start"
    echo "                              fresh from that iteration, OVERWRITING any existing checkpoint."
    echo ""
    echo "Hydra Config Overrides:"
    echo "  Any additional arguments containing '=' are passed directly to Hydra."
    echo "  Use this to override any config values without editing YAML files."
    echo ""
    echo "  Common overrides:"
    echo "    env_config.train_tasks=<file>           Training task file (default: train.jsonl)"
    echo "    env_config.test_tasks=<file>            Test task file"
    echo "    env_config.train_tasks_rollout_size=<N> Number of training tasks per rollout"
    echo "    env_config.test_tasks_rollout_size=<N>  Number of test tasks per rollout"
    echo "    env_config.test_tasks_repeat_times=<N>  How many times to repeat test tasks"
    echo "    openai_config.model=<model>             Eval judge model (default: gpt-4o-mini)"
    echo "    model_config.model_type=<type>          Model type: qwen3-instruct, qwen3-think"
    echo "    env_config.train_tasks_sampler=<type>   Sampler: uniform, ratio"
    echo ""
    echo "Examples:"
    echo "  # Single-node complete RL loop"
    echo "  $0 --data-path /data/shared --log-path /data/exp1 --rl-phase both --eval-interval 6"
    echo ""
    echo "  # Single-node train-only rollout"
    echo "  $0 --data-path /data/shared --log-path /data/exp1 --rl-phase rollout --rollout-split train-only"
    echo ""
    echo "  # Single-node eval-only rollout"
    echo "  $0 --data-path /data/shared --log-path /data/exp1 --rl-phase rollout --rollout-split eval-only"
    echo ""
    echo "  # Multi-node (3 nodes) complete RL loop - using --master/--worker"
    echo "  # On master node:"
    echo "  $0 --data-path /data/shared --log-path /data/exp1 --rl-phase both --eval-interval 6 --num-nodes 3 --rank-weights \"1,1,1\" --master"
    echo "  # On worker nodes:"
    echo "  $0 --data-path /data/shared --log-path /data/exp1 --rl-phase both --eval-interval 6 --num-nodes 3 --rank-weights \"1,1,1\" --worker 1"
    echo "  $0 --data-path /data/shared --log-path /data/exp1 --rl-phase both --eval-interval 6 --num-nodes 3 --rank-weights \"1,1,1\" --worker 2"
    echo ""
    echo "  # Debug mode (vLLM already running)"
    echo "  $0 --data-path /data/shared --log-path /data/exp1 --rl-phase rollout --debug-mode"
    echo ""
    echo "  # Override config values"
    echo "  $0 --data-path /data/shared --log-path /data/exp1 --rl-phase rollout \\"
    echo "      env_config.train_tasks=my_tasks.jsonl \\"
    echo "      env_config.train_tasks_rollout_size=50 \\"
    echo "      openai_config.model=gpt-4o"
    exit 1
}

# Check minimum arguments
if [ $# -lt 6 ]; then
    print_usage
fi

# Initialize argument variables
DATA_PATH=""
LOG_PATH=""
RL_PHASE=""
EVAL_INTERVAL=""
ROLLOUT_SPLIT=""
NUM_NODES=1
RANK_WEIGHTS=""
NODE_ROLE_OVERRIDE=""  # Will be set to "master" or worker rank number
DEBUG_MODE=false
RESUME_MODE=false
HYDRA_OVERRIDES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --log-path)
            LOG_PATH="$2"
            shift 2
            ;;
        --rl-phase)
            RL_PHASE="$2"
            shift 2
            ;;
        --eval-interval)
            EVAL_INTERVAL="$2"
            shift 2
            ;;
        --rollout-split)
            ROLLOUT_SPLIT="$2"
            shift 2
            ;;
        --num-nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --rank-weights)
            RANK_WEIGHTS="$2"
            shift 2
            ;;
        --master)
            NODE_ROLE_OVERRIDE="0"
            shift
            ;;
        --worker)
            NODE_ROLE_OVERRIDE="$2"
            shift 2
            ;;
        --debug-mode)
            DEBUG_MODE=true
            shift
            ;;
        --resume)
            RESUME_MODE=true
            shift
            ;;
        *)
            # Check if this looks like a Hydra override (contains = or starts with + or ~)
            if [[ "$1" == *"="* ]] || [[ "$1" == "+"* ]] || [[ "$1" == "~"* ]]; then
                # Accumulate Hydra overrides
                HYDRA_OVERRIDES="${HYDRA_OVERRIDES} $1"
                shift
            else
                echo "Error: Unknown argument '$1'"
                echo "       (Hint: Config overrides should contain '=' e.g., env_config.train_tasks=foo.jsonl)"
                print_usage
            fi
            ;;
    esac
done

# Trim leading whitespace from HYDRA_OVERRIDES
HYDRA_OVERRIDES="${HYDRA_OVERRIDES# }"

# Validate --data-path
if [ -z "${DATA_PATH}" ]; then
    echo "Error: --data-path is required"
    print_usage
fi

# Normalize and validate data path
DATA_PATH="${DATA_PATH%/}"
if [ ! -d "${DATA_PATH}" ]; then
    echo "Error: Data path '${DATA_PATH}' does not exist"
    exit 1
fi

# Validate --log-path
if [ -z "${LOG_PATH}" ]; then
    echo "Error: --log-path is required"
    print_usage
fi

# Normalize log path (will be created if doesn't exist)
LOG_PATH="${LOG_PATH%/}"

# Validate --rl-phase
if [ -z "${RL_PHASE}" ]; then
    echo "Error: --rl-phase is required"
    print_usage
fi

if [ "${RL_PHASE}" != "rollout" ] && [ "${RL_PHASE}" != "update" ] && [ "${RL_PHASE}" != "both" ]; then
    echo "Error: --rl-phase must be 'rollout', 'update', or 'both'"
    exit 1
fi

# Validate --eval-interval (required for 'both')
if [ "${RL_PHASE}" = "both" ]; then
    if [ -z "${EVAL_INTERVAL}" ]; then
        echo "Error: --eval-interval is required when --rl-phase is 'both'"
        exit 1
    fi
    if ! [[ "${EVAL_INTERVAL}" =~ ^[0-9]+$ ]] || [ "${EVAL_INTERVAL}" -lt 1 ]; then
        echo "Error: --eval-interval must be a positive integer"
        exit 1
    fi
fi

# Validate --rollout-split (required for 'rollout', invalid for others)
if [ "${RL_PHASE}" = "rollout" ]; then
    if [ -z "${ROLLOUT_SPLIT}" ]; then
        echo "Error: --rollout-split is required when --rl-phase is 'rollout'"
        echo "  Use --rollout-split train-only or --rollout-split eval-only"
        exit 1
    fi
    if [ "${ROLLOUT_SPLIT}" != "train-only" ] && [ "${ROLLOUT_SPLIT}" != "eval-only" ]; then
        echo "Error: --rollout-split must be 'train-only' or 'eval-only'"
        exit 1
    fi
elif [ -n "${ROLLOUT_SPLIT}" ]; then
    echo "Error: --rollout-split is only valid when --rl-phase is 'rollout'"
    exit 1
fi

# Validate --resume (requires --rollout-split and --rl-phase rollout)
if [ "${RESUME_MODE}" = "true" ]; then
    if [ "${RL_PHASE}" != "rollout" ]; then
        echo "Error: --resume is only valid when --rl-phase is 'rollout'"
        exit 1
    fi
    if [ -z "${ROLLOUT_SPLIT}" ]; then
        echo "Error: --resume requires --rollout-split to be specified"
        echo "  Use --rollout-split train-only or --rollout-split eval-only"
        exit 1
    fi
fi

# Validate --num-nodes
if ! [[ "${NUM_NODES}" =~ ^[0-9]+$ ]] || [ "${NUM_NODES}" -lt 1 ]; then
    echo "Error: --num-nodes must be a positive integer"
    exit 1
fi

# Validate --rank-weights (required for multi-node)
if [ "${NUM_NODES}" -gt 1 ]; then
    if [ -z "${RANK_WEIGHTS}" ]; then
        echo "Error: --rank-weights is required when --num-nodes > 1"
        echo "  Example: --rank-weights \"1,1,1\" for 3 nodes with equal weight"
        exit 1
    fi

    # Count weights and validate format
    IFS=',' read -ra WEIGHT_ARRAY <<< "${RANK_WEIGHTS}"
    if [ ${#WEIGHT_ARRAY[@]} -ne "${NUM_NODES}" ]; then
        echo "Error: --rank-weights must have exactly ${NUM_NODES} comma-separated values (got ${#WEIGHT_ARRAY[@]})"
        echo "  Example: --rank-weights \"1,1,1\" for 3 nodes"
        exit 1
    fi

    # Validate each weight is a positive integer
    for weight in "${WEIGHT_ARRAY[@]}"; do
        if ! [[ "${weight}" =~ ^[0-9]+$ ]] || [ "${weight}" -lt 1 ]; then
            echo "Error: Each rank weight must be a positive integer (got '${weight}')"
            exit 1
        fi
    done

    # Validate --worker rank if specified
    if [ -n "${NODE_ROLE_OVERRIDE}" ] && [ "${NODE_ROLE_OVERRIDE}" != "0" ]; then
        if ! [[ "${NODE_ROLE_OVERRIDE}" =~ ^[0-9]+$ ]] || [ "${NODE_ROLE_OVERRIDE}" -lt 1 ] || [ "${NODE_ROLE_OVERRIDE}" -ge "${NUM_NODES}" ]; then
            echo "Error: --worker rank must be between 1 and $((NUM_NODES - 1)) for ${NUM_NODES} nodes (got '${NODE_ROLE_OVERRIDE}')"
            exit 1
        fi
    fi
fi

# ========================
# Display Configuration
# ========================

echo "Data path: ${DATA_PATH}"
echo "Log path: ${LOG_PATH}"
echo "RL Phase: ${RL_PHASE}"
if [ "${RL_PHASE}" = "both" ]; then
    echo "Eval interval: ${EVAL_INTERVAL}"
fi
if [ -n "${ROLLOUT_SPLIT}" ]; then
    echo "Rollout split: ${ROLLOUT_SPLIT}"
fi
echo "Nodes: ${NUM_NODES}"
if [ "${NUM_NODES}" -gt 1 ]; then
    echo "Rank weights: ${RANK_WEIGHTS}"
fi
if [ "${DEBUG_MODE}" = "true" ]; then
    echo "Debug mode: ENABLED (vLLM management disabled)"
fi
if [ "${RESUME_MODE}" = "true" ]; then
    echo "Resume mode: ENABLED"
fi
if [ -n "${HYDRA_OVERRIDES}" ]; then
    echo "Config overrides: ${HYDRA_OVERRIDES}"
fi

# ========================
# Multi-node Configuration
# ========================

# Determine node rank
MULTINODE_RANK=""
MULTINODE_NUM_WORKERS=""

if [ "${NUM_NODES}" -gt 1 ]; then
    MULTINODE_NUM_WORKERS="${NUM_NODES}"

    # Determine node rank (priority order):
    # 1. Command-line flags (--master or --worker)
    # 2. Environment variables (RANK or NODE_RANK)
    if [ -n "${NODE_ROLE_OVERRIDE}" ]; then
        MULTINODE_RANK="${NODE_ROLE_OVERRIDE}"
        if [ "${MULTINODE_RANK}" -eq 0 ]; then
            echo "Node role: Master (rank 0) - specified via --master"
        else
            echo "Node role: Worker (rank ${MULTINODE_RANK}) - specified via --worker"
        fi
    elif [ -n "${RANK:-}" ]; then
        MULTINODE_RANK="${RANK}"
        echo "Node role: Rank ${MULTINODE_RANK} - from RANK environment variable"
    elif [ -n "${NODE_RANK:-}" ]; then
        MULTINODE_RANK="${NODE_RANK}"
        echo "Node role: Rank ${MULTINODE_RANK} - from NODE_RANK environment variable"
    else
        echo "Error: Multi-node mode requires node rank specification via one of:"
        echo "  1. Command-line: --master (for rank 0) or --worker <rank> (for rank 1+)"
        echo "  2. Environment: Set RANK or NODE_RANK variable"
        exit 1
    fi

    echo "Multi-node mode: Rank ${MULTINODE_RANK} of ${MULTINODE_NUM_WORKERS} nodes"
fi

# ========================
# Rank Load Weights (from CLI)
# ========================

# Parse --rank-weights into associative array for multi-node
declare -A RANK_LOAD_WEIGHTS
if [ "${NUM_NODES}" -gt 1 ]; then
    IFS=',' read -ra WEIGHT_ARRAY <<< "${RANK_WEIGHTS}"
    for i in "${!WEIGHT_ARRAY[@]}"; do
        RANK_LOAD_WEIGHTS[$i]="${WEIGHT_ARRAY[$i]}"
    done
fi

# ========================
# Path Configuration
# ========================

HOST_DATA_PATH="${LOG_PATH}"
HF_HOME="${DATA_PATH}/.cache/huggingface/hub/"
# Get absolute path to project root (parent of scripts/ directory)
HOST_RUN_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Log path: ${HOST_DATA_PATH}"
echo "HF cache: ${HF_HOME}"

# ========================
# Model Configuration
# ========================

# MODEL_TYPE is used for shell-level logic (vLLM config, default HF model resolution)
# Options: "qwen-instruct", "qwen-think"
# Note: model_config.model_type in default.yaml should match this setting
MODEL_TYPE="qwen-instruct"

echo "Model type: ${MODEL_TYPE}"

# ========================
# Training Configuration
# ========================

# Set TEST_ROLLOUT_INTERVAL based on --eval-interval or default for rollout-only
if [ -n "${EVAL_INTERVAL}" ]; then
    TEST_ROLLOUT_INTERVAL="${EVAL_INTERVAL}"
else
    # For rollout-only mode, eval every iteration if not split
    TEST_ROLLOUT_INTERVAL=1
fi

MODEL_PATH="${HOST_DATA_PATH}/model.pt"
VLLM_PORT=8999
CURRENT_MAX_STEPS=100
SYNC_FLAG="${HOST_DATA_PATH}/sync_flag"

# Set environment variables
export HF_HOME="${HF_HOME}"
export DISABLE_VERSION_CHECK=1
export HYDRA_OVERRIDES="${HYDRA_OVERRIDES}"

# Delta cluster only (A40 GPUs)
if [ $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -c "A40") -gt 0 ]; then
    export CUDA_HOME="$CONDA_PREFIX"
    export CUDA_PATH="$CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$CONDA_PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

# Create directories
if [ -n "$HOST_DATA_PATH" ]; then
    mkdir -p "${HOST_DATA_PATH}"
    mkdir -p "${HOST_DATA_PATH}/checkpoints"
fi

VLLM_PID=""

# ========================
# Source utility functions
# ========================

SHELL_FUNCTIONS_DIR="${HOST_RUN_PATH}/scripts/shell_functions"

source "${SHELL_FUNCTIONS_DIR}/common_utils.sh"
source "${SHELL_FUNCTIONS_DIR}/gpu_utils.sh"
source "${SHELL_FUNCTIONS_DIR}/vllm_utils.sh"
source "${SHELL_FUNCTIONS_DIR}/training_utils.sh"
source "${SHELL_FUNCTIONS_DIR}/rollout_utils.sh"

# Source multinode utilities if multi-node mode
if [ "${NUM_NODES}" -gt 1 ]; then
    source "${SHELL_FUNCTIONS_DIR}/multinode_sync.sh"
fi

# ========================
# Checkpoint Detection for Resume Mode
# ========================

# Find the latest checkpoint file for a given split
# Returns the checkpoint path if found, empty string otherwise
find_latest_checkpoint() {
    local split=$1
    local traj_dir="${HOST_DATA_PATH}/${split}_trajectories"

    if [ ! -d "${traj_dir}" ]; then
        echo ""
        return
    fi

    # Find checkpoint files matching pattern: {split}_trajectories.pt.iteration{N}.checkpoint
    # Sort by iteration number (descending) and return the latest
    local latest_checkpoint=""
    local max_iteration=0

    for f in "${traj_dir}"/${split}_trajectories.pt.iteration*.checkpoint; do
        if [ -f "$f" ]; then
            # Extract iteration number from filename
            local iter_num=$(echo "$f" | sed -n "s/.*\.iteration\([0-9]*\)\.checkpoint$/\1/p")
            if [ -n "$iter_num" ] && [ "$iter_num" -gt "$max_iteration" ]; then
                max_iteration=$iter_num
                latest_checkpoint="$f"
            fi
        fi
    done

    echo "${latest_checkpoint}"
}

# Global variable to store the checkpoint path for resume mode
RESUME_CHECKPOINT_PATH=""

# Check for checkpoint when --resume is specified
if [ "${RESUME_MODE}" = "true" ]; then
    # Determine the split based on --rollout-split
    if [ "${ROLLOUT_SPLIT}" = "train-only" ]; then
        RESUME_SPLIT="train"
    elif [ "${ROLLOUT_SPLIT}" = "eval-only" ]; then
        RESUME_SPLIT="test"
    fi

    RESUME_CHECKPOINT_PATH=$(find_latest_checkpoint "${RESUME_SPLIT}")

    if [ -z "${RESUME_CHECKPOINT_PATH}" ]; then
        echo "Error: --resume specified but no checkpoint found for split '${RESUME_SPLIT}'"
        echo "  Expected checkpoint at: ${HOST_DATA_PATH}/${RESUME_SPLIT}_trajectories/${RESUME_SPLIT}_trajectories.pt.iteration*.checkpoint"
        exit 1
    fi

    echo "✅ Found checkpoint to resume: ${RESUME_CHECKPOINT_PATH}"
fi

# ========================
# Cleanup Handler
# ========================

cleanup() {
    echo "Shutting down (cleanup)..."
    if [ "${DEBUG_MODE}" = "false" ]; then
        stop_vllm
    fi
    rm -f "${SYNC_FLAG}"
    rm -rf "${HOST_DATA_PATH}/llamafactory_data"

    if [ -n "${MULTINODE_RANK}" ]; then
        echo "Cleaning up multinode flags for rank ${MULTINODE_RANK}..."
        rm -f "${HOST_DATA_PATH}/multinode_flags/instances_released_rank_${MULTINODE_RANK}"
        rm -f "${HOST_DATA_PATH}/multinode_flags/"*"_rank_${MULTINODE_RANK}"
        echo "✅ Multinode flags cleaned up for rank ${MULTINODE_RANK}"
    fi

    echo "Cleanup complete."
}
trap cleanup EXIT

# ========================
# vLLM Management Wrappers (respects --debug-mode)
# ========================

ensure_vllm_if_needed() {
    if [ "${DEBUG_MODE}" = "true" ]; then
        echo "🔧 Debug mode: Skipping vLLM startup (assuming user-managed)"
        return 0
    fi
    ensure_vllm_running
}

stop_vllm_if_needed() {
    if [ "${DEBUG_MODE}" = "true" ]; then
        echo "🔧 Debug mode: Skipping vLLM stop (assuming user-managed)"
        return 0
    fi
    stop_vllm
}

# ========================
# Phase: ROLLOUT (single-node)
# ========================

run_rollout_phase() {
    echo "============================================"
    echo "Starting ROLLOUT phase (infinite loop)"
    if [ "${ROLLOUT_SPLIT}" = "train-only" ]; then
        echo "Mode: Train trajectories only"
    elif [ "${ROLLOUT_SPLIT}" = "eval-only" ]; then
        echo "Mode: Eval trajectories only"
    else
        echo "Mode: Train + Eval trajectories"
    fi
    echo "============================================"

    MODEL_TO_SERVE="$(resolve_model_to_serve)"
    ensure_vllm_if_needed

    iteration=1
    while true; do
        echo ""
        echo "============================================"
        echo "ROLLOUT iteration ${iteration}"
        echo "============================================"

        # Train rollout (unless eval-only)
        if [ "${ROLLOUT_SPLIT}" != "eval-only" ]; then
            rollout_atomic_train "${MODEL_TO_SERVE}"
        fi

        # Eval rollout (unless train-only)
        if [ "${ROLLOUT_SPLIT}" != "train-only" ]; then
            echo "Running eval rollout..."
            rollout_atomic_test "${MODEL_TO_SERVE}"
        fi

        echo "Completed rollout iteration ${iteration}"
        iteration=$((iteration + 1))
    done
}

# ========================
# Phase: UPDATE (single-node)
# ========================

run_update_phase() {
    echo "============================================"
    echo "Starting UPDATE phase (infinite loop)"
    echo "============================================"

    iteration=1
    while true; do
        echo ""
        echo "============================================"
        echo "UPDATE iteration ${iteration}"
        echo "============================================"

        wait_for_gpu_memory_clear

        MODEL_TO_SERVE="$(resolve_model_to_serve)"
        update_atomic "${MODEL_TO_SERVE}" "update_online"

        if [ $? -ne 0 ]; then
            echo "❌ Update failed, exiting..."
            exit 1
        fi

        echo "Completed update iteration ${iteration}"
        iteration=$((iteration + 1))

        wait_for_gpu_memory_clear
    done
}

# ========================
# Phase: BOTH (single-node)
# ========================

run_both_phase() {
    echo "============================================"
    echo "Starting BOTH phase (rollout + update loop)"
    echo "Eval interval: ${EVAL_INTERVAL}"
    echo "============================================"

    MODEL_TO_SERVE="$(resolve_model_to_serve)"
    ensure_vllm_if_needed

    iteration=1
    while true; do
        echo ""
        echo "============================================"
        echo "BOTH iteration ${iteration}"
        echo "============================================"

        # --- ROLLOUT ---
        echo "--- ROLLOUT PHASE ---"

        # Train rollout
        rollout_atomic_train "${MODEL_TO_SERVE}"

        # Eval rollout (based on interval)
        if [ "$(should_run_test_rollout "${iteration}")" = "true" ]; then
            echo "Running eval rollout..."
            rollout_atomic_test "${MODEL_TO_SERVE}"
        else
            echo "Skipping eval rollout this iteration"
        fi

        echo "Rollout phase completed."

        # --- UPDATE ---
        echo ""
        echo "--- UPDATE PHASE ---"

        stop_vllm_if_needed
        wait_for_gpu_memory_clear

        update_atomic "${MODEL_TO_SERVE}" "update_online"

        if [ $? -ne 0 ]; then
            echo "❌ Update failed, restarting vLLM and continuing..."
            ensure_vllm_if_needed
        else
            NEW_MODEL_TO_SERVE="$(resolve_model_to_serve)"
            if [ "${NEW_MODEL_TO_SERVE}" != "${MODEL_TO_SERVE}" ]; then
                echo "Model updated from ${MODEL_TO_SERVE} to ${NEW_MODEL_TO_SERVE}"
                MODEL_TO_SERVE="${NEW_MODEL_TO_SERVE}"
            fi
            ensure_vllm_if_needed
        fi

        echo "Update phase completed."
        echo "Completed iteration ${iteration}"
        iteration=$((iteration + 1))
    done
}

# ========================
# Multi-node: ROLLOUT phase
# ========================

run_multinode_rollout_phase() {
    local is_master=false
    if [ "${MULTINODE_RANK}" -eq 0 ]; then
        is_master=true
    fi

    echo "============================================"
    if [ "${is_master}" = "true" ]; then
        echo "Starting MULTINODE ROLLOUT phase (Master, rank 0)"
    else
        echo "Starting MULTINODE ROLLOUT phase (Worker, rank ${MULTINODE_RANK})"
    fi
    echo "============================================"

    # Master initializes sync
    if [ "${is_master}" = "true" ]; then
        init_multinode_sync "${MULTINODE_NUM_WORKERS}"
        cleanup_all_flags

        MASTER_IP=$(hostname -I | awk '{print $1}')
        save_master_ip "${MASTER_IP}"
    else
        # Workers wait for master
        while [ ! -f "${HOST_DATA_PATH}/multinode_flags/total_workers" ]; do
            sleep 2
        done
        MULTINODE_NUM_WORKERS=$(get_total_workers)
        check_node_rank_valid "${MULTINODE_RANK}" || exit 1
    fi

    MODEL_TO_SERVE="$(resolve_model_to_serve)"
    ensure_vllm_if_needed

    while true; do
        local iteration=$(get_next_iteration)
        echo ""
        echo "============================================"
        echo "MULTINODE ROLLOUT iteration ${iteration} (rank ${MULTINODE_RANK})"
        echo "============================================"

        # Phase 0: Sync at start
        create_phase_flag "${MULTINODE_RANK}" "iteration_${iteration}_start"
        wait_for_phase "${MULTINODE_RANK}" "iteration_${iteration}_start"

        if [ "${is_master}" = "true" ]; then
            sleep 20
            cleanup_phase_flags "iteration_${iteration}_start"
            # Remove stale Python-created flags from previous iteration
            # (cannot use cleanup_all_flags here - it would wipe flags workers create concurrently)
            rm -f "${HOST_DATA_PATH}/multinode_flags/instances_released_rank_"*
            rm -f "${HOST_DATA_PATH}/multinode_flags/train_sampling_complete_rank_"*
            rm -f "${HOST_DATA_PATH}/multinode_flags/train_task_indices_rank_"*
        fi

        # Train rollout (unless eval-only)
        if [ "${ROLLOUT_SPLIT}" != "eval-only" ]; then
            if [ "${is_master}" = "true" ]; then
                rollout_atomic_train "${MODEL_TO_SERVE}" "_rank_${MULTINODE_RANK}"
            else
                wait_for_master_phase "${MULTINODE_RANK}" "instances_released"
                sleep 120
                wait_for_master_phase "${MULTINODE_RANK}" "train_sampling_complete"
                rollout_atomic_train "${MODEL_TO_SERVE}" "_rank_${MULTINODE_RANK}"
            fi

            create_phase_flag "${MULTINODE_RANK}" "train_rollout_complete"
            wait_for_phase "${MULTINODE_RANK}" "train_rollout_complete"

            if [ "${is_master}" = "true" ]; then
                sleep 20
                cleanup_phase_flags "train_rollout_complete"
            fi

            # Aggregate train trajectories (master only)
            create_phase_flag "${MULTINODE_RANK}" "train_files_ready"
            wait_for_phase "${MULTINODE_RANK}" "train_files_ready"

            if [ "${is_master}" = "true" ]; then
                aggregate_trajectories "train"
                sleep 20
                cleanup_phase_flags "train_files_ready"
            fi
        else
            echo "⏭️  Skipping train rollout (eval-only mode)"
            # In eval-only mode, sync all nodes before proceeding to test rollout
            echo "Syncing all nodes before test rollout (eval-only)..."
            create_phase_flag "${MULTINODE_RANK}" "eval_only_sync"
            wait_for_phase "${MULTINODE_RANK}" "eval_only_sync"
            echo "All nodes synced (eval-only)"

            create_phase_flag "${MULTINODE_RANK}" "eval_only_sync_confirmed"
            wait_for_phase "${MULTINODE_RANK}" "eval_only_sync_confirmed"
            echo "All nodes confirmed sync exit (eval-only)"

            if [ "${is_master}" = "true" ]; then
                sleep 20
                cleanup_phase_flags "eval_only_sync"
                cleanup_phase_flags "eval_only_sync_confirmed"
            fi
        fi

        # Eval rollout (unless train-only)
        if [ "${ROLLOUT_SPLIT}" != "train-only" ]; then
            # Master decides if eval should run
            local RUN_EVAL="false"
            if [ "${ROLLOUT_SPLIT}" = "eval-only" ]; then
                RUN_EVAL="true"
            elif [ "$(should_run_test_rollout "${iteration}")" = "true" ]; then
                RUN_EVAL="true"
            fi

            if [ "${is_master}" = "true" ]; then
                echo "${RUN_EVAL}" > "${HOST_DATA_PATH}/multinode_flags/run_test_rollout"
                sync
            fi

            create_phase_flag "${MULTINODE_RANK}" "eval_decision"
            wait_for_phase "${MULTINODE_RANK}" "eval_decision"

            RUN_EVAL=$(cat "${HOST_DATA_PATH}/multinode_flags/run_test_rollout" 2>/dev/null || echo "false")

            if [ "${is_master}" = "true" ]; then
                sleep 20
                cleanup_phase_flags "eval_decision"
            fi

            if [ "${RUN_EVAL}" = "true" ]; then
                if [ "${is_master}" = "true" ]; then
                    rollout_atomic_test "${MODEL_TO_SERVE}" "${MULTINODE_RANK}" "${MULTINODE_NUM_WORKERS}"
                else
                    # Worker: wait for master to finish releasing instances before starting
                    echo "⏳ Worker rank ${MULTINODE_RANK}: Waiting for master to release instances..."
                    wait_for_master_phase "${MULTINODE_RANK}" "instances_released"
                    echo "✅ Worker rank ${MULTINODE_RANK}: Master finished instance cleanup"
                    echo "⏳ Worker rank ${MULTINODE_RANK}: Waiting 120s for instances to stabilize..."
                    sleep 120
                    echo "✅ Worker rank ${MULTINODE_RANK}: Ready to start allocating instances"
                    rollout_atomic_test "${MODEL_TO_SERVE}" "${MULTINODE_RANK}" "${MULTINODE_NUM_WORKERS}"
                fi

                create_phase_flag "${MULTINODE_RANK}" "eval_rollout_complete"
                wait_for_phase "${MULTINODE_RANK}" "eval_rollout_complete"

                if [ "${is_master}" = "true" ]; then
                    aggregate_trajectories "test"
                    sleep 20
                    cleanup_phase_flags "eval_rollout_complete"
                fi
            fi
        fi

        # Final sync
        create_phase_flag "${MULTINODE_RANK}" "iteration_complete"
        wait_for_phase "${MULTINODE_RANK}" "iteration_complete"

        if [ "${is_master}" = "true" ]; then
            sleep 20
            cleanup_phase_flags "iteration_complete"
        fi

        echo "✅ Multinode rollout iteration ${iteration} complete!"
    done
}

# ========================
# Multi-node: UPDATE phase
# ========================

run_multinode_update_phase() {
    local is_master=false
    if [ "${MULTINODE_RANK}" -eq 0 ]; then
        is_master=true
    fi

    echo "============================================"
    if [ "${is_master}" = "true" ]; then
        echo "Starting MULTINODE UPDATE phase (Master, rank 0)"
    else
        echo "Starting MULTINODE UPDATE phase (Worker, rank ${MULTINODE_RANK})"
    fi
    echo "============================================"

    # Master initializes sync
    if [ "${is_master}" = "true" ]; then
        init_multinode_sync "${MULTINODE_NUM_WORKERS}"
        cleanup_all_flags

        MASTER_IP=$(hostname -I | awk '{print $1}')
        save_master_ip "${MASTER_IP}"
    else
        while [ ! -f "${HOST_DATA_PATH}/multinode_flags/total_workers" ]; do
            sleep 2
        done
        MULTINODE_NUM_WORKERS=$(get_total_workers)
        check_node_rank_valid "${MULTINODE_RANK}" || exit 1
        MASTER_IP=$(wait_for_master_ip)
    fi

    while true; do
        local iteration=$(get_next_iteration)
        echo ""
        echo "============================================"
        echo "MULTINODE UPDATE iteration ${iteration} (rank ${MULTINODE_RANK})"
        echo "============================================"

        # Sync at start
        create_phase_flag "${MULTINODE_RANK}" "update_start"
        wait_for_phase "${MULTINODE_RANK}" "update_start"

        if [ "${is_master}" = "true" ]; then
            sleep 20
            cleanup_phase_flags "update_start"
        fi

        wait_for_gpu_memory_clear

        # Master prepares training data
        if [ "${is_master}" = "true" ]; then
            MODEL_TO_SERVE="$(resolve_model_to_serve)"
            update_atomic "${MODEL_TO_SERVE}" "update_online" true "${MULTINODE_NUM_WORKERS}"
        fi

        # Wait for training config
        local config_path="${HOST_DATA_PATH}/llamafactory_data/train_config.yaml"

        create_phase_flag "${MULTINODE_RANK}" "training_ready"
        wait_for_phase "${MULTINODE_RANK}" "training_ready"

        if [ "${is_master}" = "true" ]; then
            sleep 20
            cleanup_phase_flags "training_ready"
        fi

        # Run multi-node training
        if [ -f "${config_path}" ]; then
            run_llamafactory_training "${config_path}" true "${MULTINODE_RANK}" "${MULTINODE_NUM_WORKERS}" "${MASTER_IP}" 29500
            local training_exit_code=$?

            if [ "${is_master}" = "true" ] && [ ${training_exit_code} -eq 0 ]; then
                # Copy checkpoint to model.pt
                local latest_checkpoint_dir=$(find "${HOST_DATA_PATH}/checkpoints" -maxdepth 1 -type d -name "model_*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
                if [ -n "${latest_checkpoint_dir}" ] && [ -d "${latest_checkpoint_dir}" ]; then
                    local last_checkpoint=$(find_last_checkpoint "${latest_checkpoint_dir}")
                    if [ -n "${last_checkpoint}" ] && [ -d "${last_checkpoint}" ]; then
                        rm -rf "${MODEL_PATH}"
                        fast_parallel_copy "${last_checkpoint}" "${MODEL_PATH}"
                        echo "✅ Model updated: ${MODEL_PATH}"
                    fi
                fi
            fi
        fi

        # Sync at end
        create_phase_flag "${MULTINODE_RANK}" "training_complete"
        wait_for_phase "${MULTINODE_RANK}" "training_complete"

        if [ "${is_master}" = "true" ]; then
            sleep 20
            cleanup_phase_flags "training_complete"
        fi

        cleanup_deepspeed_processes

        echo "✅ Multinode update iteration ${iteration} complete!"
    done
}

# ========================
# Multi-node: BOTH phase
# ========================

run_multinode_both_phase() {
    local is_master=false
    if [ "${MULTINODE_RANK}" -eq 0 ]; then
        is_master=true
    fi

    echo "============================================"
    if [ "${is_master}" = "true" ]; then
        echo "Starting MULTINODE BOTH phase (Master, rank 0)"
    else
        echo "Starting MULTINODE BOTH phase (Worker, rank ${MULTINODE_RANK})"
    fi
    echo "Eval interval: ${EVAL_INTERVAL}"
    echo "============================================"

    # Master initializes sync
    if [ "${is_master}" = "true" ]; then
        init_multinode_sync "${MULTINODE_NUM_WORKERS}"
        cleanup_all_flags

        MASTER_IP=$(hostname -I | awk '{print $1}')
        save_master_ip "${MASTER_IP}"
    else
        while [ ! -f "${HOST_DATA_PATH}/multinode_flags/total_workers" ]; do
            sleep 2
        done
        MULTINODE_NUM_WORKERS=$(get_total_workers)
        check_node_rank_valid "${MULTINODE_RANK}" || exit 1

        MASTER_IP=$(wait_for_master_ip)
    fi

    MODEL_TO_SERVE="$(resolve_model_to_serve)"
    ensure_vllm_if_needed

    while true; do
        local iteration=$(get_next_iteration)
        echo ""
        echo "============================================"
        echo "MULTINODE BOTH iteration ${iteration} (rank ${MULTINODE_RANK})"
        echo "============================================"

        # Phase 0: Sync at iteration start
        create_phase_flag "${MULTINODE_RANK}" "iteration_${iteration}_start"
        wait_for_phase "${MULTINODE_RANK}" "iteration_${iteration}_start"
        create_phase_flag "${MULTINODE_RANK}" "iteration_${iteration}_start_confirmed"
        wait_for_phase "${MULTINODE_RANK}" "iteration_${iteration}_start_confirmed"

        if [ "${is_master}" = "true" ]; then
            sleep 20
            cleanup_phase_flags "iteration_${iteration}_start"
            cleanup_phase_flags "iteration_${iteration}_start_confirmed"

            # Cleanup previous iteration artifacts
            rm -f "${HOST_DATA_PATH}"/train_trajectories_rank_*.pt
            rm -f "${HOST_DATA_PATH}"/test_trajectories_rank_*.pt
            rm -f "${HOST_DATA_PATH}"/multinode_flags/run_test_rollout
            cleanup_all_flags
        fi

        # --- TRAIN ROLLOUT ---
        echo "--- TRAIN ROLLOUT PHASE ---"

        if [ "${is_master}" = "true" ]; then
            rollout_atomic_train "${MODEL_TO_SERVE}" "_rank_${MULTINODE_RANK}"
        else
            sleep 25
            wait_for_master_phase "${MULTINODE_RANK}" "instances_released"
            sleep 120
            wait_for_master_phase "${MULTINODE_RANK}" "train_sampling_complete"
            rollout_atomic_train "${MODEL_TO_SERVE}" "_rank_${MULTINODE_RANK}"
        fi

        create_phase_flag "${MULTINODE_RANK}" "train_rollout_complete"
        wait_for_phase "${MULTINODE_RANK}" "train_rollout_complete"

        if [ "${is_master}" = "true" ]; then
            sleep 20
            cleanup_phase_flags "train_rollout_complete"
        fi

        create_phase_flag "${MULTINODE_RANK}" "train_files_ready"
        wait_for_phase "${MULTINODE_RANK}" "train_files_ready"

        if [ "${is_master}" = "true" ]; then
            aggregate_trajectories "train"
            sleep 20
            cleanup_phase_flags "train_files_ready"
            print_memory_usage "Train Rollout"
        fi

        # --- EVAL ROLLOUT (if needed) ---
        local current_iteration=$(calculate_iteration_from_trajectories "${HOST_DATA_PATH}/train_trajectories")
        local RUN_EVAL="false"

        if [ "${is_master}" = "true" ]; then
            RUN_EVAL=$(should_run_test_rollout "${current_iteration}")
            echo "${RUN_EVAL}" > "${HOST_DATA_PATH}/multinode_flags/run_test_rollout"
            sync
        fi

        create_phase_flag "${MULTINODE_RANK}" "eval_decision_written"
        wait_for_phase "${MULTINODE_RANK}" "eval_decision_written"

        RUN_EVAL=$(cat "${HOST_DATA_PATH}/multinode_flags/run_test_rollout" 2>/dev/null || echo "false")

        create_phase_flag "${MULTINODE_RANK}" "eval_decision_made"
        wait_for_phase "${MULTINODE_RANK}" "eval_decision_made"

        if [ "${is_master}" = "true" ]; then
            sleep 20
            cleanup_phase_flags "eval_decision_written"
            cleanup_phase_flags "eval_decision_made"
        fi

        if [ "${RUN_EVAL}" = "true" ]; then
            echo "--- EVAL ROLLOUT PHASE ---"
            if [ "${is_master}" = "true" ]; then
                rollout_atomic_test "${MODEL_TO_SERVE}" "${MULTINODE_RANK}" "${MULTINODE_NUM_WORKERS}"
            else
                # Worker: wait for master to finish releasing instances before starting
                echo "⏳ Worker rank ${MULTINODE_RANK}: Waiting for master to release instances..."
                wait_for_master_phase "${MULTINODE_RANK}" "instances_released"
                echo "✅ Worker rank ${MULTINODE_RANK}: Master finished instance cleanup"
                echo "⏳ Worker rank ${MULTINODE_RANK}: Waiting 120s for instances to stabilize..."
                sleep 120
                echo "✅ Worker rank ${MULTINODE_RANK}: Ready to start allocating instances"
                rollout_atomic_test "${MODEL_TO_SERVE}" "${MULTINODE_RANK}" "${MULTINODE_NUM_WORKERS}"
            fi

            create_phase_flag "${MULTINODE_RANK}" "eval_rollout_complete"
            wait_for_phase "${MULTINODE_RANK}" "eval_rollout_complete"

            if [ "${is_master}" = "true" ]; then
                sleep 20
                cleanup_phase_flags "eval_rollout_complete"
            fi

            create_phase_flag "${MULTINODE_RANK}" "eval_files_ready"
            wait_for_phase "${MULTINODE_RANK}" "eval_files_ready"

            if [ "${is_master}" = "true" ]; then
                aggregate_trajectories "test"
                sleep 20
                cleanup_phase_flags "eval_files_ready"
                print_memory_usage "Eval Rollout"
            fi
        else
            echo "--- Skipping eval rollout this iteration ---"
        fi

        # --- UPDATE PHASE ---
        echo "--- UPDATE PHASE ---"

        stop_vllm_if_needed
        wait_for_gpu_memory_clear

        create_phase_flag "${MULTINODE_RANK}" "vllm_stopped"
        wait_for_phase "${MULTINODE_RANK}" "vllm_stopped"

        if [ "${is_master}" = "true" ]; then
            sleep 20
            cleanup_phase_flags "vllm_stopped"

            MODEL_TO_SERVE="$(resolve_model_to_serve)"
            update_atomic "${MODEL_TO_SERVE}" "update_online" true "${MULTINODE_NUM_WORKERS}"
        fi

        # Wait for training config
        local config_path="${HOST_DATA_PATH}/llamafactory_data/train_config.yaml"
        while [ ! -f "${config_path}" ]; do
            sleep 2
        done

        create_phase_flag "${MULTINODE_RANK}" "training_ready"
        wait_for_phase "${MULTINODE_RANK}" "training_ready"

        if [ "${is_master}" = "true" ]; then
            sleep 20
            cleanup_phase_flags "training_ready"
        fi

        # Run multi-node training
        if [ -f "${config_path}" ]; then
            run_llamafactory_training "${config_path}" true "${MULTINODE_RANK}" "${MULTINODE_NUM_WORKERS}" "${MASTER_IP}" 29500
            local training_exit_code=$?

            if [ "${is_master}" = "true" ] && [ ${training_exit_code} -eq 0 ]; then
                local latest_checkpoint_dir=$(find "${HOST_DATA_PATH}/checkpoints" -maxdepth 1 -type d -name "model_*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
                if [ -n "${latest_checkpoint_dir}" ] && [ -d "${latest_checkpoint_dir}" ]; then
                    local last_checkpoint=$(find_last_checkpoint "${latest_checkpoint_dir}")
                    if [ -n "${last_checkpoint}" ] && [ -d "${last_checkpoint}" ]; then
                        rm -rf "${MODEL_PATH}"

                        local ckpt_iteration=$(calculate_iteration_from_trajectories "${HOST_DATA_PATH}/train_trajectories")
                        local remainder=$((ckpt_iteration % TEST_ROLLOUT_INTERVAL))
                        if [ ${remainder} -eq 1 ]; then
                            fast_parallel_copy "${last_checkpoint}" "${MODEL_PATH}"
                            echo "✅ Model updated (eval iteration - preserved original)"
                        else
                            mv "${last_checkpoint}" "${MODEL_PATH}"
                            echo "✅ Model updated (moved to save disk space)"
                        fi
                    fi
                fi
            fi
        fi

        create_phase_flag "${MULTINODE_RANK}" "training_complete"
        wait_for_phase "${MULTINODE_RANK}" "training_complete"

        if [ "${is_master}" = "true" ]; then
            sleep 20
            cleanup_phase_flags "training_complete"
            print_memory_usage "Training Update"
        fi

        # Restart vLLM
        NEW_MODEL_TO_SERVE="$(resolve_model_to_serve)"
        if [ "${NEW_MODEL_TO_SERVE}" != "${MODEL_TO_SERVE}" ]; then
            MODEL_TO_SERVE="${NEW_MODEL_TO_SERVE}"
        fi

        cleanup_deepspeed_processes
        ensure_vllm_if_needed

        # Final sync
        create_phase_flag "${MULTINODE_RANK}" "iteration_complete"
        wait_for_phase "${MULTINODE_RANK}" "iteration_complete"

        if [ "${is_master}" = "true" ]; then
            sleep 20
            cleanup_phase_flags "iteration_complete"
        fi

        echo "⏳ Grace period: 10 seconds before next iteration..."
        sleep 10

        echo "✅ Multinode BOTH iteration ${iteration} complete!"
    done
}

# ========================
# Main Execution
# ========================

echo ""
echo "Starting WebGym RL training..."
echo "Phase: ${RL_PHASE}"
echo "Data path: ${HOST_DATA_PATH}"
echo "Model path: ${MODEL_PATH}"
echo ""

if [ "${NUM_NODES}" -gt 1 ]; then
    # Multi-node execution
    case "${RL_PHASE}" in
        "rollout")
            run_multinode_rollout_phase
            ;;
        "update")
            run_multinode_update_phase
            ;;
        "both")
            run_multinode_both_phase
            ;;
    esac
else
    # Single-node execution
    case "${RL_PHASE}" in
        "rollout")
            run_rollout_phase
            ;;
        "update")
            run_update_phase
            ;;
        "both")
            run_both_phase
            ;;
    esac
fi
