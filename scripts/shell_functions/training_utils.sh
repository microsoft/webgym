#!/bin/bash
# LLaMA-Factory training utilities
# Handles training execution and checkpoint management

# Required global variables:
#   HOST_DATA_PATH - Data directory
#   HOST_RUN_PATH - Scripts directory
#   MODEL_PATH - Model checkpoint path
#   MODEL_TO_SERVE - Model to serve

# ========================
# Fast Parallel Copy
# ========================

# Fast parallel copy - matches DeepSpeed's parallel save speed
fast_parallel_copy() {
    local src="$1"
    local dst="$2"
    local parallel_jobs=20

    echo "🚀 Starting parallel copy (${parallel_jobs} parallel jobs)..."
    echo "   Source: ${src}"
    echo "   Destination: ${dst}"

    # Create destination directory
    mkdir -p "${dst}"

    # Create directory structure first
    (cd "${src}" && find . -type d) | while read -r dir; do
        mkdir -p "${dst}/${dir}"
    done

    # Use xargs -P for parallel file copying with rsync
    # This mimics how DeepSpeed writes multiple files simultaneously
    echo "   Copying files in parallel..."
    (cd "${src}" && find . -type f -print0) | \
        xargs -0 -P ${parallel_jobs} -I {} \
        rsync -a "${src}/{}" "${dst}/{}"

    echo "✅ Parallel copy completed (${parallel_jobs} parallel jobs)"
}

# ========================
# LLaMA-Factory Training
# ========================

run_llamafactory_training() {
    local config_path=$1
    local is_multinode=${2:-false}
    local node_rank=${3:-0}
    local total_nodes=${4:-1}
    local master_addr=${5:-"127.0.0.1"}
    local master_port=${6:-29500}

    # Get script directory for llamafactory wrapper
    local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    if [ "${is_multinode}" = "true" ]; then
        echo "🚀 Starting LLaMA-Factory MULTI-NODE training..."
        echo "   Node rank: ${node_rank}/${total_nodes}"
        echo "   Master: ${master_addr}:${master_port}"
    else
        echo "🚀 Starting LLaMA-Factory training..."
    fi
    echo "Config: ${config_path}"

    # Set environment variables for training
    export HF_TRUST_REMOTE_CODE="1"

    # Suppress verbose DeepSpeed startup messages
    export DEEPSPEED_LOG_LEVEL="WARNING"
    export DS_BUILD_OPS_VERBOSE="0"
    export DS_BUILD_CPU_ADAM="0"
    export DS_BUILD_FUSED_ADAM="0"

    # Detect number of GPUs
    local NUM_GPUS
    NUM_GPUS="$(detect_num_gpus)"
    echo "🖥️ Detected ${NUM_GPUS} GPUs for training"

    # Use llamafactory-cli (loss_weight support is built into the source code)
    echo "🔧 Using LlamaFactory with built-in loss_weight support (negative gradient training)"

    # Source WandB environment variables from file created by update_prepare.py
    # These need to be set in bash because update_prepare.py's os.environ changes don't persist
    local wandb_env_file="$(dirname "${config_path}")/wandb_env.sh"
    if [ -f "${wandb_env_file}" ]; then
        echo "📊 Loading WandB environment variables from ${wandb_env_file}"
        source "${wandb_env_file}"

        # Force real-time syncing to prevent data loss on timeout/crashes
        export WANDB_MODE=online
        export WANDB_INIT_TIMEOUT=300
        # Reduce batch size for faster uploads
        export WANDB_CONSOLE=wrap

        # Display what was loaded
        if [ -n "${WANDB_RUN_ID}" ]; then
            echo "📊 WandB: Resuming ${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_RUN_NAME} (id: ${WANDB_RUN_ID})"
        else
            echo "📊 WandB: New run ${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_RUN_NAME}"
        fi
        echo "📊 WandB: Real-time syncing enabled (WANDB_MODE=online)"
    else
        echo "⚠️  WandB env file not found: ${wandb_env_file}"
        echo "   WandB logging may not work properly"
    fi

    # Set multi-node environment variables if enabled
    if [ "${is_multinode}" = "true" ]; then
        export NNODES="${total_nodes}"
        export NODE_RANK="${node_rank}"
        export NPROC_PER_NODE="${NUM_GPUS}"
        export MASTER_ADDR="${master_addr}"
        export MASTER_PORT="${master_port}"
        export FORCE_TORCHRUN=1

        echo "🌐 Multi-node configuration:"
        echo "   NNODES=${NNODES}"
        echo "   NODE_RANK=${NODE_RANK}"
        echo "   NPROC_PER_NODE=${NPROC_PER_NODE}"
        echo "   MASTER_ADDR=${MASTER_ADDR}"
        echo "   MASTER_PORT=${MASTER_PORT}"

        # Run with multi-node support with 1-hour timeout
        echo "⏱️  Training timeout: 1 hour (3600 seconds)"
        llamafactory-cli train "${config_path}" &
        local training_pid=$!
    else
        # Run single-node training with multi-GPU support with 1-hour timeout
        echo "🚀 Launching single-node training with ${NUM_GPUS} GPUs"
        echo "⏱️  Training timeout: 1 hour (3600 seconds)"
        FORCE_TORCHRUN=1 NPROC_PER_NODE="${NUM_GPUS}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
            llamafactory-cli train "${config_path}" &
        local training_pid=$!
    fi

    # Monitor training with 1-hour timeout
    local timeout_seconds=3600
    local elapsed=0
    local check_interval=10

    while kill -0 "${training_pid}" 2>/dev/null; do
        sleep ${check_interval}
        elapsed=$((elapsed + check_interval))

        if [ ${elapsed} -ge ${timeout_seconds} ]; then
            echo "⏰ Training timeout reached (${timeout_seconds}s) - initiating graceful shutdown..."

            # Send SIGTERM first to allow graceful shutdown (wandb sync, checkpointing)
            kill -TERM "${training_pid}" 2>/dev/null || true
            pkill -TERM -P "${training_pid}" 2>/dev/null || true

            echo "   Waiting 30s for graceful shutdown (wandb upload, checkpoint save)..."
            local graceful_wait=0
            while kill -0 "${training_pid}" 2>/dev/null && [ ${graceful_wait} -lt 30 ]; do
                sleep 1
                graceful_wait=$((graceful_wait + 1))
            done

            # If still running, force kill
            if kill -0 "${training_pid}" 2>/dev/null; then
                echo "   Graceful shutdown timeout - force killing..."
                kill -KILL "${training_pid}" 2>/dev/null || true
                pkill -KILL -P "${training_pid}" 2>/dev/null || true
                pkill -9 -f "llamafactory.*train" 2>/dev/null || true
                pkill -9 -f "torchrun.*llamafactory" 2>/dev/null || true
            else
                echo "   ✅ Graceful shutdown completed"
            fi

            echo "❌ Training stopped due to timeout"
            return 2  # Return 2 to indicate timeout
        fi
    done

    # Training finished, get exit code
    wait "${training_pid}"
    local exit_code=$?

    if [ ${exit_code} -eq 0 ]; then
        echo "✅ LLaMA-Factory training completed successfully"
        return 0
    else
        echo "❌ LLaMA-Factory training failed with exit code ${exit_code}"
        return 1
    fi
}

# ========================
# Update Atomic Operation
# ========================

update_atomic() {
    local model_to_serve=$1
    local update_config=$2  # "update_online"
    local skip_training=${3:-false}  # Set to true in multinode mode to skip training here
    local num_nodes=${4:-1}  # Total number of nodes

    local actual_config="${update_config}"

    # Create timestamped output directory in checkpoints/
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_checkpoint_dir="checkpoints/model_${timestamp}"

    echo "Step 1: Preparing data for LLaMA-Factory training (${actual_config})..."
    echo "Training output will be saved to: ${HOST_DATA_PATH}/${output_checkpoint_dir}"

    # Check for existing checkpoint
    local checkpoint_name=""
    if [ -d "${MODEL_PATH}" ] && [ "$(ls -A "${MODEL_PATH}")" ]; then
        checkpoint_name=$(basename "${MODEL_PATH}")
        echo "DEBUG: Using checkpoint: ${checkpoint_name}"
    else
        echo "DEBUG: No checkpoint found, using HF model only"
    fi

    # Pass the timestamped checkpoint directory as the output location
    python "${HOST_RUN_PATH}/scripts/update_prepare.py" \
        --config-path "${HOST_RUN_PATH}/scripts/config/main" \
        --config-name "${actual_config}" \
        hydra.job.chdir=False \
        policy_config.base_model="${model_to_serve}" \
        policy_config.checkpoint_path="${checkpoint_name}" \
        algorithm_config.model_output_name="${output_checkpoint_dir}" \
        +algorithm_config.num_nodes="${num_nodes}" \
        save_path="${HOST_DATA_PATH}"

    if [ $? -ne 0 ]; then
        echo "❌ Data preparation failed, exiting..."
        return 1
    fi

    echo "✅ Single-pass data preparation completed."

    # Verify that training files were created successfully
    echo "Step 1.5: Verifying training files exist..."
    local config_path="${HOST_DATA_PATH}/llamafactory_data/train_config.yaml"

    # Check for either filename (depends on whether validation split is used)
    local train_data_path_with_val="${HOST_DATA_PATH}/llamafactory_data/finetune_train.json"
    local train_data_path_no_val="${HOST_DATA_PATH}/llamafactory_data/finetune.json"

    # Wait briefly for filesystem sync (important on networked storage)
    sleep 2

    if [ ! -f "${config_path}" ]; then
        echo "❌ Training config not found at ${config_path}"
        return 1
    fi

    # Check for either training data file format
    if [ -f "${train_data_path_with_val}" ]; then
        train_data_path="${train_data_path_with_val}"
        echo "✅ Found training data (with validation split): ${train_data_path}"
    elif [ -f "${train_data_path_no_val}" ]; then
        train_data_path="${train_data_path_no_val}"
        echo "✅ Found training data (no validation split): ${train_data_path}"
    else
        echo "❌ Training data not found"
        echo "   Expected either: ${train_data_path_with_val}"
        echo "   Or: ${train_data_path_no_val}"
        return 1
    fi

    echo "✅ Verified all training files exist"

    # Skip training if in multinode mode (training happens at Phase 7 with sync)
    if [ "${skip_training}" = "true" ]; then
        echo "⏭️  Skipping training in update_atomic (multinode mode - training at Phase 7)"
        return 0
    fi

    echo "Step 2: Running LLaMA-Factory training..."

    if [ -f "${config_path}" ]; then
        # Run LLaMA-Factory training
        run_llamafactory_training "${config_path}"

        if [ $? -eq 0 ]; then
            echo "Training with LLaMA-Factory completed successfully."
        else
            echo "❌ LLaMA-Factory training failed"
            return 1
        fi
    else
        echo "❌ LLaMA-Factory config not found at ${config_path}"
        return 1
    fi

    echo "Step 3: Finding last checkpoint with optimizer states..."
    local trained_model_dir="${HOST_DATA_PATH}/${output_checkpoint_dir}"
    local last_checkpoint=$(find_last_checkpoint "${trained_model_dir}")

    if [ -n "${last_checkpoint}" ] && [ -d "${last_checkpoint}" ]; then
        echo "✅ Found last checkpoint: ${last_checkpoint}"

        # Check if optimizer states exist (either HuggingFace or DeepSpeed format)
        if [ -f "${last_checkpoint}/optimizer.pt" ]; then
            echo "✅ Checkpoint contains optimizer states (HuggingFace format)"
        elif [ -d "${last_checkpoint}/global_step"* ] 2>/dev/null && ls "${last_checkpoint}"/global_step*/bf16_zero_pp_rank_*_optim_states.pt 1> /dev/null 2>&1; then
            echo "✅ Checkpoint contains optimizer states (DeepSpeed ZeRO format)"
        else
            echo "⚠️ Warning: Checkpoint does not contain optimizer states"
        fi

        echo "Step 4: Updating ${MODEL_PATH} with last checkpoint..."
        # Remove old model.pt/model_offline.pt if it exists
        if [ -d "${MODEL_PATH}" ]; then
            echo "Removing old model at ${MODEL_PATH}..."
            rm -rf "${MODEL_PATH}"
        fi

        # Copy last checkpoint to MODEL_PATH using parallel rsync
        fast_parallel_copy "${last_checkpoint}" "${MODEL_PATH}"
        echo "✅ Model updated: ${MODEL_PATH} now contains checkpoint with optimizer states"
    else
        echo "⚠️ Warning: No checkpoint found in ${trained_model_dir}"
        echo "Looking for final model instead..."

        # If no checkpoint subdirectory, use the trained model directory itself
        if [ -d "${trained_model_dir}" ] && [ "$(ls -A "${trained_model_dir}")" ]; then
            echo "Using final model from ${trained_model_dir}"

            # Remove old model
            if [ -d "${MODEL_PATH}" ]; then
                echo "Removing old model at ${MODEL_PATH}..."
                rm -rf "${MODEL_PATH}"
            fi

            # Copy entire trained model directory using parallel rsync
            fast_parallel_copy "${trained_model_dir}" "${MODEL_PATH}"
            echo "✅ Model updated: ${MODEL_PATH}"
        else
            echo "❌ Error: No model found at ${trained_model_dir}"
            return 1
        fi
    fi

    return 0
}
