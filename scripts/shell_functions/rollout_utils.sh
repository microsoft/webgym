#!/bin/bash
# Rollout utilities for trajectory collection
# Handles train and test rollout operations

# Required global variables:
#   HOST_RUN_PATH - Scripts directory
#   HOST_DATA_PATH - Data directory
#   MODEL_PATH - Model checkpoint path
#   VLLM_PORT - vLLM server port
#   MULTINODE_RANK - (optional) Rank for multinode
#   MULTINODE_NUM_WORKERS - (optional) Total workers
#   RANK_LOAD_WEIGHTS - (optional) Load weights per rank
#   HYDRA_OVERRIDES - (optional) Additional Hydra config overrides from CLI
#   RESUME_CHECKPOINT_PATH - (optional) Path to checkpoint file for resumption

# ========================
# Train Rollout
# ========================

rollout_atomic_train() {
    local model_to_serve=$1
    local rank_suffix=${2:-""}        # Optional rank suffix for multinode mode

    # Kill any process on port 5000 before starting rollout
    # Only do this if:
    # - Not on a single machine (CUDA_VISIBLE_DEVICES not set), OR
    # - On a single machine but we're the master (rank 0)
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] || [ -z "${rank_suffix}" ] || [ "${rank_suffix}" = "_rank_0" ]; then
        kill_port_process 5000
    fi

    # Only set checkpoint if model.pt exists as a directory
    local checkpoint_name=""
    if [ -d "${MODEL_PATH}" ] && [ "$(ls -A "${MODEL_PATH}")" ]; then
        checkpoint_name=$(basename "${MODEL_PATH}")
        echo "DEBUG: Using checkpoint: ${checkpoint_name}"
    else
        echo "DEBUG: No checkpoint found, using HF model only"
    fi

    # Select machine-specific rollout config
    local rollout_config="rollout_train"

    # Build rollout command with optional multinode rank suffix
    local rollout_cmd="python ${HOST_RUN_PATH}/scripts/rollout.py \
        --config-path ${HOST_RUN_PATH}/scripts/config/main \
        --config-name ${rollout_config} \
        save_path=${HOST_DATA_PATH} \
	    data_path=${DATA_PATH} \
        env_config.vllm_server_url=http://localhost:${VLLM_PORT} \
        policy_config.base_model=${model_to_serve} \
        policy_config.checkpoint_path=${checkpoint_name}"

    # Add multinode parameters if in multinode mode
    if [ -n "${rank_suffix}" ]; then
        # Extract rank number from suffix (e.g., "_rank_0" -> 0)
        local rank_num=$(echo "${rank_suffix}" | sed 's/_rank_//')

        rollout_cmd="${rollout_cmd} +env_config.multinode_rank_suffix=${rank_suffix}"
        rollout_cmd="${rollout_cmd} +env_config.multinode_rank=${rank_num}"
        rollout_cmd="${rollout_cmd} +env_config.multinode_total_workers=${MULTINODE_NUM_WORKERS}"

        # Calculate total load weight across all ranks and build weights list
        local total_load_weight=0
        local all_rank_weights=""
        for ((i=0; i<MULTINODE_NUM_WORKERS; i++)); do
            total_load_weight=$((total_load_weight + RANK_LOAD_WEIGHTS[$i]))
            if [ $i -eq 0 ]; then
                all_rank_weights="${RANK_LOAD_WEIGHTS[$i]}"
            else
                all_rank_weights="${all_rank_weights},${RANK_LOAD_WEIGHTS[$i]}"
            fi
        done

        # Pass load weight ratio to rollout.py (let it read base sizes from config and calculate)
        local rank_load_weight=${RANK_LOAD_WEIGHTS[$rank_num]}
        rollout_cmd="${rollout_cmd} +env_config.rank_load_weight=${rank_load_weight}"
        rollout_cmd="${rollout_cmd} +env_config.total_load_weight=${total_load_weight}"
        rollout_cmd="${rollout_cmd} +env_config.all_rank_weights=\\'${all_rank_weights}\\'"

        echo "Collecting training set trajectories (multinode rank ${rank_num}/${MULTINODE_NUM_WORKERS})..."
        echo "  Load weight: ${rank_load_weight}/${total_load_weight}"
    else
        echo "Collecting training set trajectories..."
    fi

    # Add resume checkpoint path if in resume mode
    if [ -n "${RESUME_CHECKPOINT_PATH:-}" ]; then
        rollout_cmd="${rollout_cmd} +env_config.resume_checkpoint_path=${RESUME_CHECKPOINT_PATH}"
        echo "  Resuming from checkpoint: ${RESUME_CHECKPOINT_PATH}"
    fi

    # Append any CLI config overrides
    if [ -n "${HYDRA_OVERRIDES:-}" ]; then
        rollout_cmd="${rollout_cmd} ${HYDRA_OVERRIDES}"
    fi

    # Execute rollout
    eval "${rollout_cmd}"
}

# ========================
# Test Rollout
# ========================

rollout_atomic_test() {
    local model_to_serve=$1
    local rank=${2:-0}                # Rank for test rollout slicing (default 0)
    local total_workers=${3:-1}       # Total workers for test rollout slicing (default 1)

    # Kill any process on port 5000 before starting rollout
    # Only do this if:
    # - Not on a single machine (CUDA_VISIBLE_DEVICES not set), OR
    # - On a single machine but we're the master (rank 0)
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] || [ "${rank}" = "0" ]; then
        kill_port_process 5000
    fi

    # Only set checkpoint if model.pt exists as a directory
    local checkpoint_name=""
    if [ -d "${MODEL_PATH}" ] && [ "$(ls -A "${MODEL_PATH}")" ]; then
        checkpoint_name=$(basename "${MODEL_PATH}")
        echo "DEBUG: Using checkpoint: ${checkpoint_name}"
    else
        echo "DEBUG: No checkpoint found, using HF model only"
    fi

    # Select machine-specific test rollout config
    local test_rollout_config="rollout_test"

    # Build test rollout command
    local test_rollout_cmd="python ${HOST_RUN_PATH}/scripts/rollout.py \
        --config-path ${HOST_RUN_PATH}/scripts/config/main \
        --config-name ${test_rollout_config} \
        save_path=${HOST_DATA_PATH} \
        data_path=${DATA_PATH} \
        env_config.vllm_server_url=http://localhost:${VLLM_PORT} \
        policy_config.base_model=${model_to_serve} \
        policy_config.checkpoint_path=${checkpoint_name}"

    # Add multinode slicing parameters if in multinode mode
    if [ "${total_workers}" -gt 1 ]; then
        # Add rank suffix for test trajectory saving (same as train rollout)
        local rank_suffix="_rank_${rank}"
        test_rollout_cmd="${test_rollout_cmd} +env_config.multinode_rank_suffix=${rank_suffix}"
        test_rollout_cmd="${test_rollout_cmd} +env_config.test_rank=${rank} +env_config.test_total_workers=${total_workers}"
        test_rollout_cmd="${test_rollout_cmd} +env_config.multinode_total_workers=${total_workers}"

        # Calculate total load weight across all ranks and build weights list
        local total_load_weight=0
        local all_rank_weights=""
        for ((i=0; i<total_workers; i++)); do
            total_load_weight=$((total_load_weight + RANK_LOAD_WEIGHTS[$i]))
            if [ $i -eq 0 ]; then
                all_rank_weights="${RANK_LOAD_WEIGHTS[$i]}"
            else
                all_rank_weights="${all_rank_weights},${RANK_LOAD_WEIGHTS[$i]}"
            fi
        done

        # Pass load weight ratio to rollout.py (let it read base sizes from config and calculate)
        local rank_load_weight=${RANK_LOAD_WEIGHTS[$rank]}
        test_rollout_cmd="${test_rollout_cmd} +env_config.rank_load_weight=${rank_load_weight}"
        test_rollout_cmd="${test_rollout_cmd} +env_config.total_load_weight=${total_load_weight}"
        test_rollout_cmd="${test_rollout_cmd} +env_config.all_rank_weights=\\'${all_rank_weights}\\'"

        echo "Collecting test set trajectories (multinode rank=${rank}, total_workers=${total_workers})..."
        echo "  Load weight: ${rank_load_weight}/${total_load_weight}"
    else
        echo "Collecting test set trajectories..."
    fi

    # Add resume checkpoint path if in resume mode
    if [ -n "${RESUME_CHECKPOINT_PATH:-}" ]; then
        test_rollout_cmd="${test_rollout_cmd} +env_config.resume_checkpoint_path=${RESUME_CHECKPOINT_PATH}"
        echo "  Resuming from checkpoint: ${RESUME_CHECKPOINT_PATH}"
    fi

    # Append any CLI config overrides
    if [ -n "${HYDRA_OVERRIDES:-}" ]; then
        test_rollout_cmd="${test_rollout_cmd} ${HYDRA_OVERRIDES}"
    fi

    # Execute test rollout
    eval "${test_rollout_cmd}"
}
