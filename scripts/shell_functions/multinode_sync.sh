#!/bin/bash
# Multi-node synchronization utilities for distributed rollout

# Configuration
SYNC_DIR="${HOST_DATA_PATH}/multinode_flags"

# ========================
# Synchronization Functions
# ========================

init_multinode_sync() {
    local num_workers=$1

    mkdir -p "${SYNC_DIR}"
    # Ensure all nodes can write sync flags regardless of user/permissions
    chmod 777 "${SYNC_DIR}"

    # Store the total number of workers (including master) in a metadata file
    echo "${num_workers}" > "${SYNC_DIR}/total_workers"

    echo "✅ Initialized multinode sync directory: ${SYNC_DIR}"
    echo "   Total nodes (including master): ${num_workers}"
}

save_master_ip() {
    local master_ip=$1
    echo "${master_ip}" > "${SYNC_DIR}/master_ip"
    echo "💾 Saved master IP: ${master_ip}"
}

get_master_ip() {
    if [ -f "${SYNC_DIR}/master_ip" ]; then
        cat "${SYNC_DIR}/master_ip"
    else
        echo ""
    fi
}

wait_for_master_ip() {
    echo "⏳ Waiting for master node to publish IP address..." >&2
    while [ ! -f "${SYNC_DIR}/master_ip" ]; do
        sleep 2
    done
    local master_ip=$(get_master_ip)
    echo "✅ Master IP discovered: ${master_ip}" >&2
    echo "${master_ip}"
}

get_total_workers() {
    if [ -f "${SYNC_DIR}/total_workers" ]; then
        cat "${SYNC_DIR}/total_workers"
    else
        echo "0"
    fi
}

# New phase-based flag functions with descriptive names
create_phase_flag() {
    local rank=$1
    local phase_name=$2
    local flag_file="${SYNC_DIR}/${phase_name}_rank_${rank}"

    echo "$(date +%s)" > "${flag_file}"
    sync
    if [ -f "${flag_file}" ]; then
        cat "${flag_file}" > /dev/null
    fi
    echo "🚩 Node rank ${rank}: Created ${phase_name} flag at ${flag_file}"
}

remove_phase_flag() {
    local rank=$1
    local phase_name=$2
    local flag_file="${SYNC_DIR}/${phase_name}_rank_${rank}"

    if [ -f "${flag_file}" ]; then
        rm -f "${flag_file}"
        echo "🗑️  Node rank ${rank}: Removed ${phase_name} flag"
    fi
}

wait_for_phase() {
    local rank=$1
    local phase_name=$2
    local total_workers=$(get_total_workers)

    echo "⏳ Node rank ${rank}: Waiting for all nodes at phase '${phase_name}'..."

    while true; do
        stat "${SYNC_DIR}" > /dev/null 2>&1
        ls -la "${SYNC_DIR}" > /dev/null 2>&1

        local flag_count=$(ls -1 "${SYNC_DIR}"/${phase_name}_rank_* 2>/dev/null | wc -l)

        if [ "${flag_count}" -eq "${total_workers}" ]; then
            echo "✅ Node rank ${rank}: All ${total_workers} nodes ready at '${phase_name}'!"
            return 0
        fi

        echo "   Node rank ${rank}: ${flag_count}/${total_workers} nodes ready at '${phase_name}'..."
        sleep 5
    done
}

wait_for_master_phase() {
    local rank=$1
    local phase_name=$2
    local master_flag="${SYNC_DIR}/${phase_name}_rank_0"

    echo "⏳ Node rank ${rank}: Waiting for master to complete '${phase_name}'..."

    while [ ! -f "${master_flag}" ]; do
        sleep 2
    done

    echo "✅ Node rank ${rank}: Master completed '${phase_name}'!"
}

cleanup_phase_flags() {
    local phase_name=$1
    echo "🧹 Cleaning up ${phase_name} flags..."
    rm -f "${SYNC_DIR}"/${phase_name}_rank_*
    echo "✅ ${phase_name} flags cleaned up"
}

# Legacy functions for backward compatibility (will be removed)
create_node_flag() {
    local rank=$1
    create_phase_flag "${rank}" "generic_flag"
}

remove_node_flag() {
    local rank=$1
    remove_phase_flag "${rank}" "generic_flag"
}

wait_for_all_nodes() {
    local rank=$1
    wait_for_phase "${rank}" "generic_flag"
}

cleanup_all_flags() {
    echo "🧹 Cleaning up all multinode flags..."
    rm -f "${SYNC_DIR}"/*_rank_*
    echo "✅ All flags cleaned up"
}

aggregate_trajectories() {
    local traj_type=$1  # 'train' or 'test'
    local total_workers=$(get_total_workers)

    echo "📦 Master node: Aggregating ${traj_type} trajectories from ${total_workers} nodes..."

    # Use Python to merge trajectory files for specified type (incremental format)
    python3 << EOF
import torch
import os
import sys
import re

host_data_path = "${HOST_DATA_PATH}"
total_workers = ${total_workers}
traj_type = "${traj_type}"

print(f"\n{'='*60}")
print(f"Aggregating {traj_type} trajectories (incremental format)...")
print(f"{'='*60}")

# Directory containing trajectory iteration files
traj_dir = os.path.join(host_data_path, f"{traj_type}_trajectories")

if not os.path.exists(traj_dir):
    print(f"⚠️  Trajectory directory not found: {traj_dir}")
    print(f"   No {traj_type} trajectories to aggregate (this is OK if no {traj_type} rollout was run)")
    sys.exit(0)

# Find all rank-specific iteration files (both regular and checkpoint files)
# Checkpoint files (.checkpoint extension) are created by checkpoint_utils.py during rollout:
#   - Checkpoint at 50%: saves partial trajectories to .checkpoint file
#   - Grace period at 98%: appends remaining trajectories to .checkpoint file
# After rollout, we merge all rank checkpoint files into one final iteration file.
# Pattern 1: train_trajectories.pt.iteration5_rank_0.checkpoint (checkpoint files - preferred)
# Pattern 2: train_trajectories.pt.iteration5_rank_0 (regular files - fallback)
pattern_checkpoint = re.compile(rf'{traj_type}_trajectories\.pt\.iteration(\d+)_rank_(\d+)\.checkpoint$')
pattern_regular = re.compile(rf'{traj_type}_trajectories\.pt\.iteration(\d+)_rank_(\d+)$')
rank_files = {}  # iteration_num -> {rank -> filepath}

# First pass: find checkpoint files (preferred)
for filename in os.listdir(traj_dir):
    match = pattern_checkpoint.match(filename)
    if match:
        iteration_num = int(match.group(1))
        rank = int(match.group(2))
        filepath = os.path.join(traj_dir, filename)

        if iteration_num not in rank_files:
            rank_files[iteration_num] = {}
        rank_files[iteration_num][rank] = filepath

# Second pass: find regular files (fallback for ranks without checkpoint files)
for filename in os.listdir(traj_dir):
    match = pattern_regular.match(filename)
    if match:
        iteration_num = int(match.group(1))
        rank = int(match.group(2))
        filepath = os.path.join(traj_dir, filename)

        if iteration_num not in rank_files:
            rank_files[iteration_num] = {}
        # Only add if this rank doesn't already have a checkpoint file
        if rank not in rank_files[iteration_num]:
            rank_files[iteration_num][rank] = filepath

if not rank_files:
    print(f"⚠️  No rank-specific iteration files found in {traj_dir}")
    print(f"   This is OK if this is a single-node run or no {traj_type} rollout was performed")
    sys.exit(0)

# Process each iteration
for iteration_num in sorted(rank_files.keys()):
    print(f"\n--- Processing iteration {iteration_num} ---")

    # Check if aggregated file already exists
    output_file = os.path.join(traj_dir, f"{traj_type}_trajectories.pt.iteration{iteration_num}")
    if os.path.exists(output_file):
        print(f"⚠️  Aggregated file already exists: {output_file}")
        print(f"   Skipping aggregation for iteration {iteration_num}")
        continue

    all_trajectories = []
    aggregated_metadata = {}
    rank_files_for_iteration = rank_files[iteration_num]

    # Load trajectories from all ranks for this iteration
    for rank in sorted(rank_files_for_iteration.keys()):
        filepath = rank_files_for_iteration[rank]
        try:
            worker_data = torch.load(filepath, weights_only=False)
            # Extract trajectories from dict format
            worker_trajs = worker_data['trajectories']
            all_trajectories.extend(worker_trajs)

            # Merge metadata from first rank (all ranks should have same config metadata)
            if not aggregated_metadata and 'metadata' in worker_data:
                aggregated_metadata = worker_data['metadata'].copy()

            print(f"✅ Loaded {len(worker_trajs)} {traj_type} trajectories from rank {rank}")
        except Exception as e:
            print(f"❌ Failed to load {traj_type} trajectories from rank {rank}: {e}")
            sys.exit(1)

    # Save aggregated trajectories in dict format
    if all_trajectories:
        import datetime

        # Ensure metadata exists with required fields
        if not aggregated_metadata:
            aggregated_metadata = {}
        aggregated_metadata['iteration'] = iteration_num
        aggregated_metadata['timestamp'] = aggregated_metadata.get('timestamp', datetime.datetime.now().isoformat())
        aggregated_metadata['num_ranks_aggregated'] = len(rank_files_for_iteration)

        save_data = {
            'trajectories': all_trajectories,
            'metadata': aggregated_metadata
        }
        torch.save(save_data, output_file)
        print(f"💾 Saved {len(all_trajectories)} {traj_type} trajectories to iteration {iteration_num}")
    else:
        print(f"⚠️  No {traj_type} trajectories to save for iteration {iteration_num}")

print(f"\n✅ Aggregation complete for {len(rank_files)} iterations")
EOF

    if [ $? -eq 0 ]; then
        # Ensure trajectory files are fully written to disk before proceeding
        sync
        sleep 2  # Additional wait for networked storage propagation
        echo "✅ Master node: ${traj_type} trajectory aggregation complete"

        # Clean up rank-specific files (no longer needed after aggregation)
        # Note: With incremental format, aggregated files preserve history, so no archiving needed
        echo "🧹 Cleaning up ${traj_type} worker rank files..."
        local traj_dir="${HOST_DATA_PATH}/${traj_type}_trajectories"
        if [ -d "${traj_dir}" ]; then
            for rank_file in "${traj_dir}"/${traj_type}_trajectories.pt.iteration*_rank_*; do
                if [ -f "${rank_file}" ]; then
                    rm -f "${rank_file}"
                    echo "   Removed: $(basename ${rank_file})"
                fi
            done
        fi
        echo "✅ Rank file cleanup complete"

        return 0
    else
        echo "❌ Master node: ${traj_type} trajectory aggregation failed"
        return 1
    fi
}

check_node_rank_valid() {
    local rank=$1
    local total_workers=$(get_total_workers)

    if [ "${rank}" -lt 0 ] || [ "${rank}" -ge "${total_workers}" ]; then
        echo "❌ Invalid rank ${rank}. Must be between 0 and $((total_workers - 1))"
        return 1
    fi

    return 0
}
