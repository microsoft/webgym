#!/bin/bash
# Multi-node launcher for 2-node setup
# Usage: bash launch.sh master
#        bash launch.sh worker

set -euo pipefail

# Check if argument is provided
if [ $# -lt 1 ]; then
    echo "Error: Missing node role argument"
    echo ""
    echo "Usage: $0 <master|worker>"
    echo ""
    echo "Examples:"
    echo "  # On master node:"
    echo "  bash $0 master"
    echo ""
    echo "  # On worker node (master IP auto-discovered from shared filesystem):"
    echo "  bash $0 worker"
    exit 1
fi

NODE_ROLE="$1"

# Validate node role
if [ "$NODE_ROLE" != "master" ] && [ "$NODE_ROLE" != "worker" ]; then
    echo "Error: Invalid node role '$NODE_ROLE'"
    echo "Must be either 'master' or 'worker'"
    exit 1
fi

# For worker nodes, master IP will be discovered from shared filesystem

# Build the run command
RUN_CMD="bash run.sh \
  --data-path /data/v-baihao/tasks \
  --log-path /data/v-baihao/logs \
  --rl-phase both \
  --eval-interval 6 \
  --num-nodes 2 \
  --rank-weights \"1,1\""

# Add node role flag and determine output file
if [ "$NODE_ROLE" = "master" ]; then
    RUN_CMD="$RUN_CMD --master"
    RANK_NUM=0
    echo "Launching as MASTER node (rank 0)..."
else
    RUN_CMD="$RUN_CMD --worker 1"
    RANK_NUM=1
    echo "Launching as WORKER node (rank 1)..."
    echo "Master IP: will be discovered from shared filesystem"
fi

# Output file specific to this rank
OUTPUT_FILE="debug_${RANK_NUM}.out"

# Execute the command
echo "Command: $RUN_CMD"
echo "Output file: $OUTPUT_FILE"
echo ""

rm -f "$OUTPUT_FILE"
eval "$RUN_CMD > $OUTPUT_FILE 2>&1"
