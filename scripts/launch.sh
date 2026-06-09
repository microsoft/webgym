#!/bin/bash
# Minimal launch template for AsyncWebRL / WebGym training.
#
# This is a generic, scheduler-agnostic EXAMPLE — not a turnkey script.
# Adapt it to your environment (Slurm / Ray / Kubernetes / local) and fill in
# the placeholder paths in scripts/configs/config_8x8h100_grpo_inst_nokl.yaml.
#
# Prerequisites:
#   * A working environment — see docs/installation.md, or build the Dockerfile.
#   * A running WebGym / Omniboxes browser cluster (a separate service):
#       https://github.com/microsoft/webgym/tree/webgym
#       https://webgym.readthedocs.io/en/latest/server/quickstart_server.html
#   * WebGym task jsonls and a local Qwen3-VL-8B-Instruct checkpoint, with
#     their paths set in the config yaml.
set -euo pipefail

CONFIG=scripts/configs/config_8x8h100_grpo_inst_nokl.yaml

# Token for the WebGym / Omniboxes browser cluster. Export it before launching
# (single-quote the value if it contains shell-special characters).
: "${CPU_CLUSTER_TOKEN:?export CPU_CLUSTER_TOKEN before launching}"

# Multi-node: start your Ray cluster first, then run train.py on the head node.
# Single node: just run train.py directly.
python3 webgym/train.py --config "$CONFIG"
