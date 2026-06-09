

# Quickstart

This page walks through training a Qwen3-VL-8B-Instruct policy on WebGym
with AsyncWebRL.

## Main run (8 nodes × 8 H100)

Training is launched by running `webgym/train.py` with the reference config:

```bash
python webgym/train.py --config scripts/configs/config_8x8h100_grpo_inst_nokl.yaml
```

`scripts/launch.sh` is a minimal, scheduler-agnostic template around this
command — adapt it to your environment (Slurm / Ray / Kubernetes / local).
For multi-node runs, start your Ray cluster first, then launch on the head
node.

This run needs a WebGym / Omniboxes browser cluster, which is a separate
service (not shipped in this repo); deploy it from
[`microsoft/webgym` (`webgym` branch)](https://github.com/microsoft/webgym/tree/webgym)
per the
[WebGym server quickstart](https://webgym.readthedocs.io/en/latest/server/quickstart_server.html).

Default allocation (see `allocation_mode` in the yaml):
- 48 GPUs (6 nodes) SGLang rollout
- 16 GPUs (2 nodes) FSDP actor

## Adapting the config to your cluster

The single yaml in `scripts/configs/` is the template to copy. Things to
change when porting:

1. `cluster.n_nodes` / `cluster.n_gpus_per_node` — match your hardware.
2. `cluster.fileroot` — a shared filesystem path where AReaL writes
   checkpoints, recover info, and the wandb id file.
3. `allocation_mode` — must satisfy `consumer_batch_size % actor_dp == 0`
   (see `.claude/rules/webgym.md`).
4. `actor.path` and `test_rollout.base_model_checkpoint` — point at a
   pre-downloaded local HF checkpoint of Qwen3-VL-8B-Instruct
   (`update_weights_from_disk` is unreliable with HF Hub paths on
   shared-GPU worker nodes).
5. `train_dataset.path` / `valid_dataset.path` — WebGym task jsonls.
   The path must contain the substring `webgym` (the dataset router in
   `areal/dataset/__init__.py` dispatches by substring).
6. `webgym.cpu_cluster_token_env_var` — name of the env var holding the
   Omniboxes cluster token. Export it (single-quoted!) before launching.
7. `stats_logger.wandb.entity` / `project` — your wandb destination.

## Output

By default, training writes to `${cluster.fileroot}`. This includes:

- Checkpoints under `checkpoints/`
- Recover info under `recover/` (used by auto-resume after time-limit kill)
- Per-step metrics streamed to wandb (set `WANDB_API_KEY` in your env)
- Test rollouts on the WebGym OOD split at the configured cadence

## Resuming

Relaunching with the same `experiment_name` + `trial_name` automatically
resumes from the latest recover checkpoint and reattaches to the original
wandb run (the run id is persisted via `id_suffix: timestamp`). If your
scheduler enforces a wall-clock limit, set `recover.freq_secs` so a checkpoint
is written before each kill.
