

# Configuration

AsyncWebRL is configured by a single YAML passed to `webgym/train.py`
(`scripts/configs/config_8x8h100_grpo_inst_nokl.yaml` is the paper main run).
Any field can be overridden on the command line, e.g.:

```bash
python3 webgym/train.py --config scripts/configs/config_8x8h100_grpo_inst_nokl.yaml \
  actor.optimizer.lr=1e-5 seed=42
```

The full AReaL config surface is large; below are the groups that actually
matter for a WebGym run. Everything else can keep the template defaults.

## Cluster & allocation

| Field | What it does |
|---|---|
| `cluster.n_nodes` / `cluster.n_gpus_per_node` | Match your hardware. |
| `cluster.fileroot` | Shared filesystem path for checkpoints, recover info, wandb id. |
| `cluster.name_resolve.type` | `ray` for multi-node, `local` for single node. |
| `allocation_mode` | GPU split between rollout and actor, e.g. `sglang[rollout]:d48+fsdp[actor]:d16`. The actor `dp` must divide `consumer_batch_size`. |
| `scheduler.type` | `ray` (multi-node) or `local` (single node). |

## Actor (policy + GRPO loss)

| Field | What it does |
|---|---|
| `actor.path` | **Local** HF checkpoint of Qwen3-VL-8B-Instruct (Hub paths produce garbage on shared-GPU nodes). |
| `actor.optimizer.lr` | Learning rate (`5e-6` in the main run). |
| `actor.actor_loss` | `grpo`. |
| `actor.kl_ctl` | `0.0` for the no-KL run. |
| `actor.step_adv_const_length` | **Paper §3.2 fix ("constT10")**: constant `1/k` step-advantage normalizer instead of per-trajectory `1/\|τ\|`. Set to `10.0`. |
| `actor.behav_imp_weight_cap` | Caps the off-policy importance ratio (`5.0`) to clip rare gradient spikes. |
| `actor.reward_norm.{mean_level,std_level,group_size}` | Group-level reward normalization; `group_size: ${gconfig.n_samples}`. |
| `actor.mb_spec.max_tokens_per_mb` | Must hold a full multi-step trajectory (≥16384; bump for longer horizons). |

## Generation / sampling

| Field | What it does |
|---|---|
| `gconfig.n_samples` | GRPO group size (8). |
| `gconfig.max_new_tokens` / `gconfig.max_tokens` | Per-turn / total token budget. |
| `gconfig.temperature` / `gconfig.top_p` | Sampling (`1.0` / `1.0`). |

## Async rollout

| Field | What it does |
|---|---|
| `rollout.consumer_batch_size` | Trajectories per train step; must be divisible by actor `dp`. |
| `rollout.max_head_offpolicyness` | Staleness bound for async off-policy rollouts. |
| `rollout.max_concurrent_rollouts` / `max_concurrent_per_worker` | Concurrency caps. |

## SGLang (VLM serving)

| Field | What it does |
|---|---|
| `sglang.context_length` | Must cover the multi-step prompt + images. |
| `sglang.mem_fraction_static` | Static KV-cache fraction (`0.75`). |
| `sglang.enable_multimodal` | `true` for the vision agent. |
| `actor.scheduling_spec[0].env_vars.SGLANG_VLM_CACHE_SIZE_MB` | Raise to `8192+`; the 100 MB default overflows on screenshots and yields zero reward. |

## WebGym environment

| Field | What it does |
|---|---|
| `webgym.cpu_cluster_token_env_var` | Name of the env var holding the Omniboxes token (single-quote it before launch). |
| `webgym.train_difficulty_max_steps` / `test_difficulty_max_steps` | Per-difficulty step horizons. |
| `webgym.prompt_version` / `model_type` / `history_window` | Prompt template, model family, screenshot history depth. |
| `webgym.interaction_mode` | `coordinates`. |

## Datasets

| Field | What it does |
|---|---|
| `train_dataset.path` / `valid_dataset.path` | WebGym task jsonls; **path must contain `webgym`** (substring router) and `type: rl`. |
| `train_dataset.batch_size` | Match `rollout.consumer_batch_size`. |

## Logging, save & resume

| Field | What it does |
|---|---|
| `stats_logger.wandb.{project,entity,mode}` | wandb destination; needs `WANDB_API_KEY`. |
| `recover.freq_secs` | Write a recover checkpoint on a timer — essential when the scheduler enforces a wall-clock limit. |
| `saver.freq_*` / `evaluator.freq_*` | Checkpoint and eval cadence. |
| `experiment_name` + `trial_name` | Reusing the same pair auto-resumes and reattaches the wandb run. |

For the complete AReaL config surface (every dataclass and field), see the
upstream reference at <https://inclusionai.github.io/AReaL/>.
