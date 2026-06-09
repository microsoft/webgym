"""Training entry point for WebGym web-agent RL with AReaL.

Usage::

    # Fresh start (cleans previous experiment state)
    python webgym/train.py --config scripts/configs/config_8x8h100_grpo_inst_nokl.yaml

    # Resume from checkpoint (continues wandb run)
    python webgym/train.py --config scripts/configs/config_8x8h100_grpo_inst_nokl.yaml --resume
"""

import argparse
import math
import os
import pathlib
import shutil
import subprocess
import sys

import requests

# Must be set before any torch import so all child processes (actor, critic,
# SGLang) inherit it. Reduces CUDA memory fragmentation on shared GPUs.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

sys.path.append(str(pathlib.Path(__file__).parent))
from configs import WebGymConfig

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.logging import getLogger

_logger = getLogger("WebGymTrain")


def query_cluster_capacity(env_config: dict) -> int | None:
    """Query the Omniboxes CPU cluster for total browser instance capacity."""
    host_ip = env_config.get("host_ip", "localhost")
    master_port = env_config.get("master_port", 443)
    api_key = env_config.get("cpu_cluster_token", "")
    if not api_key:
        token_var = env_config.get("cpu_cluster_token_env_var", "CPU_CLUSTER_TOKEN")
        api_key = os.environ.get(token_var, "")

    protocol = "https" if master_port == 443 else "http"
    base_url = f"{protocol}://{host_ip}:{master_port}"
    headers = {"x-api-key": api_key} if api_key else {}

    try:
        resp = requests.get(
            f"{base_url}/info", headers=headers, verify=False, timeout=30
        )
        resp.raise_for_status()
        info = resp.json()
    except Exception:
        _logger.warning("Could not query cluster capacity", exc_info=True)
        return None

    if "nodes" not in info:
        return None

    total = sum(node.get("capacity", 0) for node in info["nodes"])
    n_nodes = len(info["nodes"])
    # Use 80% of capacity to leave headroom for connection stability
    usable = int(total * 0.8)
    _logger.info(
        "Omniboxes cluster: %d nodes, total capacity=%d, usable (80%%)=%d",
        n_nodes,
        total,
        usable,
    )
    return usable


def _infer_experiment_name(config) -> str:
    """Auto-generate experiment name from config settings.

    Example: qwen3vl8b-think-grpo-offpolicy4-kl0.01-prox-batch48-k8-lr5e-06
    """
    parts = []

    # Model name (short form from HF path)
    model_path = getattr(config.actor, "path", "")
    if model_path:
        model_short = model_path.split("/")[-1].lower()
        model_short = model_short.replace("qwen3-vl-", "qwen3vl")
        model_short = model_short.replace("-instruct", "-inst")
        model_short = model_short.replace("-thinking", "-think")
        parts.append(model_short)

    # Algorithm: critic_loss + actor_loss
    critic_loss = getattr(config.actor, "critic_loss", "none")
    actor_loss = getattr(config.actor, "actor_loss", "grpo")
    if critic_loss == "archer" and actor_loss == "fbc":
        parts.append("archer-fbc")
    elif actor_loss == "fbc":
        parts.append("fbc")
    elif critic_loss != "none":
        parts.append(f"{critic_loss}-{actor_loss}")
    else:
        parts.append(actor_loss)

    # Off-policy tolerance
    offpolicy = getattr(config.rollout, "max_head_offpolicyness", 1)
    parts.append(f"offpolicy{offpolicy}")

    # KL penalty
    kl = getattr(config.actor, "kl_ctl", 0.0)
    if kl == 0.0:
        parts.append("nokl")
    else:
        parts.append(f"kl{kl}")

    # Proximal logprob recomputation
    if getattr(config.actor, "recompute_logprob", False):
        parts.append("prox")
    else:
        parts.append("noprox")

    # Batch size (consumer_batch_size = effective train batch)
    batch = getattr(config.rollout, "consumer_batch_size", None)
    if batch is None:
        batch = getattr(config.train_dataset, "batch_size", "?")
    parts.append(f"batch{batch}")

    # n_samples
    k = getattr(config.gconfig, "n_samples", 1)
    parts.append(f"k{k}")

    # Learning rate
    lr = getattr(getattr(config.actor, "optimizer", None), "lr", None)
    if lr is not None:
        parts.append(f"lr{lr}")

    return "-".join(str(p) for p in parts)


def main(args):
    # No --resume flag needed — if a DCP checkpoint exists, it is auto-loaded.
    # Saving is only triggered by test rollout completion.
    remaining = list(args)

    # Always clean stale temp state
    shutil.rmtree("/tmp/areal/experiments", ignore_errors=True)
    shutil.rmtree("/tmp/areal/name_resolve", ignore_errors=True)

    config, _ = load_expr_config(remaining, WebGymConfig)

    # Determine experiment name: "auto" → infer from config, otherwise use as-is.
    if config.experiment_name == "auto":
        _name = _infer_experiment_name(config)
    else:
        _name = config.experiment_name
    for obj in [config, config.rollout, config.actor, config.saver,
                config.recover, config.evaluator, config.stats_logger,
                config.perf_tracer]:
        if hasattr(obj, "experiment_name"):
            obj.experiment_name = _name
    if config.ref is not None and hasattr(config.ref, "experiment_name"):
        config.ref.experiment_name = _name
    if config.critic is not None and hasattr(config.critic, "experiment_name"):
        config.critic.experiment_name = _name

    # Clean rollout trajectories from previous runs on fresh start.
    # Use background subprocess with parallel xargs because NFS rmtree
    # on 10K+ files is extremely slow when done sequentially.
    # Clean old rollout trajectory dumps (not checkpoints) on every start
    for _subdir in ("rollout", "test-rollout"):
        _rollout_dir = os.path.join(
            config.cluster.fileroot,
            "logs/root",
            config.experiment_name,
            config.trial_name,
            _subdir,
        )
        if os.path.isdir(_rollout_dir):
            subprocess.Popen(
                f"find {_rollout_dir} -mindepth 1 -maxdepth 1 -type d"
                f" | xargs -P 16 -I{{}} rm -rf {{}}",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
        processor=processor,
    )

    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = get_custom_dataset(
            split="test",
            dataset_config=config.valid_dataset,
            tokenizer=tokenizer,
            processor=processor,
        )

    # Build env_config from the webgym section of the config
    env_config = {}
    if config.webgym:
        env_config = dict(config.webgym)

    # Build model_config for ContextManager
    model_config = {
        "model_type": env_config.pop("model_type", "qwen3-instruct"),
        "prompt_version": env_config.pop("prompt_version", "complete"),
        "interaction_mode": env_config.get("interaction_mode", "coordinates"),
        "history_window": env_config.pop("history_window", 4),
        "keep_thinking_in_history": env_config.pop("keep_thinking_in_history", False),
    }

    # Pass max_concurrent_rollouts so per-worker limits can be computed
    env_config["max_concurrent_rollouts"] = (
        config.rollout.max_concurrent_rollouts or config.rollout.consumer_batch_size
    )

    # Set screenshot_dir on shared PVC so workers and master can both access screenshots.
    from areal.utils.stats_logger import StatsLogger

    _log_path = StatsLogger.get_log_path(
        experiment_name=config.experiment_name,
        trial_name=config.trial_name,
        fileroot=config.cluster.fileroot,
    )
    env_config["screenshot_dir"] = os.path.join(_log_path, ".screenshots")

    # Build workflow kwargs
    workflow_kwargs = dict(
        reward_fn="areal.reward.webgym.webgym_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        processor=config.tokenizer_path,
        train_difficulty_max_steps=env_config.pop("train_difficulty_max_steps", None),
        test_difficulty_max_steps=env_config.pop("test_difficulty_max_steps", None),
        max_steps_schedule=env_config.pop("max_steps_schedule", None),
        turn_discount=env_config.pop("turn_discount", 1.0),
        env_config=env_config,
        model_config=model_config,
    )

    # Optionally configure the AI evaluator for reward
    if config.openai_config is not None:
        workflow_kwargs["openai_config"] = dict(config.openai_config)

    # Release all existing browser instances before training (clean slate).
    # Only run on the controller process (rank 0) to avoid duplicate calls.
    if int(os.getenv("RANK", "0")) == 0:
        from areal.workflow.webgym_workflow import release_all_instances

        api_key = env_config.get("cpu_cluster_token", "")
        if not api_key:
            token_var = env_config.get("cpu_cluster_token_env_var", "CPU_CLUSTER_TOKEN")
            api_key = os.environ.get(token_var, "")
        release_all_instances(
            host_ip=env_config.get("host_ip", "localhost"),
            master_port=env_config.get("master_port", 443),
            api_key=api_key,
        )

    # Initialize the PixelRefStore broker actor for cross-node pixel sharing.
    # Must be created before training starts so rollout workers and FSDP
    # training workers can find it by name.
    if config.cluster.name_resolve.type == "ray":
        from areal.utils.pixel_store import get_or_create_pixel_store

        get_or_create_pixel_store()

    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)
    # Give eval workers a separate monitor_port range so they don't conflict
    # with training workers. Train uses BASE..BASE+n_train-1, eval uses BASE+100..
    if "env_config" in workflow_kwargs and isinstance(
        workflow_kwargs["env_config"], dict
    ):
        eval_env_config = workflow_kwargs["env_config"].copy()
        base_port = eval_env_config.get("monitor_port", 0)
        if base_port > 0:
            eval_env_config["monitor_port"] = base_port + 100
        eval_workflow_kwargs["env_config"] = eval_env_config

    # Auto-distribute max_concurrent_rollouts between train and test rollout
    # proportionally based on GPU count. The total is queried from the
    # Omniboxes CPU cluster (actual browser instance capacity), falling back
    # to the configured max_concurrent_rollouts if the query fails.
    # Must happen BEFORE PPOTrainer() since it creates the rollout controller.
    test_rollout_enabled = (
        config.test_rollout is not None
        and config.test_rollout.enabled
        and valid_dataset is not None
    )
    if test_rollout_enabled:
        from areal.api.alloc_mode import AllocationMode

        cluster_capacity = query_cluster_capacity(
            workflow_kwargs.get("env_config", {})
        )
        total_concurrent = cluster_capacity or (
            config.rollout.max_concurrent_rollouts
            or config.rollout.consumer_batch_size
        )
        alloc = AllocationMode.from_str(config.allocation_mode)
        train_n_gpus = alloc.gen.data_parallel_size
        test_n_gpus = config.test_rollout.n_gpus
        total_gpus = train_n_gpus + test_n_gpus

        train_share = int(total_concurrent * train_n_gpus / total_gpus)
        test_share = total_concurrent - train_share  # remainder to test

        config.rollout.max_concurrent_rollouts = train_share
        config.test_rollout.max_concurrent_per_gpu = max(
            1, test_share // test_n_gpus
        )

        # Update per-worker browser limits in workflow env_configs
        if isinstance(workflow_kwargs.get("env_config"), dict):
            workflow_kwargs["env_config"]["max_concurrent_rollouts"] = train_share
        if isinstance(eval_workflow_kwargs.get("env_config"), dict):
            eval_workflow_kwargs["env_config"]["max_concurrent_rollouts"] = test_share

        _logger.info(
            "Concurrent rollouts split: total=%d%s, "
            "train=%d (%d GPUs), test=%d (%d GPUs)",
            total_concurrent,
            " (from cluster)" if cluster_capacity else " (from config)",
            train_share,
            train_n_gpus,
            test_share,
            test_n_gpus,
        )

    # GRPO adjustments: config values are in episode units, but
    # StalenessManager counts tasks (each task = n_samples episodes).
    # Convert concurrency limits from episode units → task units.
    n_samples = config.gconfig.n_samples
    if n_samples > 1:
        from areal.api.alloc_mode import AllocationMode

        alloc = AllocationMode.from_str(config.allocation_mode)
        dp_size = alloc.train.data_parallel_size
        lcm = (n_samples * dp_size) // math.gcd(n_samples, dp_size)
        cbs = config.rollout.consumer_batch_size
        if cbs % lcm != 0:
            raise ValueError(
                f"consumer_batch_size={cbs} is not divisible by "
                f"LCM(n_samples={n_samples}, dp_size={dp_size})={lcm}. "
                f"Valid values near {cbs}: {(cbs // lcm) * lcm or lcm}, "
                f"{((cbs // lcm) + 1) * lcm}."
            )

        # Convert concurrency limits: user sets in episode units,
        # StalenessManager needs task units (1 task = n_samples episodes).
        if config.rollout.max_concurrent_per_worker is not None:
            old_pw = config.rollout.max_concurrent_per_worker
            config.rollout.max_concurrent_per_worker = max(
                1, old_pw // n_samples
            )
            _logger.info(
                "GRPO: max_concurrent_per_worker %d episodes → %d tasks "
                "(n_samples=%d)",
                old_pw,
                config.rollout.max_concurrent_per_worker,
                n_samples,
            )
        if config.rollout.max_concurrent_rollouts is not None:
            old_mcr = config.rollout.max_concurrent_rollouts
            config.rollout.max_concurrent_rollouts = max(
                1, old_mcr // n_samples
            )
            _logger.info(
                "GRPO: max_concurrent_rollouts %d episodes → %d tasks "
                "(n_samples=%d)",
                old_mcr,
                config.rollout.max_concurrent_rollouts,
                n_samples,
            )
        # NOTE: consumer_batch_size stays in episode units (120). Training
        # distributes 120 episodes across dp_size=6 → 20 per shard → works.
        # prepare_batch already converts to task units via effective_batch_size.
        # StalenessManager converts CBS to task units at construction time
        # using config.rollout.n_samples.

    # Propagate n_samples to rollout config so StalenessManager can convert
    # consumer_batch_size from episode units to task units internally.
    config.rollout.n_samples = config.gconfig.n_samples

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        # Attach adaptive task sampling tracker to rollout controller
        _webgym_cfg = config.webgym if config.webgym else {}
        adaptive_cfg = _webgym_cfg.get("adaptive_sampling") if isinstance(_webgym_cfg, dict) else None
        if adaptive_cfg and adaptive_cfg.get("enabled", False):
            from areal.utils.task_success_tracker import TaskSuccessTracker

            tracker = TaskSuccessTracker(
                alpha=adaptive_cfg.get("alpha", 0.3),
                min_samples=adaptive_cfg.get("min_samples", 2),
                epsilon=adaptive_cfg.get("epsilon", 0.1),
            )
            if hasattr(trainer.rollout, "_task_success_tracker"):
                trainer.rollout._task_success_tracker = tracker
            else:
                setattr(trainer.rollout, "_task_success_tracker", tracker)
            _logger.info(
                "[adaptive_sampling] enabled: epsilon=%.2f, alpha=%.2f, min_samples=%d",
                tracker.epsilon, tracker._alpha, tracker._min_samples,
            )

            # BERT-based difficulty predictor
            if adaptive_cfg.get("bert_predictor", False):
                import torch as _torch

                from areal.utils.difficulty_predictor import DifficultyPredictor

                # GPU 0 is shared with FSDP rank 0; BERT (~67M, ~270MB fp32)
                # is negligible on a B200. Training runs concurrently with
                # actor.ppo_update in a background thread; the cross-process
                # CUDA contexts overlap on the same device.
                _bert_device = (
                    adaptive_cfg.get("bert_device")
                    or ("cuda:0" if _torch.cuda.is_available() else "cpu")
                )
                predictor = DifficultyPredictor(
                    model_name=adaptive_cfg.get(
                        "bert_model", "distilbert-base-uncased"
                    ),
                    device=_bert_device,
                )
                trainer.rollout._difficulty_predictor = predictor
                # Full dataset for random sampling during task selection
                trainer.rollout._full_dataset = list(train_dataset)
                # Target pass-rate that BERT selector pulls toward
                trainer.rollout._difficulty_target_p = float(
                    adaptive_cfg.get("target_p", 0.3)
                )
                # Store ref on trainer for training trigger
                trainer._difficulty_predictor = predictor
                trainer._difficulty_tracker = tracker
                trainer._bert_min_samples = adaptive_cfg.get("min_training_samples", 20)
                trainer._bert_epochs = adaptive_cfg.get("bert_epochs", 3)
                trainer._bert_lr = adaptive_cfg.get("bert_lr", 2e-5)
                # Try to load saved BERT checkpoint on resume.
                # Prefer recover_info dir (synced with recover handler);
                # fall back to legacy HF-snapshot location.
                from areal.utils.recover import RecoverHandler
                _recover_dir = RecoverHandler.recover_info_path(
                    config.recover.experiment_name,
                    config.recover.trial_name,
                    config.recover.fileroot,
                )
                _recover_bert_path = os.path.join(
                    _recover_dir, "difficulty_predictor.pt"
                )
                _legacy_bert_path = os.path.join(
                    config.cluster.fileroot,
                    "checkpoints", "root",
                    config.experiment_name, config.trial_name,
                    "difficulty_predictor", "predictor.pt",
                )
                if os.path.exists(_recover_bert_path):
                    bert_path = _recover_bert_path
                elif os.path.exists(_legacy_bert_path):
                    bert_path = _legacy_bert_path
                else:
                    bert_path = None
                if bert_path is not None:
                    with open(bert_path, "rb") as f:
                        predictor.load_state_dict_bytes(f.read())
                    _logger.info(
                        "[adaptive_sampling] BERT predictor loaded from %s", bert_path
                    )
                else:
                    _logger.info(
                        "[adaptive_sampling] BERT predictor enabled (fresh): model=%s",
                        adaptive_cfg.get("bert_model", "distilbert-base-uncased"),
                    )

        # Launch test rollout if enabled (runs on dedicated GPUs in background)
        test_rollout_mgr = None
        # Set flag on ALL ranks so the collective save broadcast works
        trainer._test_rollout_enabled = test_rollout_enabled
        trainer._test_rollout = None

        if test_rollout_enabled and trainer.actor.is_data_parallel_head():
            from areal.utils.test_rollout import TestRolloutManager

            test_rollout_mgr = TestRolloutManager(
                config=config.test_rollout,
                eval_dataloader=trainer.valid_dataloader,
                workflow="areal.workflow.webgym_workflow.WebGymWorkflow",
                workflow_kwargs=eval_workflow_kwargs,
                initial_model_path=config.actor.path,
                sglang_config=config.sglang,
                rollout_config=config.rollout,
                allocation_mode=trainer.allocation_mode,
                scheduler=trainer.scheduler,
                experiment_name=config.experiment_name,
                trial_name=config.trial_name,
                fileroot=config.cluster.fileroot,
            )
            trainer._test_rollout = test_rollout_mgr
            test_rollout_mgr.start()

        # Rollout-side filtering (rejected episodes are automatically re-sampled):
        # - FBC: reject reward=0 episodes
        # - GRPO: reject groups with zero variance (all-correct or all-incorrect)
        if config.actor.actor_loss == "fbc":
            rollout_filter = "areal.reward.filters.fbc_accept"
        elif config.gconfig.n_samples > 1:
            rollout_filter = "areal.reward.filters.grpo_accept"
        else:
            rollout_filter = None

        try:
            trainer.train(
                workflow="areal.workflow.webgym_workflow.WebGymWorkflow",
                workflow_kwargs=workflow_kwargs,
                # Disable sync eval when test rollout is active
                eval_workflow=(
                    None
                    if test_rollout_enabled
                    else "areal.workflow.webgym_workflow.WebGymWorkflow"
                ),
                eval_workflow_kwargs=eval_workflow_kwargs,
                dynamic_filter_fn=rollout_filter,
            )
        finally:
            if test_rollout_mgr is not None:
                test_rollout_mgr.stop()


if __name__ == "__main__":
    main(sys.argv[1:])
