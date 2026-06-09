from __future__ import annotations

import functools
import gc
import os
import shutil
import threading
import time as _dbg_time_mod
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch.distributed as dist
from datasets import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    InferenceEngineConfig,
    PPOActorConfig,
    PPOConfig,
    PPOCriticConfig,
    SchedulingStrategy,
    SchedulingStrategyType,
    SGLangConfig,
    TrainDatasetConfig,
    ValidDatasetConfig,
    vLLMConfig,
)
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.api.scheduler_api import Scheduler
from areal.api.workflow_api import RolloutWorkflow, WorkflowLike
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.infra import (
    LocalScheduler,
    RayScheduler,
    RolloutController,
    SlurmScheduler,
    current_platform,
)
from areal.utils import logging, perf_tracer, seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.environ import is_single_controller
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.perf_tracer import Category
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

if TYPE_CHECKING:
    from areal.engine.fsdp_engine import FSDPPPOActor, FSDPPPOCritic
    from areal.engine.megatron_engine import MegatronPPOActor, MegatronPPOCritic
    from areal.experimental.engine.archon_engine import ArchonPPOActor, ArchonPPOCritic
    from areal.trainer.ppo.actor import PPOActorController
    from areal.trainer.ppo.critic import PPOCriticController

logger = logging.getLogger("RLTrainer")


def _prepare_archer_critic_targets(
    batch: dict[str, Any], gamma: float
) -> None:
    """Compute ArCHer critic training targets and add them to the batch.

    Creates ``archer_critic_mask`` (one-hot at prompt position) and
    ``archer_returns`` (γ-discounted MC returns) from ``loss_mask``,
    ``episode_id``, and ``rewards``.  These are needed by the critic
    *before* ``compute_advantages`` runs.
    """
    import torch

    from areal.infra.rpc.rtensor import RTensor

    loss_mask = batch["loss_mask"]
    if isinstance(loss_mask, RTensor):
        loss_mask = RTensor.localize(loss_mask)
    loss_mask = loss_mask.float()

    bs = loss_mask.shape[0]
    max_seqlen = loss_mask.shape[1]
    device = loss_mask.device
    batch_indices = torch.arange(bs, device=device)

    prompt_pos = loss_mask.int().argmax(dim=-1)  # [bs]

    # Compute reward_score (same scaling as compute_advantages)
    rewards = batch["rewards"]
    if isinstance(rewards, RTensor):
        rewards = RTensor.localize(rewards)
    reward_score = rewards.float()

    # MC returns per step
    mc_returns = torch.zeros(bs, dtype=torch.float32, device=device)
    episode_ids = batch.get("episode_id")
    if isinstance(episode_ids, RTensor):
        episode_ids = RTensor.localize(episode_ids)

    if episode_ids is not None:
        unique_eps, inverse = torch.unique(episode_ids, return_inverse=True)
        n_episodes = unique_eps.shape[0]

        first_idx = torch.zeros(n_episodes, dtype=torch.long, device=device)
        seen = torch.zeros(n_episodes, dtype=torch.bool, device=device)
        for idx in range(bs):
            ep = inverse[idx]
            if not seen[ep]:
                first_idx[ep] = idx
                seen[ep] = True
        ep_rewards = reward_score[first_idx]

        for ep_idx in range(n_episodes):
            ep_step_mask = inverse == ep_idx
            ep_indices = ep_step_mask.nonzero(as_tuple=True)[0]
            N = ep_indices.shape[0]
            R = ep_rewards[ep_idx].item()
            for i, idx in enumerate(ep_indices):
                mc_returns[idx] = gamma ** (N - 1 - i) * R
    else:
        # No episode_id: treat each row as a single-step episode (γ^0 * R = R)
        mc_returns = reward_score.clone()

    # Write targets into batch
    archer_critic_mask = torch.zeros(
        bs, max_seqlen, dtype=torch.float32, device=device
    )
    archer_critic_mask[batch_indices, prompt_pos] = 1.0
    archer_returns_2d = torch.zeros(
        bs, max_seqlen, dtype=torch.float32, device=device
    )
    archer_returns_2d[batch_indices, prompt_pos] = mc_returns

    batch["archer_critic_mask"] = archer_critic_mask
    batch["archer_returns"] = archer_returns_2d

    logger.info(
        "[ArCHer targets] bs=%d gamma=%.3f mc_returns=%s",
        bs,
        gamma,
        [round(x, 3) for x in mc_returns.tolist()[:8]],
    )


def _truncate_for_critic(batch: dict[str, Any]) -> dict[str, Any]:
    """Truncate sequences to prompt-only for ArCHer critic.

    In ArCHer, the critic only needs V(s) at the prompt position. Due to
    causal masking, response tokens after the prompt position don't affect
    the hidden state there. Removing them saves 30-40% of critic compute.
    """
    import torch

    from areal.infra.rpc.rtensor import RTensor

    archer_mask = batch.get("archer_critic_mask")
    if archer_mask is None:
        return batch

    # Localize archer_mask if it's an RTensor
    archer_mask = (
        RTensor.localize(archer_mask)
        if isinstance(archer_mask, RTensor)
        else archer_mask
    )

    # archer_critic_mask is [bs, max_seqlen] with 1 at exactly one position
    # per sequence (the prompt position). Find the rightmost prompt position
    # across all sequences: trunc_len = max(prompt_pos) + 1.
    per_seq_pos = archer_mask.int().argmax(dim=1)  # [bs]
    trunc_len = int(per_seq_pos.max().item()) + 1

    orig_seqlen = archer_mask.shape[1]
    if trunc_len >= orig_seqlen:
        return batch

    # Build a per-sequence mask to zero out response tokens after each
    # sequence's own prompt_pos.  After truncation to trunc_len, sequences
    # with prompt_pos < trunc_len-1 still have useless response tokens.
    # Zeroing attention_mask at those positions makes the model ignore them;
    # other tensors are also zeroed for cleanliness.
    # keep_mask[i, j] = 1 iff j <= per_seq_pos[i]
    col_idx = torch.arange(trunc_len, device=archer_mask.device).unsqueeze(0)  # [1, trunc_len]
    keep_mask = col_idx <= per_seq_pos.unsqueeze(1)  # [bs, trunc_len] bool

    n_zeroed = int((per_seq_pos < trunc_len - 1).sum().item())
    logger.info(
        "Critic truncation: %d -> %d tokens (saved %.0f%%), "
        "response tokens zeroed for %d/%d seqs",
        orig_seqlen,
        trunc_len,
        (1 - trunc_len / orig_seqlen) * 100,
        n_zeroed,
        per_seq_pos.shape[0],
    )

    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, RTensor):
            v_local = RTensor.localize(v)
        else:
            v_local = v

        if isinstance(v_local, torch.Tensor) and v_local.ndim == 2 and v_local.shape[1] == orig_seqlen:
            truncated = v_local[:, :trunc_len]
            # Zero out response tokens: multiply for float, masked_fill for int
            if truncated.is_floating_point():
                truncated = truncated * keep_mask.to(truncated.dtype)
            else:
                truncated = truncated.masked_fill(~keep_mask, 0)
            out[k] = truncated.contiguous()
        else:
            out[k] = v
    return out


class PPOTrainer:
    def __init__(
        self,
        config: PPOConfig,
        train_dataset: Dataset,
        valid_dataset: Dataset | None = None,
    ):
        rank = int(os.getenv("RANK", "0"))
        if is_single_controller():
            # Set up file logging for controller process
            logging.setup_file_logging(StatsLogger.get_log_path(config.stats_logger))

        self.config = config
        self.processor, self.tokenizer = load_hf_processor_and_tokenizer(
            config.tokenizer_path
        )
        self.scheduler = None
        if is_single_controller():
            self.scheduler = self._init_scheduler()

        # Set seed.
        seeding.set_random_seed(config.seed, key=f"trainer{rank}")

        # Parse allocation mode.
        self.allocation_mode = AllocationMode.from_str(config.allocation_mode)

        # Pre-allocate mixed node layout if applicable (must happen before
        # any workers are created so the unified PG is ready).
        if (
            is_single_controller()
            and self.scheduler is not None
            and isinstance(self.scheduler, RayScheduler)
            and self.allocation_mode.gen_backend in ("sglang", "vllm")
        ):
            from areal.infra.utils.launcher import get_scheduling_spec

            rollout_spec = get_scheduling_spec(config.rollout)
            actor_spec = get_scheduling_spec(config.actor)
            self.scheduler.setup_mixed_layout(
                self.allocation_mode,
                config.cluster.n_nodes,
                config.cluster.n_gpus_per_node,
                rollout_spec,
                actor_spec,
            )

        self._amend_xccl_weight_update_envvar()

        # Create models: actor, critic, etc.
        self.actor = self._create_actor(config.actor)
        self.critic = None
        if config.critic is not None:
            self.critic = self._create_critic(config.critic)
        self.ref = None
        if (
            config.actor.kl_ctl > 0
            and config.ref is not None
            and not config.actor.kl_use_prox
        ):
            self.ref = self._create_actor(config.ref)

        # Create dataloaders
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_dataloader = self._create_dataloader(
            train_dataset,
            dataset_config=self.config.train_dataset,
            rank=self.actor.data_parallel_rank,
            world_size=self.actor.data_parallel_world_size,
        )
        self.valid_dataloader = None
        if self.config.valid_dataset is not None and valid_dataset is not None:
            self.valid_dataloader = self._create_dataloader(
                valid_dataset,
                dataset_config=self.config.valid_dataset,
                rank=self.actor.data_parallel_rank,
                world_size=self.actor.data_parallel_world_size,
            )

        # Initialize inference
        self.rollout = self._init_rollout(config.rollout, is_eval=False)
        # eval_rollout is created lazily; skipped when TestRolloutManager is active
        # (it creates its own workers with the "test-rollout" role)
        self.eval_rollout = None

        # Proxy worker initialization (lazy, for AgentWorkflow support)
        self._proxy_started = False

        ft_spec = FinetuneSpec(
            total_train_epochs=config.total_train_epochs,
            dataset_size=len(self.train_dataloader) * config.train_dataset.batch_size,
            train_batch_size=config.train_dataset.batch_size,
        )

        # Initialize models
        self.parallel_strategy = self.allocation_mode.train
        assert self.parallel_strategy is not None
        engine_init_kwargs = {
            "addr": None,
            "ft_spec": ft_spec,
            "alloc_mode": self.allocation_mode,
        }
        self.actor.initialize(**engine_init_kwargs, role="actor")
        if self.critic is not None:
            self.critic.initialize(**engine_init_kwargs, role="critic")
        if self.ref is not None:
            self.ref.initialize(**engine_init_kwargs, role="ref")

        # Prepare weight update meta and connect to inference engine
        if self.config.actor.weight_update_mode == "disk":
            disk_kwargs = {
                "experiment_name": config.experiment_name,
                "trial_name": config.trial_name,
                "file_root": config.cluster.fileroot,
                "name": "default",
                "clear_checkpoint_after_load": True,
            }
            if config.actor.use_lora:
                disk_kwargs.update(
                    {
                        "use_lora": config.actor.use_lora,
                        "lora_name": config.gconfig.lora_name,
                        "base_model_name": config.actor.path,
                    }
                )
            self.weight_update_meta = WeightUpdateMeta.from_disk(**disk_kwargs)
        elif self.config.actor.weight_update_mode == "xccl":
            # NCCL/XCCL weight update
            if self.allocation_mode.train_backend == "megatron":
                self.weight_update_meta = WeightUpdateMeta.from_megatron_xccl(
                    self.allocation_mode
                )
            else:
                xccl_kwargs = {"allocation_mode": self.allocation_mode}
                if config.actor.use_lora:
                    xccl_kwargs.update(
                        {
                            "use_lora": config.actor.use_lora,
                            "lora_name": config.gconfig.lora_name,
                            "base_model_name": config.actor.path,
                        }
                    )
                self.weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(**xccl_kwargs)
        else:
            raise ValueError(
                f"Invalid weight update mode: {self.config.actor.weight_update_mode}"
            )
        self.actor.connect_engine(self.rollout, self.weight_update_meta)

        # Set up evaluation
        self.evaluator = Evaluator(config.evaluator, ft_spec)

        # Set up save as HF model
        self.saver = Saver(config.saver, ft_spec)
        self.recover_handler = RecoverHandler(config.recover, ft_spec)

        # Set up statistics logging (wandb, tensoboard, etc.)
        self.stats_logger = StatsLogger(config, ft_spec)

        # Set up checkpointing for recover
        engines_to_recover: dict = dict(default=self.actor)
        if self.critic is not None:
            engines_to_recover["critic"] = self.critic
        load_only = getattr(config.recover, "load_only", "all")
        self.recover_info = self.recover_handler.load(
            engines_to_recover,
            self.saver,
            self.evaluator,
            self.stats_logger,
            self.train_dataloader,
            inference_engine=self.rollout,
            weight_update_meta=self.weight_update_meta,
            load_only=load_only,
        )

        self._config_perf_tracer()

    def train(
        self,
        workflow: WorkflowLike,
        eval_workflow: WorkflowLike | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
        eval_workflow_kwargs: dict[str, Any] | None = None,
        dynamic_filter_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        total_epochs: int | None = None,
    ):
        config = self.config
        start_step = (
            self.recover_info.last_step_info.next().global_step
            if self.recover_info is not None
            else 0
        )

        # Cumulative counters for FBC actor gradient tracking
        self._actor_gradient_steps = 0
        self._actor_gradient_samples = 0

        if total_epochs is None:
            total_epochs = config.total_train_epochs
        if total_epochs <= 0:
            raise ValueError(f"Total epochs must be positive: {total_epochs}")
        steps_per_epoch = len(self.train_dataloader)
        max_steps = total_epochs * steps_per_epoch

        # Lazily create eval_rollout if sync eval is needed
        if eval_workflow is not None and self.eval_rollout is None:
            self.eval_rollout = self._init_rollout(config.rollout, is_eval=True)

        # Initialize proxy workers if not using RolloutWorkflow
        if self._requires_proxy_workflow(workflow):
            self._ensure_proxy_started()

        for global_step in range(start_step, max_steps):
            if (
                config.total_train_steps is not None
                and global_step >= config.total_train_steps
            ):
                break
            epoch = global_step // steps_per_epoch
            step = global_step % steps_per_epoch

            with (
                stats_tracker.record_timing("rollout"),
                perf_tracer.trace_scope(
                    "train.rollout",
                    category=Category.COMPUTE,
                    args={
                        "global_step": global_step,
                        "epoch_step": step,
                    },
                ),
            ):
                rollout_batch = self.actor.prepare_batch(
                    self.train_dataloader,
                    workflow=workflow,
                    workflow_kwargs=workflow_kwargs,
                    should_accept_fn=dynamic_filter_fn,
                    group_size=config.gconfig.n_samples,
                    dynamic_bs=self.config.dynamic_bs,
                )

            import time as _dbg_time

            # Images are already in PixelRefStore (put by rollout workers).
            # Just record the keys for FSDP workers to fetch, and remove
            # the key list from the batch dict.
            _pixel_keys = rollout_batch.pop("_image_store_keys", None)
            if _pixel_keys is not None:
                import json as _json

                rollout_batch["_image_store_keys_json"] = _json.dumps(_pixel_keys)

                # Pre-check: PixelRefStore is a Ray actor with max_restarts=-1
                # and loses its in-memory dict on every restart. If the actor
                # was restarted between rollout and this train step (e.g.
                # transient worker-overlay disk-full crash), get_slice would
                # return empty for these keys and FSDP _prepare_mb would
                # KeyError mid-microbatch — desyncing the FSDP collective and
                # hanging NCCL. Drop the whole batch up-front from the
                # controller so all ranks deterministically skip this step.
                try:
                    import ray as _ray

                    from areal.utils.pixel_store import (
                        get_or_create_pixel_store as _gops,
                    )

                    _store = _gops()
                    _missing = _ray.get(
                        _store.count_missing.remote(_pixel_keys), timeout=30
                    )
                except Exception:
                    logger.warning(
                        "[pixel_keys_check] failed to query PixelRefStore; "
                        "assuming healthy and continuing",
                        exc_info=True,
                    )
                    _missing = 0
                if _missing > 0:
                    logger.warning(
                        "[skip step] %d/%d pixel keys missing from "
                        "PixelRefStore (likely restarted) — dropping batch "
                        "step=%d",
                        _missing, len(_pixel_keys), global_step,
                    )
                    self._cleanup_step(rollout_batch, global_step=global_step)
                    self._export_and_commit_stats(
                        epoch=epoch, epoch_step=step, global_step=global_step
                    )
                    self._maybe_periodic_save(config, epoch, step, global_step)
                    self.rollout.set_version(global_step + 1)
                    continue

            # ── Phase 1: Critic training (before actor, on FULL data) ──
            _skip_actor_update = False

            if self.critic is not None and config.actor.critic_loss == "ppo":
                # PPO critic: compute values first (needed for GAE)
                with (
                    stats_tracker.record_timing("critic_values"),
                    perf_tracer.trace_scope(
                        "train.compute_values",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    rollout_batch["values"] = self.critic.compute_values(rollout_batch)
                    self.critic.get_device_stats().log("critic values")

            if self.critic is not None and config.actor.critic_loss == "archer":
                # ArCHer critic: train on full (unfiltered) data, then
                # recompute values. Critic doesn't need logp so we can
                # do this before any FBC filtering.

                # Prepare MC return targets for critic training
                _prepare_archer_critic_targets(
                    rollout_batch, gamma=config.actor.archer_gamma
                )
                critic_batch = _truncate_for_critic(rollout_batch)

                with (
                    stats_tracker.record_timing("critic_train_step"),
                    perf_tracer.trace_scope(
                        "train.critic_ppo_update",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    self.critic.ppo_update(critic_batch)
                    self.critic.step_lr_scheduler()
                    self.critic.get_device_stats().log("ppo critic update")

                # Recompute values with updated critic
                with perf_tracer.trace_scope(
                    "train.recompute_values",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ):
                    import torch
                    import torch.nn.functional as F

                    from areal.infra.rpc.rtensor import RTensor

                    trunc_values = self.critic.compute_values(critic_batch)
                    trunc_values = RTensor.localize(trunc_values)
                    am = rollout_batch["attention_mask"]
                    orig_seqlen = (
                        RTensor.localize(am).shape[1]
                        if isinstance(am, RTensor)
                        else am.shape[1]
                    )
                    trunc_seqlen = critic_batch["attention_mask"].shape[1]
                    if trunc_seqlen < orig_seqlen:
                        rollout_batch["values"] = F.pad(
                            trunc_values,
                            (0, orig_seqlen - trunc_seqlen),
                        )
                    else:
                        rollout_batch["values"] = trunc_values

                del critic_batch, trunc_values
                gc.collect()

            # FBC filtering is now done at rollout side via should_accept_fn
            # (reward=0 episodes are rejected and re-sampled automatically).
            # The trainer always receives a full batch of reward>0 episodes.

            if _skip_actor_update:
                logger.info("ppo_update=0.00s step=%d (skipped, no reward>0)", global_step)
                self._cleanup_step(rollout_batch, global_step=global_step)
                self._export_and_commit_stats(
                    epoch=epoch, epoch_step=step, global_step=global_step
                )
                self._maybe_periodic_save(config, epoch, step, global_step)
                # Bump version so staleness manager allows new rollout tasks
                self.rollout.set_version(global_step + 1)
                continue

            # ── Phase 3: Actor forward passes (on possibly filtered data) ──
            if config.actor.should_compute_prox_logp():
                with (
                    stats_tracker.record_timing("recompute_logp"),
                    perf_tracer.trace_scope(
                        "train.recompute_logp",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    rollout_batch["prox_logp"] = self.actor.compute_logp(rollout_batch)
                    self.actor.get_device_stats().log("recompute logp")

            if self.ref is not None:
                with (
                    stats_tracker.record_timing("ref_logp"),
                    perf_tracer.trace_scope(
                        "train.ref_logp",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    rollout_batch["ref_logp"] = self.ref.compute_logp(rollout_batch)
                    self.ref.get_device_stats().log("ref logp")

            # ── Phase 4: Compute advantages ──
            with (
                stats_tracker.record_timing("compute_advantage"),
                perf_tracer.trace_scope(
                    "train.compute_advantage",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                adv_batch = self.actor.compute_advantages(rollout_batch)
                self.actor.get_device_stats().log("compute advantages")

            # ── Phase 5: ArCHer+FBC second filter (step_adv > 1/H) ──
            if config.actor.critic_loss == "archer" and config.actor.actor_loss == "fbc":
                import torch

                from areal.infra.rpc.rtensor import RTensor

                advantages = adv_batch["advantages"]
                if isinstance(advantages, RTensor):
                    advantages = RTensor.localize(advantages)
                loss_mask = adv_batch["loss_mask"]
                if isinstance(loss_mask, RTensor):
                    loss_mask = RTensor.localize(loss_mask)

                # Extract per-step scalar advantage
                prompt_pos = loss_mask.int().argmax(dim=-1)
                batch_idx = torch.arange(advantages.shape[0], device=advantages.device)
                step_adv = advantages[batch_idx, prompt_pos]

                # Compute horizon: use max_steps from rollout config (not actual steps)
                horizon_field = adv_batch.get("horizon")
                if horizon_field is not None:
                    if isinstance(horizon_field, RTensor):
                        horizon_field = RTensor.localize(horizon_field)
                    horizon = horizon_field.float()
                else:
                    # Fallback: count actual steps per episode
                    episode_ids = adv_batch.get("episode_id")
                    if episode_ids is not None:
                        if isinstance(episode_ids, RTensor):
                            episode_ids = RTensor.localize(episode_ids)
                        unique_eps, inverse = torch.unique(episode_ids, return_inverse=True)
                        horizon = torch.zeros(advantages.shape[0], device=advantages.device)
                        for ep_idx in range(unique_eps.shape[0]):
                            ep_mask = inverse == ep_idx
                            horizon[ep_mask] = ep_mask.sum().float()
                    else:
                        horizon = torch.ones(advantages.shape[0], device=advantages.device)

                adv_threshold = 1.0 / horizon
                keep_mask = step_adv > adv_threshold
                n_keep = int(keep_mask.sum().item())
                n_total = keep_mask.shape[0]
                logger.info(
                    "[archer+fbc] Second filter: %d/%d steps pass "
                    "(step_adv > 1/H), adv=%s, threshold=%s",
                    n_keep, n_total,
                    [round(x, 3) for x in step_adv.tolist()[:8]],
                    [round(x, 3) for x in adv_threshold.tolist()[:8]],
                )

                dp_size = self.allocation_mode.train.dp_size
                min_batch = dp_size * config.actor.ppo_n_minibatches
                if n_keep < min_batch:
                    logger.warning(
                        "[archer+fbc] Only %d steps survive (need >= %d = dp%d * mb%d), "
                        "skipping update",
                        n_keep, min_batch, dp_size, config.actor.ppo_n_minibatches,
                    )
                    logger.info("ppo_update=0.00s step=%d (skipped, too few steps)", global_step)
                    self._cleanup_step(rollout_batch, adv_batch, global_step=global_step)
                    self._export_and_commit_stats(
                        epoch=epoch, epoch_step=step, global_step=global_step
                    )
                    # Periodic save even when actor is skipped (critic still trains)
                    self._maybe_periodic_save(
                        config, epoch, step, global_step,
                    )
                    # Bump version so staleness manager allows new rollout tasks
                    self.rollout.set_version(global_step + 1)
                    continue
                else:
                    keep_indices = keep_mask.nonzero(as_tuple=True)[0].tolist()
                    for key in list(adv_batch.keys()):
                        v = adv_batch[key]
                        if isinstance(v, RTensor):
                            v = RTensor.localize(v)
                        if isinstance(v, torch.Tensor) and v.shape[0] == n_total:
                            adv_batch[key] = v[keep_mask]
                        elif isinstance(v, list) and len(v) == n_total:
                            adv_batch[key] = [v[i] for i in keep_indices]

                    # FBC: all surviving steps are equally good → advantage = 1
                    loss_mask_filtered = adv_batch["loss_mask"]
                    if isinstance(loss_mask_filtered, RTensor):
                        loss_mask_filtered = RTensor.localize(loss_mask_filtered)
                    adv_batch["advantages"] = loss_mask_filtered.float()

            # Wait for async checkpoint staging to complete before modifying parameters
            self.saver.maybe_wait_for_staging()

            # Launch BERT difficulty-predictor training in a background thread
            # so it runs concurrently with actor.ppo_update. BERT lives on the
            # trainer-master process (cuda:0); ppo_update is RPC'd to FSDP
            # workers, so the master is mostly idle on Python while BERT
            # trains. We join before update_weights so the published BERT
            # weights are the latest.
            _bert_thread, _bert_box = self._maybe_launch_bert_train()

            # ── Phase 6: Actor update ──
            if not _skip_actor_update:
                with (
                    stats_tracker.record_timing("train_step"),
                    perf_tracer.trace_scope(
                        "train.ppo_update",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    self.actor.ppo_update(adv_batch)
                    self.actor.step_lr_scheduler()
                    self.actor.get_device_stats().log("ppo update")
                    # Track actor gradient samples
                    _n_samples = adv_batch["loss_mask"].shape[0] if "loss_mask" in adv_batch else 0
                    self._actor_gradient_steps += 1
                    self._actor_gradient_samples += _n_samples

            # ── Phase 7: PPO critic update (after actor, needs returns) ──
            if self.critic is not None and config.actor.critic_loss == "ppo":
                with (
                    stats_tracker.record_timing("critic_train_step"),
                    perf_tracer.trace_scope(
                        "train.critic_ppo_update",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    self.critic.ppo_update(adv_batch)
                    self.critic.step_lr_scheduler()
                    self.critic.get_device_stats().log("ppo critic update")

            # Images are in PixelRefStore actor; freed in clear_batches below.

            # Wait for the background BERT training thread (launched before
            # Phase 6) so the new predictor weights are in place before we
            # pause rollout and push actor weights to SGLang.
            self._maybe_join_bert_train(_bert_thread, _bert_box)

            # pause inference for updating weights, save, and evaluation
            self.rollout.pause()

            with (
                stats_tracker.record_timing("update_weights"),
                perf_tracer.trace_scope(
                    "train.update_weights",
                    category=Category.COMM,
                    args={"global_step": global_step},
                ),
            ):
                self.actor.update_weights(self.weight_update_meta)

                self.actor.set_version(global_step + 1)
                if self.critic is not None:
                    self.critic.set_version(global_step + 1)
                self.rollout.set_version(global_step + 1)
                if self.eval_rollout is not None:
                    self.eval_rollout.set_version(global_step + 1)

            with (
                stats_tracker.record_timing("eval"),
                perf_tracer.trace_scope(
                    "train.eval",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                self._evaluate(
                    eval_workflow=eval_workflow,
                    eval_workflow_kwargs=eval_workflow_kwargs,
                    epoch=epoch,
                    epoch_step=step,
                    global_step=global_step,
                )

            # Save checkpoints: test rollout trigger OR periodic timer
            _test_rollout = getattr(self, "_test_rollout", None)
            _should_save = False
            _save_reason = ""
            if _test_rollout is not None and _test_rollout.should_save():
                _should_save = True
                _save_reason = "test rollout"
            # Time-based save when test rollout is disabled
            if not _should_save and _test_rollout is None:
                _recover_freq = getattr(config.recover, "freq_secs", None)
                if _recover_freq is not None:
                    _now = _dbg_time.monotonic()
                    _last_save = getattr(self, "_last_periodic_save_time", 0.0)
                    if _now - _last_save >= _recover_freq:
                        _should_save = True
                        _save_reason = f"periodic ({_recover_freq}s)"
                        self._last_periodic_save_time = _now
            if _should_save:
                logger.info(
                    "Saving HF + DCP checkpoints (%s, step=%d)",
                    _save_reason, global_step,
                )
                with (
                    stats_tracker.record_timing("save"),
                    perf_tracer.trace_scope(
                        "train.save",
                        category=Category.IO,
                        args={"global_step": global_step},
                    ),
                ):
                    self._save_hf(
                        epoch=epoch, epoch_step=step, global_step=global_step
                    )

                with (
                    stats_tracker.record_timing("checkpoint_for_recover"),
                    perf_tracer.trace_scope(
                        "train.checkpoint",
                        category=Category.IO,
                        args={"global_step": global_step},
                    ),
                ):
                    self._save_recover_checkpoint(
                        epoch=epoch, epoch_step=step, global_step=global_step
                    )

            with (
                stats_tracker.record_timing("clear_batches"),
                perf_tracer.trace_scope(
                    "train.clear_batches",
                    category=Category.INSTR,
                    args={"global_step": global_step},
                ),
            ):
                self._cleanup_step(rollout_batch, adv_batch, global_step=global_step)

            with perf_tracer.trace_scope(
                "train.log_stats",
                category=Category.INSTR,
                args={"global_step": global_step},
            ):
                self._export_and_commit_stats(
                    epoch=epoch, epoch_step=step, global_step=global_step
                )

            # Force GC to free any lingering CPU tensors from this step
            gc.collect()

            # Resume rollout
            self.rollout.resume()

            self._save_perf_tracer(step=global_step)

    def close(self):
        self.saver.finalize()
        self.stats_logger.close()
        if self.eval_rollout is not None:
            self.eval_rollout.destroy()
        self.rollout.destroy()
        if self.ref is not None:
            self.ref.destroy()
        if self.critic is not None:
            self.critic.destroy()
        self.actor.destroy()
        perf_tracer.save(force=True)

    def _config_perf_tracer(self):
        rank = int(os.getenv("RANK", "0"))
        if self.config.perf_tracer is None:
            return
        perf_tracer.configure(self.config.perf_tracer, rank=rank, role="master")

        if not is_single_controller():
            return

        self.actor.config_perf_tracer(self.config.perf_tracer, role="actor")
        if self.critic is not None:
            self.critic.config_perf_tracer(self.config.perf_tracer, role="critic")
        if self.ref is not None:
            self.ref.config_perf_tracer(self.config.perf_tracer, role="ref")
        self.rollout.config_perf_tracer(self.config.perf_tracer, role="rollout")
        if self.eval_rollout is not None:
            self.eval_rollout.config_perf_tracer(
                self.config.perf_tracer, role="test-rollout"
            )

    def _save_perf_tracer(self, step: int):
        self.actor.save_perf_tracer(step=step)
        if self.ref is not None:
            self.ref.save_perf_tracer(step=step)
        if self.critic is not None:
            self.critic.save_perf_tracer(step=step)
        if self.eval_rollout is not None:
            self.eval_rollout.save_perf_tracer(step=step)
        self.rollout.save_perf_tracer(step=step)
        perf_tracer.save(step=step)

    def _init_scheduler(self) -> Scheduler:
        cfg = self.config.scheduler
        if cfg.type == "local":
            return LocalScheduler(exp_config=self.config)
        elif cfg.type == "ray":
            return RayScheduler(exp_config=self.config)
        elif cfg.type == "slurm":
            return SlurmScheduler(exp_config=self.config)
        raise NotImplementedError(f"Unknown scheduler type: {cfg.type}")

    def _create_dataloader(
        self,
        dataset: Dataset,
        dataset_config: TrainDatasetConfig | ValidDatasetConfig,
        rank: int,
        world_size: int,
    ) -> StatefulDataLoader:
        return create_dataloader(
            dataset,
            rank=rank,
            world_size=world_size,
            dataset_config=dataset_config,
        )

    def _amend_xccl_weight_update_envvar(self):
        if not is_single_controller():
            # These environs are set by the launcher in the SPMD mode.
            return
        if self.allocation_mode.gen_backend != "sglang":
            return

        # Disable some environ for NCCL weight update.
        for spec in self.config.actor.scheduling_spec:
            spec.env_vars["NCCL_CUMEM_ENABLE"] = "0"
            spec.env_vars["NCCL_NVLS_ENABLE"] = "0"

    def _maybe_launch_bert_train(
        self,
    ) -> tuple[threading.Thread | None, dict[str, Any] | None]:
        """Launch BERT difficulty-predictor training in a background thread.

        Returns (thread, stats_box). Both are None when the predictor or
        tracker is not attached, or when the dataset is below the minimum.

        Threaded so it can overlap with actor.ppo_update on the trainer
        master process. The DifficultyPredictor.train_on_data takes its own
        lock, serializing against concurrent predict() calls from the
        rollout controller's task_input_generator.
        """
        _dp = getattr(self, "_difficulty_predictor", None)
        _dt = getattr(self, "_difficulty_tracker", None)
        if _dp is None or _dt is None:
            return None, None

        texts, targets = _dt.get_training_data()
        _min = getattr(self, "_bert_min_samples", 20)
        logger.info(
            "[difficulty_predictor] training data: %d samples (min=%d)",
            len(texts), _min,
        )
        if len(texts) < _min:
            return None, None

        epochs = getattr(self, "_bert_epochs", 3)
        lr = getattr(self, "_bert_lr", 2e-5)
        stats_box: dict[str, Any] = {}

        def _train() -> None:
            t0 = _dbg_time_mod.monotonic()
            try:
                stats_box["stats"] = _dp.train_on_data(
                    texts, targets, epochs=epochs, lr=lr
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("[bert_train] failed: %s", exc)
                stats_box["error"] = exc
            stats_box["elapsed"] = _dbg_time_mod.monotonic() - t0

        thread = threading.Thread(target=_train, name="bert-train", daemon=True)
        thread.start()
        return thread, stats_box

    def _maybe_join_bert_train(
        self,
        thread: threading.Thread | None,
        stats_box: dict[str, Any] | None,
    ) -> None:
        """Wait for BERT training thread, publish stats to wandb."""
        # Always publish task-selection stats (available even when BERT
        # itself didn't train this step).
        _dp = getattr(self, "_difficulty_predictor", None)
        if _dp is not None:
            _sel_stats = getattr(_dp, "last_selection_stats", None)
            if _sel_stats:
                stats_tracker.get("master").scalar(
                    **{
                        "adaptive_sampling/difficulty_avg": _sel_stats[
                            "difficulty_avg"
                        ],
                        "adaptive_sampling/difficulty_min": _sel_stats[
                            "difficulty_min"
                        ],
                        "adaptive_sampling/difficulty_max": _sel_stats[
                            "difficulty_max"
                        ],
                        "adaptive_sampling/p_hat_avg": _sel_stats["p_hat_avg"],
                        "adaptive_sampling/in_zone": _sel_stats["in_zone"],
                    }
                )

        if thread is None or stats_box is None:
            return

        thread.join()

        if "stats" in stats_box:
            _stats = stats_box["stats"]
            stats_tracker.get("master").scalar(
                **{
                    "difficulty_predictor/mse": _stats["mse"],
                    "difficulty_predictor/n_samples": _stats["n_samples"],
                    "difficulty_predictor/train_time": stats_box.get(
                        "elapsed", 0.0
                    ),
                }
            )

    def _create_actor(
        self, actor_config: PPOActorConfig
    ) -> FSDPPPOActor | MegatronPPOActor | ArchonPPOActor | PPOActorController:
        if self.allocation_mode.train_backend == "fsdp":
            from areal.engine.fsdp_engine import FSDPPPOActor

            actor_cls = FSDPPPOActor
        elif self.allocation_mode.train_backend == "megatron":
            from areal.engine.megatron_engine import MegatronPPOActor

            actor_cls = MegatronPPOActor
        elif self.allocation_mode.train_backend == "archon":
            from areal.experimental.engine.archon_engine import ArchonPPOActor

            actor_cls = ArchonPPOActor
        else:
            raise ValueError(
                f"Invalid backend: {self.allocation_mode.train_backend}, expected fsdp, megatron or archon"
            )
        if is_single_controller():
            actor = actor_cls.as_controller(actor_config, self.scheduler)
        else:
            actor = actor_cls(config=actor_config)
        actor.create_process_group(parallel_strategy=self.allocation_mode.train)
        return actor

    def _create_critic(
        self, critic_config: PPOCriticConfig
    ) -> FSDPPPOCritic | MegatronPPOCritic | ArchonPPOCritic | PPOCriticController:
        if self.allocation_mode.train_backend == "fsdp":
            from areal.engine.fsdp_engine import FSDPPPOCritic

            critic_cls = FSDPPPOCritic
        elif self.allocation_mode.train_backend == "megatron":
            from areal.engine.megatron_engine import MegatronPPOCritic

            critic_cls = MegatronPPOCritic
        elif self.allocation_mode.train_backend == "archon":
            from areal.experimental.engine.archon_engine import ArchonPPOCritic

            critic_cls = ArchonPPOCritic
        else:
            raise ValueError(
                f"Invalid backend: {self.allocation_mode.train_backend}, expected fsdp, megatron or archon"
            )
        if is_single_controller():
            critic = critic_cls.as_controller(critic_config, self.scheduler)
        else:
            critic = critic_cls(config=critic_config)
        critic.create_process_group(parallel_strategy=self.allocation_mode.train)
        return critic

    def _init_rollout(
        self, rollout_config: InferenceEngineConfig, is_eval: bool = False
    ) -> InferenceEngine | RolloutController:
        # Create a working copy of config
        config = deepcopy(rollout_config)
        if is_eval:
            # NOTE: eval does not have any offpolicyness control
            config.max_head_offpolicyness = int(1e12)
            # eval-rollout uses the same inference servers as rollout
            config.scheduling_strategy = SchedulingStrategy(
                type=SchedulingStrategyType.colocation, target="rollout"
            )
            for spec in config.scheduling_spec:
                spec.gpu = 0

        # Determine engine class and server args based on backend
        if self.allocation_mode.gen_backend == "sglang":
            engine_cls = RemoteSGLangEngine
            server_args = SGLangConfig.build_args(
                sglang_config=self.config.sglang,
                tp_size=self.allocation_mode.gen.tp_size,
                base_gpu_id=0,
            )
        elif self.allocation_mode.gen_backend == "vllm":
            engine_cls = RemotevLLMEngine
            server_args = vLLMConfig.build_args(
                vllm_config=self.config.vllm,
                tp_size=self.allocation_mode.gen.tp_size,
                pp_size=self.allocation_mode.gen.pp_size,
            )
        else:
            raise ValueError(
                f"Invalid backend: {self.allocation_mode.gen_backend}, expected sglang or vllm"
            )

        if not is_single_controller():
            engine = engine_cls(config)
            engine.initialize(
                train_data_parallel_size=self.allocation_mode.train.dp_size
            )
            return engine

        # Single-controller mode - no engine instantiation needed
        controller = engine_cls.as_controller(config, self.scheduler)
        init_kwargs = dict(
            role="rollout",
            alloc_mode=self.allocation_mode,
            server_args=server_args,
        )
        if is_eval:
            assert len(self.rollout.server_infos) > 0
            init_kwargs["server_infos"] = self.rollout.server_infos
            init_kwargs["role"] = "test-rollout"
        controller.initialize(**init_kwargs)

        # After SGLang servers start with HF Hub model, reload weights from
        # a local checkpoint if available. SGLang loads HF Hub paths correctly
        # for tokenizer/processor, but produces garbage model output on
        # shared-GPU worker nodes. Reloading from a local checkpoint fixes this.
        if self.allocation_mode.gen_backend == "sglang" and not is_eval:
            local_ckpt = self._resolve_local_checkpoint()
            if local_ckpt is not None:
                from areal.api.io_struct import WeightUpdateMeta

                meta = WeightUpdateMeta(
                    type="disk",
                    path=local_ckpt,
                    clear_checkpoint_after_load=False,
                    skip_name_resolve=True,
                )
                controller._collective_rpc("update_weights_from_disk", meta=meta)
                logger.info(
                    "[SGLang] Train rollout loaded weights from: %s", local_ckpt
                )

        return controller

    def _resolve_local_checkpoint(self) -> str | None:
        """Find a local HF checkpoint for SGLang weight reload.

        Returns the path to a local checkpoint, or None if not found.
        Priority: training checkpoint > base_model_checkpoint.
        """
        # 1. Check if a training checkpoint already exists
        try:
            ckpt_root = os.path.join(
                self.config.cluster.fileroot,
                "checkpoints/root",
                self.config.experiment_name,
                self.config.trial_name,
                "default",
            )
            if os.path.isdir(ckpt_root):
                candidates = []
                for d in os.listdir(ckpt_root):
                    full = os.path.join(ckpt_root, d)
                    if os.path.isdir(full) and os.path.exists(
                        os.path.join(full, ".save_complete")
                    ):
                        candidates.append(full)
                if candidates:
                    latest = max(candidates, key=os.path.getmtime)
                    logger.info(
                        "[SGLang] Found training checkpoint: %s", latest
                    )
                    return latest
        except Exception:
            pass

        # 2. Check base_model_checkpoint from test_rollout config
        test_rollout_cfg = getattr(self.config, "test_rollout", None)
        if test_rollout_cfg is not None:
            base_ckpt = getattr(test_rollout_cfg, "base_model_checkpoint", None)
            if base_ckpt and os.path.isdir(base_ckpt):
                logger.info(
                    "[SGLang] Found base_model_checkpoint: %s", base_ckpt
                )
                return base_ckpt

        logger.warning(
            "[SGLang] No local checkpoint found. SGLang may produce garbage "
            "output on shared-GPU worker nodes with HF Hub model paths."
        )
        return None

    def _cleanup_step(
        self,
        rollout_batch: dict,
        adv_batch: dict | None = None,
        global_step: int | None = None,
    ):
        """Free RTensor, pixel cache, and PixelRefStore resources for a step.

        Must be called before any ``continue`` that skips the normal
        end-of-step cleanup, otherwise the PixelRefStore fills up and
        train rollout stalls.
        """
        import gc

        self.actor.clear_batches(rollout_batch, adv_batch or {})
        self.actor.clear_pixel_cache()
        if self.critic is not None:
            self.critic.clear_pixel_cache()
        if self.ref is not None:
            self.ref.clear_pixel_cache()

        _keys_json = rollout_batch.get("_image_store_keys_json")
        if _keys_json is None and adv_batch:
            _keys_json = adv_batch.get("_image_store_keys_json")
        if _keys_json is not None:
            import json as _json

            import ray

            from areal.utils.pixel_store import get_or_create_pixel_store

            _store = get_or_create_pixel_store()
            _freed = ray.get(_store.free.remote(_json.loads(_keys_json)))
            _stats = ray.get(_store.stats.remote())
            logger.info(
                "[cleanup_step] pixel store freed %d keys, "
                "remaining=%d total_puts=%d",
                _freed,
                _stats["stored"],
                _stats["total_puts"],
            )
        if global_step is not None:
            try:
                self._purge_old_rollout_dumps(global_step)
            except Exception:
                logger.warning("rollout dump purge failed", exc_info=True)
        gc.collect()

    def _purge_old_rollout_dumps(self, global_step: int) -> None:
        """Delete rollout dump dirs for versions outside the staleness window.

        Trajectories from versions older than ``global_step -
        max_head_offpolicyness`` can no longer be consumed by training
        (the rollout staleness cap rejects them), so their JSONL +
        screenshot files are dead weight on the PVC. Without this, a
        22K-step run accumulates hundreds of GB of dump files with no
        runtime use.

        Train rollouts: keep ``max_head_offpolicyness + 2`` versions.
        Test rollouts: keep last 5 (per-eval-round, slower turnover,
        useful for offline inspection).
        """
        if not getattr(self.config.rollout, "dump_to_file", False):
            return
        fileroot = getattr(self.config.cluster, "fileroot", "")
        exp = getattr(self.config, "experiment_name", "")
        trial = getattr(self.config, "trial_name", "")
        if not (fileroot and exp and trial):
            return
        from areal.utils.stats_logger import StatsLogger

        log_path = StatsLogger.get_log_path(
            experiment_name=exp, trial_name=trial, fileroot=fileroot
        )
        train_keep = (
            max(int(getattr(self.config.rollout, "max_head_offpolicyness", 0)), 0)
            + 2
        )
        eval_keep = 5
        n_freed = 0
        for sub, keep in (("rollout", train_keep), ("test-rollout", eval_keep)):
            base = os.path.join(log_path, sub)
            if not os.path.isdir(base):
                continue
            threshold = global_step - keep
            if threshold < 0:
                continue
            try:
                entries = os.listdir(base)
            except OSError:
                continue
            for entry in entries:
                try:
                    v = int(entry)
                except ValueError:
                    continue
                if v > threshold:
                    continue
                shutil.rmtree(os.path.join(base, entry), ignore_errors=True)
                n_freed += 1
        if n_freed:
            logger.info(
                "[cleanup_step] purged %d stale rollout dump dir(s) "
                "(train_keep=%d eval_keep=%d step=%d)",
                n_freed, train_keep, eval_keep, global_step,
            )

    def _save_hf(self, epoch: int, epoch_step: int, global_step: int):
        # Save as HF models for evaluation
        self.saver.save(
            self.actor,
            epoch,
            epoch_step,
            global_step,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        # Save BERT difficulty predictor alongside actor checkpoint
        _dp = getattr(self, "_difficulty_predictor", None)
        if _dp is not None:
            bert_dir = os.path.join(
                self.config.cluster.fileroot,
                "checkpoints",
                "root",
                self.config.experiment_name,
                self.config.trial_name,
                "difficulty_predictor",
            )
            os.makedirs(bert_dir, exist_ok=True)
            bert_bytes = _dp.state_dict_bytes()
            # Latest (loaded on resume)
            bert_path = os.path.join(bert_dir, "predictor.pt")
            with open(bert_path, "wb") as f:
                f.write(bert_bytes)
            # Per-step snapshot for trend analysis
            bert_snapshot = os.path.join(
                bert_dir, f"predictor_step{global_step}.pt"
            )
            with open(bert_snapshot, "wb") as f:
                f.write(bert_bytes)
            logger.info(
                "[save] BERT difficulty predictor saved: %s (+step snapshot)",
                bert_path,
            )
        if self.critic is not None:
            self.saver.save(
                self.critic,
                epoch,
                epoch_step,
                global_step,
                tokenizer=self.tokenizer,
                processor=self.processor,
                name="critic",
            )
        # Async mode: synchronization handled by AsyncCheckpointManager
        if not self.saver.is_async:
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()

    def _maybe_save_for_test_rollout(
        self, epoch: int, epoch_step: int, global_step: int
    ):
        """Save HF checkpoint if test rollout has requested one.

        The DP head checks the flag; the decision is broadcast to all ranks
        so that all ranks participate in the collective save operation.
        """
        import torch

        # DP head checks if test rollout thread requested a save; None on non-head ranks
        mgr = getattr(self, "_test_rollout", None)
        should_save = torch.tensor(
            [1 if mgr is not None and mgr.should_save() else 0],
            dtype=torch.int32,
        )
        if dist.is_initialized():
            dist.broadcast(should_save, src=0, group=self.actor.cpu_group)

        if should_save.item() == 0:
            return

        # Compute save path (same on all ranks — deterministic from config)
        path = mgr.get_save_path(global_step) if mgr is not None else ""
        # Broadcast path from head to all ranks
        if dist.is_initialized():
            path_list = [path]
            dist.broadcast_object_list(path_list, src=0, group=self.actor.cpu_group)
            path = path_list[0]

        logger.info(
            "[TestRollout] Saving HF checkpoint for test rollout: %s (step %d)",
            path,
            global_step,
        )
        import time as _time

        _t0 = _time.monotonic()
        from areal.api.io_struct import SaveLoadMeta

        meta = SaveLoadMeta(
            path=path,
            weight_format="hf",
            with_optim=False,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        self.actor.save(meta)

        # Also save critic checkpoint if available
        if self.critic is not None:
            critic_path = path + "_critic"
            critic_meta = SaveLoadMeta(
                path=critic_path,
                weight_format="hf",
                with_optim=False,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
            self.critic.save(critic_meta)

        # Save DCP recovery checkpoint (actor + critic with optimizer states)
        dcp_path = path + "_dcp"
        actor_dcp_meta = SaveLoadMeta(
            path=dcp_path + "/actor",
            weight_format="dcp",
            with_optim=True,
        )
        self.actor.save(actor_dcp_meta)
        if self.critic is not None:
            critic_dcp_meta = SaveLoadMeta(
                path=dcp_path + "/critic",
                weight_format="dcp",
                with_optim=True,
            )
            self.critic.save(critic_dcp_meta)
        logger.info("[TestRollout] DCP recovery checkpoint saved: %s", dcp_path)

        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()
        logger.info("[TestRollout] Checkpoint saved: %.1fs", _time.monotonic() - _t0)

        # Notify the eval thread (DP head only)
        if mgr is not None:
            mgr.on_save_complete(path, global_step)

    def _save_recover_checkpoint(self, epoch: int, epoch_step: int, global_step: int):
        # Save recoverable checkpoints (DCP format with optimizer states)
        to_save: dict = dict(default=self.actor)
        if self.critic is not None:
            to_save["critic"] = self.critic
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=epoch_step,
            steps_per_epoch=len(self.train_dataloader),
        )
        self.recover_handler.dump(
            to_save,
            step_info,
            self.saver,
            self.evaluator,
            self.stats_logger,
            self.train_dataloader,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        # BERT difficulty predictor sits next to recover_info so resume picks
        # up the same step. Rank 0 only — single shared file.
        _dp = getattr(self, "_difficulty_predictor", None)
        if _dp is not None and (
            not dist.is_initialized() or dist.get_rank() == 0
        ):
            from areal.utils.recover import RecoverHandler
            recover_dir = RecoverHandler.recover_info_path(
                self.config.recover.experiment_name,
                self.config.recover.trial_name,
                self.config.recover.fileroot,
            )
            os.makedirs(recover_dir, exist_ok=True)
            bert_path = os.path.join(recover_dir, "difficulty_predictor.pt")
            with open(bert_path, "wb") as f:
                f.write(_dp.state_dict_bytes())
            logger.info(
                "[recover] BERT difficulty predictor saved: %s", bert_path
            )

        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()

    def _evaluate_fn(
        self,
        eval_workflow: WorkflowLike,
        eval_workflow_kwargs,
    ):
        if self.actor.is_data_parallel_head():
            cnt = 0
            for data in self.valid_dataloader:
                for item in data:
                    self.eval_rollout.submit(
                        item,
                        eval_workflow,
                        eval_workflow_kwargs,
                        group_size=self.config.eval_gconfig.n_samples,
                        is_eval=True,
                    )
                    cnt += 1
            self.eval_rollout.wait(cnt, timeout=None)

        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()

    def _evaluate(
        self,
        eval_workflow: WorkflowLike | None,
        eval_workflow_kwargs,
        epoch: int,
        epoch_step: int,
        global_step: int,
    ):
        if self.valid_dataloader is None or eval_workflow is None:
            return
        self.evaluator.evaluate(
            functools.partial(
                self._evaluate_fn,
                eval_workflow=eval_workflow,
                eval_workflow_kwargs=eval_workflow_kwargs,
            ),
            epoch,
            epoch_step,
            global_step,
        )
        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()

    def _maybe_periodic_save(self, config, epoch, epoch_step, global_step):
        """Check and perform periodic save (time-based, for when test rollout is disabled)."""
        import time as _time
        _recover_freq = getattr(config.recover, "freq_secs", None)
        if _recover_freq is None:
            return
        _now = _time.monotonic()
        _last_save = getattr(self, "_last_periodic_save_time", 0.0)
        if _now - _last_save < _recover_freq:
            return
        self._last_periodic_save_time = _now
        logger.info(
            "Saving HF + DCP checkpoints (periodic (%ds), step=%d)",
            _recover_freq, global_step,
        )
        self._save_hf(epoch=epoch, epoch_step=epoch_step, global_step=global_step)
        self._save_recover_checkpoint(
            epoch=epoch, epoch_step=epoch_step, global_step=global_step
        )

    def _export_and_commit_stats(self, epoch: int, epoch_step: int, global_step: int):
        # Upload statistics to the logger (e.g., wandb)
        stats = self.actor.export_stats()
        if self.critic is not None:
            stats.update(self.critic.export_stats())
        stats.update(self.rollout.export_stats())
        if self.eval_rollout is not None:
            stats.update(self.eval_rollout.export_stats())

        # Actor FBC gradient tracking
        if hasattr(self, "_actor_gradient_steps"):
            stats["actor/gradient_steps"] = self._actor_gradient_steps
            stats["actor/gradient_samples"] = self._actor_gradient_samples

        # Rollout stats from the staleness manager.
        # All metrics are in EPISODE units (flatten group_size).
        # StalenessManager counts tasks; multiply by n_samples to get episodes.
        if hasattr(self.rollout, "staleness_manager"):
            rollout_stat = self.rollout.staleness_manager.get_stats()
            gs = self.config.gconfig.n_samples
            stats["staleness/rollout/trajectories_accepted"] = (
                rollout_stat.accepted * gs
            )
            stats["staleness/rollout/trajectories_rejected"] = (
                rollout_stat.rejected * gs
            )
            stats["staleness/rollout/pending_dispatch"] = rollout_stat.enqueued * gs
            stats["staleness/rollout/in_progress"] = rollout_stat.running * gs
            stats["staleness/update/awaiting_train"] = (
                self.rollout.pending_results_count * gs
            )
            stats["staleness/update/consumed_this_step"] = (
                self.rollout._last_consumed_count * gs
            )

        # Inject test rollout metrics (logged with training step as x-axis)
        _test_rollout = getattr(self, "_test_rollout", None)
        if _test_rollout is not None:
            # step_reward: running average from current/latest test rollout round
            step_metrics = _test_rollout.get_step_metrics()
            if step_metrics is not None:
                stats.update(step_metrics)
            # iteration_reward: final reward from completed round
            # (flag is still set here, cleared after commit)
            if _test_rollout.should_save():
                iter_metrics = _test_rollout.get_iteration_metrics()
                if iter_metrics is not None:
                    stats.update(iter_metrics)
                _test_rollout.clear_save_request()

        self.stats_logger.commit(epoch, epoch_step, global_step, stats)

        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()

    def _requires_proxy_workflow(self, workflow: WorkflowLike) -> bool:
        """Check if workflow requires proxy workers (i.e., not a RolloutWorkflow).

        Returns True if:
        - Workflow is NOT a RolloutWorkflow instance
        - Workflow is NOT a RolloutWorkflow class
        - Workflow is a string that does NOT import to a RolloutWorkflow

        This enables any callable object with a compatible signature to work
        without requiring inheritance from AgentWorkflow.
        """
        # Direct RolloutWorkflow instances
        if isinstance(workflow, RolloutWorkflow):
            return False

        # RolloutWorkflow classes
        if isinstance(workflow, type) and issubclass(workflow, RolloutWorkflow):
            return False

        # String import paths
        if isinstance(workflow, str):
            from areal.utils.dynamic_import import import_from_string

            try:
                imported_obj = import_from_string(workflow)
            except (ValueError, ImportError, AttributeError):
                # If import fails, assume it needs proxy (fail-safe)
                return True

            # Check if imported object is RolloutWorkflow
            if isinstance(imported_obj, RolloutWorkflow):
                return False
            if isinstance(imported_obj, type) and issubclass(
                imported_obj, RolloutWorkflow
            ):
                return False

        # Everything else requires proxy workers
        return True

    def _ensure_proxy_started(self) -> None:
        """Lazily initialize proxy workers when agent workflows are used.

        This method is called before training when a non-RolloutWorkflow is detected.
        It creates proxy workers colocated with rollout workers to handle
        OpenAI-compatible API requests from agent subprocesses.
        """
        if self._proxy_started:
            return

        # Only initialize proxy in single-controller mode with RolloutController
        if not is_single_controller():
            raise NotImplementedError("Proxy workers not supported in SPMD mode")

        if self.config.scheduler.type == "ray":
            raise NotImplementedError("Proxy workers not supported with RayScheduler")

        assert isinstance(self.rollout, RolloutController)

        logger.info("Initializing proxy workers for AgentWorkflow support")
        self.rollout.start_proxy()
        if self.eval_rollout is not None:
            self.eval_rollout.start_proxy()
        self._proxy_started = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type is not None:
            raise exc_value
