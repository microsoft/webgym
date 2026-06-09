import functools
import math
from typing import Any

import torch

from areal.api.cli_args import MicroBatchSpec, PPOActorConfig
from areal.api.engine_api import TrainEngine
from areal.infra import TrainController
from areal.utils import logging, stats_tracker
from areal.utils.constants import (
    PROX_APPROX_METHOD_LINEAR,
    PROX_APPROX_METHOD_LOGLINEAR,
    PROX_APPROX_METHOD_ROLLOUT,
    PROX_APPROX_METHODS_ALL,
    PROX_LOGP_METHOD_LOGLINEAR,
    PROX_LOGP_METHOD_METRICS,
    PROX_LOGP_METHOD_RECOMPUTE,
    ProxLogpMethod,
)
from areal.utils.data import (
    KLEstimator,
    Normalization,
    split_padded_tensor_dict_into_mb_list,
)
from areal.utils.functional import (
    ppo_actor_loss_fn,
    reward_overlong_penalty,
    sapo_loss_fn,
)
from areal.utils.perf_tracer import trace_perf

logger = logging.getLogger("PPOActor")


class PPOActor:
    def __init__(self, config: PPOActorConfig, engine: TrainEngine):
        self.config = config
        self.engine = engine

        self.reward_bias = config.reward_bias
        self.reward_scaling = config.reward_scaling
        self.reward_clip = config.reward_clip

        self.kl_ctl = config.kl_ctl
        self.kl_estimator = KLEstimator(config.kl_estimator)

        self.adv_norm = Normalization(config.adv_norm) if config.adv_norm else None
        self.reward_norm = (
            Normalization(config.reward_norm) if config.reward_norm else None
        )

        self.discount = config.discount
        self.gae_lambda = config.gae_lambda
        self.mask_no_eos_with_zero = config.mask_no_eos_with_zero

        self.temperature = config.temperature

        self.m2_threshold = config.m2_threshold

        # Log critical GSPO/GRPO configuration for reproducibility
        self._log_configuration()

    def _log_configuration(self):
        """Log PPO configuration including how proximal policy is computed."""
        config = self.config

        logger.info("=" * 70)
        logger.info("PPOActor Configuration")
        logger.info("=" * 70)

        logger.info("  critic_loss: %s", config.critic_loss)
        logger.info("  actor_loss: %s", config.actor_loss)
        if config.actor_loss == "fbc" and config.critic_loss == "none":
            logger.info("Mode: Online Filtered BC")
            logger.info("  Loss: clipped surrogate on reward>0 trajectories")
            logger.info("=" * 70)
            return
        if config.actor_loss == "fbc" and config.critic_loss == "archer":
            logger.info("Mode: ArCHer Critic + FBC Actor")
            logger.info("  Critic: ArCHer step-level V(s), trained on full data")
            logger.info("  Actor: FBC on reward>0 + step_adv>1/H trajectories")
            logger.info("=" * 70)
            return

        # Log PPO mode and proximal policy computation
        if not config.use_decoupled_loss:
            logger.info("Mode: Standard PPO (on-policy)")
            if config.recompute_logprob:
                logger.info("  old_logp (π_old): RECOMPUTED from current policy")
            else:
                logger.info(
                    "  old_logp (π_old): FROM INFERENCE (cached during rollout)"
                )
        else:
            logger.info("Mode: Decoupled PPO (off-policy)")
            logger.info("  log_p_behave (π_behave): FROM INFERENCE (behavior policy)")

            # Log proximal policy computation method
            method_descriptions = {
                PROX_LOGP_METHOD_RECOMPUTE: "RECOMPUTED via forward pass (standard decoupled PPO)",
                PROX_LOGP_METHOD_LOGLINEAR: "LOG-LINEAR APPROXIMATION (no forward pass)",
                PROX_LOGP_METHOD_METRICS: "RECOMPUTED + APPROXIMATION METRICS (for evaluation)",
            }
            desc = method_descriptions.get(
                config.prox_logp_method, f"UNKNOWN ({config.prox_logp_method})"
            )
            logger.info(f"  Proximal policy (π_prox): {desc}")

            logger.info("  log_p_theta (π_θ): TRAINING FORWARD PASS (current policy)")

            if config.behav_imp_weight_cap:
                logger.info(
                    f"  Importance weight cap: {config.behav_imp_weight_cap:.1f} "
                    "(filters out tokens with extreme weights)"
                )

        # Log other critical config
        logger.info("=" * 70)
        logger.info("Training Parameters:")
        logger.info(
            f"  importance_sampling_level: {getattr(config, 'importance_sampling_level', 'token')}"
        )
        logger.info(
            f"  adv_norm: {config.adv_norm if config.adv_norm else 'DISABLED (None)'}"
        )
        logger.info(
            f"  reward_norm: {config.reward_norm if config.reward_norm else 'DISABLED (None)'}"
        )
        logger.info(f"  eps_clip: {config.eps_clip}")
        logger.info("=" * 70)

    @trace_perf("ppo_actor.compute_logp", category="compute")
    @torch.no_grad()
    def compute_logp(self, data: dict[str, Any]) -> torch.Tensor:
        self.engine.eval()
        return self.engine.forward(
            input_=data,
            aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
        )

    @trace_perf("ppo_actor.compute_advantages", category="compute")
    def compute_advantages(self, data: dict[str, Any]) -> dict[str, Any]:
        import time as _time

        _ca_t0 = _time.monotonic()
        bs = data["input_ids"].shape[0]
        max_seqlen = data["input_ids"].shape[1]

        # Online Filtered BC (online_fbc=true) runs through the standard path.
        # Early filtering (reward>0) happens in rl_trainer.py; config handles
        # the rest: critic=null → values=0, adv_norm=null, reward_norm=null.
        # KL regularization works naturally when ref model is configured.

        batch_indices = torch.arange(
            bs, device=data["input_ids"].device, dtype=torch.long
        )

        # Reward Penalty on length (legacy DAPO single-turn path)
        if self.config.overlong_reward_penalty:
            overlong_tokens = self.config.overlong_tokens
            overlong_penalty_factor = self.config.overlong_penalty_factor
            assert overlong_tokens is not None
            assert overlong_penalty_factor is not None
            # Only apply for single-turn (each row = one episode).
            if data.get("episode_id") is None:
                data = reward_overlong_penalty(
                    data,
                    overlong_tokens=overlong_tokens,
                    overlong_penalty_factor=overlong_penalty_factor,
                    max_response_length=self.config.max_new_tokens,
                )

        # Reward Scaling
        reward_score = data["rewards"]
        reward_score = (reward_score + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(
            reward_score, max=self.reward_clip, min=-self.reward_clip
        )

        # FBC-only: filter steps whose screenshot is identical to the next
        # step's (i.e. the action produced no visible change). These rows
        # carry no learning signal, so zero their entire loss_mask row before
        # any downstream consumption (KL/GAE/loss/denominator).
        if (
            self.config.actor_loss == "fbc"
            and self.config.filter_unchanged_screenshot
            and "is_screenshot_unchanged" in data
        ):
            _isu = data["is_screenshot_unchanged"]
            _lm = data["loss_mask"]
            if _isu.dim() == 1 and _isu.shape[0] == _lm.shape[0]:
                _keep = (_isu == 0).to(_lm.dtype).unsqueeze(-1)
                data["loss_mask"] = _lm * _keep
                _kept = int(_keep.sum().item())
                _total = _isu.shape[0]
                logger.info(
                    "[fbc filter_unchanged_screenshot] kept %d/%d rows "
                    "(filtered %d unchanged-screenshot steps)",
                    _kept, _total, _total - _kept,
                )

        loss_mask = data["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
        # Apply the mask to log probabilities.
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
            # Overwrite logprobs produced by the inference engine
            prox_logp_value = data["prox_logp"]
            if prox_logp_value is None:
                raise ValueError(
                    "prox_logp is None but recompute_logprob=True. "
                    "This indicates compute_logp() was skipped incorrectly."
                )
            old_logp = data["logprobs"] = prox_logp_value
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
            if not self.config.use_decoupled_loss:
                # prox logp not available, use inferenced logp
                data["prox_logp"] = old_logp
        ref_logp = data.get("ref_logp")
        if ref_logp is None:
            ref_logp = torch.zeros_like(old_logp)
        # Shape sanity check before element-wise ops
        if ref_logp.shape != loss_mask.shape:
            logger.error(
                "[BUG] ref_logp.shape=%s != loss_mask.shape=%s  old_logp.shape=%s  "
                "prox_logp type=%s  data keys=%s",
                ref_logp.shape, loss_mask.shape, old_logp.shape,
                type(data.get("prox_logp")).__name__,
                list(data.keys()),
            )
            raise RuntimeError(
                f"Shape mismatch in compute_advantages: ref_logp {ref_logp.shape} "
                f"vs loss_mask {loss_mask.shape}. This usually means a 1D packed "
                f"tensor (from compute_logp) was not unpacked to 2D."
            )
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # ── ArCHer step-level advantages ──
        if self.config.archer_mode:
            advantages, kl_rewards, tot_rewards = self._compute_advantages_archer(
                data, bs, max_seqlen, batch_indices, loss_mask,
                old_logp, ref_logp, reward_score, _ca_t0,
            )
        else:
            # ── Standard GAE path ──
            # Episode-aware reward normalization.
            # For GRPO: group by group_id (same task's episodes share one
            # group_id) so normalization compares same-prompt rollouts.
            # Falls back to episode_id with positional grouping if no group_id.
            episode_ids = data.get("episode_id")
            group_ids = data.get("group_id")
            if episode_ids is not None and self.reward_norm:
                unique_eps, inverse = torch.unique(episode_ids, return_inverse=True)
                n_episodes = unique_eps.shape[0]

                first_idx = torch.zeros(
                    n_episodes, dtype=torch.long, device=episode_ids.device
                )
                seen = torch.zeros(n_episodes, dtype=torch.bool, device=episode_ids.device)
                for idx in range(bs):
                    ep = inverse[idx]
                    if not seen[ep]:
                        first_idx[ep] = idx
                        seen[ep] = True

                ep_rewards = reward_score[first_idx]
                ep_rewards_raw = ep_rewards.clone()

                # Use group_id for normalization grouping if available
                if group_ids is not None:
                    ep_group_ids = group_ids[first_idx]
                    ep_rewards = self.reward_norm(
                        ep_rewards, group_ids=ep_group_ids
                    )
                else:
                    ep_rewards = self.reward_norm(ep_rewards)

                reward_score = ep_rewards[inverse]
            elif self.reward_norm:
                reward_score = self.reward_norm(reward_score)

            # ── Compute step-level scalar advantage BEFORE GAE ──
            # step_adv[i] is a per-row scalar = trajectory-level advantage
            # divided by T_g (or by a fixed T_const when
            # step_adv_const_length is set).
            #
            # KL is no longer injected into per-token rewards — it is now
            # a SEPARATE loss term in grpo_loss_fn. kl_rewards below is
            # kept only for logging compatibility.
            horizons_data = data.get("horizon")
            multi_step = (
                episode_ids is not None and horizons_data is not None
            )
            if multi_step:
                step_adv = self._compute_step_level_advantages(
                    reward_score=reward_score,
                    episode_ids=episode_ids,
                )
            else:
                # Single-turn fallback: each row is its own trajectory.
                step_adv = reward_score

            attn_mask = data["attention_mask"]
            seqlens = attn_mask.sum(-1).long()
            seq_no_eos_mask = seqlens == attn_mask.shape[1]
            # KL stays in per-token rewards (standard GRPO): per-token KL
            # penalty across the response, plus step_adv placed at the
            # second-to-last token. GAE then propagates both.
            # When kl_use_prox is set, the second distribution is the
            # recomputed proximal logp (data["prox_logp"]) rather than
            # the frozen reference logp.
            # FRPO mode injects KL directly as advantage modification at
            # loss time (grpo_loss_fn), so skip the reward-side injection
            # here to avoid double-counting.
            if self.config.use_frpo_loss_term:
                kl_rewards = torch.zeros_like(old_logp)
            else:
                kl_target = (
                    data["prox_logp"] if self.config.kl_use_prox else ref_logp
                )
                kl_rewards = -self.kl_ctl * self.kl_estimator(old_logp, kl_target)

            # Diagnostic: bin per-token |KL| (unscaled) by relative position
            # within each row's response (loss_mask=1) and log per-quartile
            # mean. Verifies whether KL shrinks toward the end of the response.
            with torch.no_grad():
                kl_unscaled_abs = (
                    kl_rewards.detach().abs() / max(self.kl_ctl, 1e-12)
                )
                lm = loss_mask.bool()
                if lm.any():
                    # Per-row: positions are 0..seq_len-1; cumulative count of
                    # masked tokens gives 1-indexed rank within the response.
                    cum = lm.long().cumsum(dim=1)  # [bs, seq_len]
                    n_resp = lm.long().sum(dim=1).clamp(min=1)  # [bs]
                    # Relative position in (0, 1] for masked tokens
                    rel = cum.float() / n_resp.unsqueeze(1).float()
                    # Quartile id 0..3
                    q = (rel * 4.0 - 1e-9).clamp(min=0.0).long().clamp(max=3)
                    flat_q = q[lm]
                    flat_kl = kl_unscaled_abs[lm]
                    q_means = []
                    for qi in range(4):
                        sel = flat_q == qi
                        if sel.any():
                            q_means.append(flat_kl[sel].mean().item())
                        else:
                            q_means.append(float("nan"))
                    stats_tracker.scalar(
                        kl_pos_q1_abs=q_means[0],
                        kl_pos_q2_abs=q_means[1],
                        kl_pos_q3_abs=q_means[2],
                        kl_pos_q4_abs=q_means[3],
                    )

            rewards = kl_rewards.clone()
            rewards[batch_indices, seqlens - 1] = 0
            indices = torch.clip(seqlens - 2, min=0)
            if self.mask_no_eos_with_zero:
                rewards[batch_indices, indices] += torch.where(
                    seq_no_eos_mask, 0, step_adv
                )
            else:
                rewards[batch_indices, indices] += step_adv

            # GAE
            _ca_t1 = _time.monotonic()
            if "values" not in data:
                values = torch.zeros_like(rewards)
            else:
                values = data["values"]

            advantages = torch.zeros(
                bs, max_seqlen, dtype=torch.float32, device=values.device
            )
            lastgaelam = 0
            nextvalues = values[:, max_seqlen - 1] * seq_no_eos_mask
            for t in reversed(range(max_seqlen - 1)):
                delta = rewards[:, t] + self.discount * nextvalues - values[:, t]
                newgaelam = delta + self.discount * self.gae_lambda * lastgaelam
                mask = loss_mask[:, t]
                nextvalues = nextvalues * (1 - mask) + values[:, t] * mask
                lastgaelam = lastgaelam * (1 - mask) + newgaelam * mask
                advantages[:, t] = lastgaelam

            _ca_t2 = _time.monotonic()
            data["returns"] = advantages + values
            tot_rewards = rewards

            logger.info(
                "[compute_advantages GAE] bs=%d max_seqlen=%d "
                "reward_norm=%.2fs GAE_loop=%.2fs",
                bs, max_seqlen, _ca_t1 - _ca_t0, _ca_t2 - _ca_t1,
            )

        # ── Shared: advantage normalization (applies to both paths) ──
        # For GRPO: group by group_id (same task) rather than positional slicing.
        if self.adv_norm is not None:
            episode_ids = data.get("episode_id")
            group_ids = data.get("group_id")
            if episode_ids is not None:
                unique_eps, inverse = torch.unique(episode_ids, return_inverse=True)
                n_episodes = unique_eps.shape[0]

                # Build episode→group mapping from group_id if available
                if group_ids is not None:
                    # Map each unique episode to its group_id
                    first_idx = torch.zeros(
                        n_episodes, dtype=torch.long, device=episode_ids.device
                    )
                    _seen = torch.zeros(
                        n_episodes, dtype=torch.bool, device=episode_ids.device
                    )
                    for idx in range(bs):
                        ep = inverse[idx]
                        if not _seen[ep]:
                            first_idx[ep] = idx
                            _seen[ep] = True
                    ep_group_ids = group_ids[first_idx]
                    unique_groups = torch.unique(ep_group_ids)
                    n_groups = unique_groups.shape[0]

                    normed = torch.zeros_like(advantages)
                    for gid in unique_groups:
                        # All episodes belonging to this group
                        ep_mask = ep_group_ids == gid
                        group_mask = torch.zeros(
                            bs, dtype=torch.bool, device=advantages.device
                        )
                        for ep_idx in range(n_episodes):
                            if ep_mask[ep_idx]:
                                group_mask |= inverse == ep_idx
                        g_adv = advantages[group_mask]
                        g_lm = loss_mask[group_mask]
                        active = g_adv[g_lm.bool()]
                        if active.numel() > 1:
                            mean = active.mean()
                            std = active.std() + self.adv_norm.eps
                            normed[group_mask] = (
                                (advantages[group_mask] - mean) / std
                            )
                        else:
                            normed[group_mask] = advantages[group_mask]
                    advantages = normed * loss_mask
                    logger.info(
                        "[compute_advantages] adv_norm: n_episodes=%d "
                        "n_groups=%d grouped_by=group_id",
                        n_episodes,
                        n_groups,
                    )
                else:
                    # Fallback: positional grouping by group_size (FBC path)
                    group_size = self.adv_norm.group_size or n_episodes
                    normed = torch.zeros_like(advantages)
                    for g_start in range(0, n_episodes, group_size):
                        g_end = min(g_start + group_size, n_episodes)
                        group_mask = torch.zeros(
                            bs, dtype=torch.bool, device=advantages.device
                        )
                        for ep_idx in range(g_start, g_end):
                            group_mask |= inverse == ep_idx
                        g_adv = advantages[group_mask]
                        g_lm = loss_mask[group_mask]
                        active = g_adv[g_lm.bool()]
                        if active.numel() > 1:
                            mean = active.mean()
                            std = active.std() + self.adv_norm.eps
                            normed[group_mask] = (
                                (advantages[group_mask] - mean) / std
                            )
                        else:
                            normed[group_mask] = advantages[group_mask]
                    advantages = normed * loss_mask
                    logger.info(
                        "[compute_advantages] adv_norm: n_episodes=%d "
                        "group_size=%d n_groups=%d grouped_by=position",
                        n_episodes,
                        group_size,
                        (n_episodes + group_size - 1) // group_size,
                    )
            else:
                advantages = self.adv_norm(advantages, loss_mask)


        _ca_t3 = _time.monotonic()
        data["advantages"] = advantages
        data["kl_rewards"] = kl_rewards
        data["tot_rewards"] = tot_rewards
        data["loss_mask"] = loss_mask
        data["logprobs"] = old_logp

        logger.info(
            "[compute_advantages] bs=%d max_seqlen=%d device=%s total=%.2fs",
            bs, max_seqlen, data["input_ids"].device, _ca_t3 - _ca_t0,
        )

        return data

    def _compute_step_level_advantages(
        self,
        reward_score: torch.Tensor,
        episode_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-row step-level advantage scalar BEFORE GAE.

        Per-step normalization: step_adv = A_g / T_g  (or A_g / T_const
        when step_adv_const_length is set).

        Returns
        -------
        step_adv : torch.Tensor [bs]
            Final per-row scalar advantage. Caller places this at the
            second-to-last token of each row before GAE backward.
        """
        step_adv = reward_score.clone().float()

        unique_eps, inverse = torch.unique(episode_ids, return_inverse=True)

        # Optional fixed constant T for per-step advantage normalization.
        T_const = self.config.step_adv_const_length

        for ep_i in range(unique_eps.shape[0]):
            ep_mask = inverse == ep_i
            ep_indices = ep_mask.nonzero(as_tuple=True)[0]
            T_g = int(ep_indices.shape[0])
            if T_g == 0:
                continue

            # Vanilla per-step advantage:
            #   - trajectory-level (default):  step_adv = A_g / T_g
            #   - fixed-constant (config set): step_adv = A_g / T_const
            normalizer = float(T_const) if T_const is not None else float(T_g)
            step_adv[ep_indices] = reward_score[ep_indices].float() / normalizer

        logger.info(
            "[step_level_adv] eps=%d", unique_eps.shape[0],
        )
        return step_adv

    def _compute_advantages_archer(
        self,
        data: dict[str, Any],
        bs: int,
        max_seqlen: int,
        batch_indices: torch.Tensor,
        loss_mask: torch.Tensor,
        old_logp: torch.Tensor,
        ref_logp: torch.Tensor,
        reward_score: torch.Tensor,
        _ca_t0: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ArCHer step-level advantages.

        Returns (advantages, kl_rewards, tot_rewards) — caller stores in data.
        Critic targets (archer_critic_mask, archer_returns) are prepared
        separately by _prepare_archer_critic_targets in rl_trainer.py.
        """
        import time as _time

        _ca_t1 = _time.monotonic()
        device = data["input_ids"].device

        prompt_pos = loss_mask.int().argmax(dim=-1)

        if "values" not in data:
            step_values = torch.zeros(bs, dtype=torch.float32, device=device)
        else:
            values = data["values"]
            step_values = values[batch_indices, prompt_pos]

        episode_ids = data.get("episode_id")
        step_advantages = torch.zeros(bs, dtype=torch.float32, device=device)

        if episode_ids is not None:
            unique_eps, inverse = torch.unique(episode_ids, return_inverse=True)
            n_episodes = unique_eps.shape[0]

            for ep_idx in range(n_episodes):
                ep_step_mask = inverse == ep_idx
                ep_indices = ep_step_mask.nonzero(as_tuple=True)[0]
                N = ep_indices.shape[0]

                for i in range(N - 1):
                    step_advantages[ep_indices[i]] = (
                        step_values[ep_indices[i + 1]]
                        - step_values[ep_indices[i]]
                    )
                step_advantages[ep_indices[N - 1]] = (
                    -step_values[ep_indices[N - 1]]
                )

        _ca_t2 = _time.monotonic()

        # Broadcast step advantages to token level
        advantages = (
            step_advantages.unsqueeze(1).expand(bs, max_seqlen) * loss_mask
        )

        # KL rewards (same as standard path: kl_use_prox swaps pi_ref for pi_prox).
        kl_target = (
            data["prox_logp"] if self.config.kl_use_prox else ref_logp
        )
        kl_rewards = -self.kl_ctl * self.kl_estimator(old_logp, kl_target)
        tot_rewards = torch.zeros_like(loss_mask)

        # Store archer-specific fields for recompute
        data["archer_reward_score"] = reward_score

        # Log critic value spread: avg(V(last) - V(first)) per episode,
        # split by reward. Tracks whether critic learns step-level discrimination.
        if episode_ids is not None:
            v_spread_pos = []  # reward > 0 episodes
            v_spread_neg = []  # reward = 0 episodes
            for ep_idx in range(unique_eps.shape[0]):
                ep_indices = (inverse == ep_idx).nonzero(as_tuple=True)[0]
                if ep_indices.shape[0] < 2:
                    continue
                spread = (
                    step_values[ep_indices[-1]] - step_values[ep_indices[0]]
                ).item()
                if reward_score[ep_indices[0]].item() > 0:
                    v_spread_pos.append(spread)
                else:
                    v_spread_neg.append(spread)
            avg_spread_pos = sum(v_spread_pos) / max(len(v_spread_pos), 1)
            avg_spread_neg = sum(v_spread_neg) / max(len(v_spread_neg), 1)
            avg_v = step_values.mean().item()
            stats_tracker.scalar(
                critic_v_spread_pos=avg_spread_pos,
                critic_v_spread_neg=avg_spread_neg,
                critic_v_avg=avg_v,
                critic_v_std=step_values.std().item(),
            )

        logger.info(
            "[ArCHer advantages] bs=%d "
            "step_values=%s step_adv=%s comp=%.2fs",
            bs,
            [round(x, 3) for x in step_values.tolist()[:8]],
            [round(x, 3) for x in step_advantages.tolist()[:8]],
            _ca_t2 - _ca_t1,
        )
        return advantages, kl_rewards, tot_rewards

    def recompute_archer_advantages(self, data: dict[str, Any]) -> None:
        """Recompute ArCHer step advantages using updated critic values.

        Called after critic training so the actor trains with updated V(t).
        Only step-level TD residuals; no trajectory-level signal.
        """
        loss_mask = data["loss_mask"].float()
        bs = loss_mask.shape[0]
        max_seqlen = loss_mask.shape[1]
        device = loss_mask.device
        batch_indices = torch.arange(bs, device=device)

        prompt_pos = loss_mask.int().argmax(dim=-1)
        step_values = data["values"][batch_indices, prompt_pos]

        episode_ids = data.get("episode_id")
        reward_score = data["archer_reward_score"]  # raw scaled rewards

        step_advantages = torch.zeros(bs, dtype=torch.float32, device=device)

        if episode_ids is not None:
            unique_eps, inverse = torch.unique(episode_ids, return_inverse=True)
            n_episodes = unique_eps.shape[0]

            first_idx = torch.zeros(
                n_episodes, dtype=torch.long, device=device
            )
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

                for i in range(N - 1):
                    step_advantages[ep_indices[i]] = (
                        step_values[ep_indices[i + 1]]
                        - step_values[ep_indices[i]]
                    )
                step_advantages[ep_indices[N - 1]] = (
                    -step_values[ep_indices[N - 1]]
                )

        advantages = (
            step_advantages.unsqueeze(1).expand(bs, max_seqlen) * loss_mask
        )
        data["advantages"] = advantages

        logger.info(
            "[ArCHer recompute] bs=%d step_values=%s step_adv=%s",
            bs,
            [round(x, 3) for x in step_values.tolist()[:8]],
            [round(x, 3) for x in step_advantages.tolist()[:8]],
        )

    @trace_perf("ppo_actor.ppo_update", category="compute")
    @stats_tracker.scope_func_wrapper("ppo_actor")
    def ppo_update(self, data: dict[str, Any]) -> None:
        attn_mask = data["attention_mask"]
        loss_mask = data["loss_mask"]
        reward_score = data["rewards"]
        seqlens = attn_mask.sum(-1)

        ########## Logging code starts ##########
        result_denominators = {
            "correct_n_seqs": (reward_score > 0).bool(),
            "incorrect_n_seqs": (reward_score <= 0).bool(),
        }
        if self.config.log_agent_stats:
            if "begin_of_trajectory" not in data:
                raise RuntimeError(
                    "'begin_of_trajectory' is expected to log agent statistics"
                )
            if len(self.config.log_agent_stats_keys) == 0:
                raise RuntimeError(
                    "`log_agent_stats_keys` should not be empty when log_agent_stats=True"
                )
            agent_denominator = (data["begin_of_trajectory"] > 0).bool()
            result_denominators["agent"] = agent_denominator
        global_denominators = dict(
            n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
            n_tokens=torch.ones_like(loss_mask, dtype=torch.bool),
            n_valid_tokens=loss_mask.bool(),
            **result_denominators,
        )
        stats_tracker.denominator(**global_denominators)
        stats_tracker.stat(
            correct_seq_len=seqlens.float(), denominator="correct_n_seqs"
        )
        stats_tracker.stat(
            incorrect_seq_len=seqlens.float(), denominator="incorrect_n_seqs"
        )

        stats = dict(
            advantages=data["advantages"],
            kl_rewards=data["kl_rewards"],
            final_reward=data["tot_rewards"],
        )
        stats_tracker.stat(**stats, denominator="n_valid_tokens")

        prompt_lens = data["attention_mask"].sum(-1) - data["loss_mask"].sum(-1)
        response_lens = data["loss_mask"].sum(-1)
        seq_stats = dict(
            no_eos_ratios=(seqlens == attn_mask.shape[-1]).float(),
            prompt_len=prompt_lens.float(),
            response_len=response_lens.float(),
            seq_len=seqlens.float(),
        )
        stats_tracker.stat(**seq_stats, denominator="n_seqs")

        # GRPO group metrics (pass@k, variance) are logged on the rollout
        # side in GroupedRolloutWorkflow where raw per-episode rewards and
        # correct group boundaries are available.

        scalars = dict(
            mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
            eps_clip=self.config.eps_clip,
        )
        if self.config.c_clip is not None:
            scalars["c_clip"] = self.config.c_clip
            scalars["use_dual_clip"] = 1
        else:
            scalars["use_dual_clip"] = 0
        if self.config.behav_imp_weight_cap is not None:
            scalars["behav_imp_weight_cap"] = self.config.behav_imp_weight_cap
        stats_tracker.scalar(**scalars)

        if self.config.log_agent_stats:
            stats_tracker.stat(
                **{k: data[k].float() for k in self.config.log_agent_stats_keys},
                denominator="agent",
            )
        ########## Logging code ends ##########

        # Pop keys that are no longer needed after advantage computation
        # Note: "versions" is kept if needed for approximation/metrics in loss function
        for key in ["rewards", "tot_rewards", "kl_rewards"]:
            data.pop(key, None)

        # NOTE: Online Filtered BC trajectory filtering happens in
        # rl_trainer.py (before dispatch) to keep FSDP shards balanced.

        # NOTE: calling engine.train() is critical to enabling gradient checkpointing
        self.engine.train()
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )

        with stats_tracker.scope("update"):
            # Get current version for proximal approximation metrics
            current_version = self.engine.get_version()

            # online_fbc uses the same GRPO loss but with advantages=1 (set in
            # compute_advantages) and filtering done in rl_trainer.py
            loss_fn = functools.partial(
                grpo_loss_fn,
                eps_clip=self.config.eps_clip,
                eps_clip_higher=self.config.eps_clip_higher,
                c_clip=self.config.c_clip,
                behav_imp_weight_cap=self.config.behav_imp_weight_cap,
                m2_threshold=self.m2_threshold,
                importance_sampling_level=self.config.importance_sampling_level,
                current_version=current_version,
                prox_logp_method=self.config.prox_logp_method,
                use_sapo_loss=self.config.use_sapo_loss,
                sapo_tau_pos=self.config.sapo_tau_pos,
                sapo_tau_neg=self.config.sapo_tau_neg,
                use_decoupled_loss=self.config.use_decoupled_loss,
                kl_ctl=self.config.kl_ctl,
                use_frpo_loss_term=self.config.use_frpo_loss_term,
                frpo_clip_eps=self.config.frpo_clip_eps,
                frpo_length_norm=self.config.frpo_length_norm,
                use_frpo_weight_clip=self.config.use_frpo_weight_clip,
                frpo_weight_clip_eps=self.config.frpo_weight_clip_eps,
            )

            for _epoch in range(self.config.ppo_n_epochs):
                for mb in mb_inputs.mbs:
                    train_stat = self.engine.train_batch(
                        mb,
                        loss_fn=loss_fn,
                        loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
                    )
                    stats_tracker.scalar(**train_stat)


class PPOActorController(TrainController):
    def compute_logp(self, *args, **kwargs):
        return self._custom_function_call("compute_logp", *args, **kwargs)

    def compute_advantages(self, data, *args, **kwargs):
        # Run locally — compute_advantages is pure math (no model weights needed).
        # Only localize fields actually used; skip remotize entirely by
        # wrapping results as RTensors with real data (no ray.put).
        from areal.infra.rpc.rtensor import RTensor, _find_layout_rtensor

        layout = _find_layout_rtensor(data)

        # Fields that compute_advantages reads
        _NEEDED_KEYS = {
            "rewards",
            "loss_mask",
            "logprobs",
            "prox_logp",
            "ref_logp",
            "group_id",
            "attention_mask",
            "episode_id",
            "values",
            "horizon",  # ArCHer: per-step max_steps from difficulty
            "is_screenshot_unchanged",  # FBC filter_unchanged_screenshot
        }
        # Fields that compute_advantages creates or modifies.
        # Includes prox_logp (overwritten in else branch) and rewards
        # (mutated by reward_overlong_penalty) for correctness.
        _MODIFIED_KEYS = {
            "advantages",
            "returns",
            "kl_rewards",
            "tot_rewards",
            "loss_mask",
            "logprobs",
            "prox_logp",
            "rewards",
            "archer_critic_mask",  # ArCHer: prompt-position mask for critic
            "archer_returns",  # ArCHer: MC returns at prompt positions
            "archer_reward_score",  # ArCHer: processed reward for recompute
        }

        # Selectively localize only needed fields
        local_subset: dict[str, Any] = {}
        for k, v in data.items():
            if k in _NEEDED_KEYS:
                local_subset[k] = RTensor.localize(v) if isinstance(v, RTensor) else v

        # ── Shape reconciliation ──
        # RTensor.localize() pads each field's shards independently, so
        # different 2D fields can end up with different seq_len.  Re-pad all
        # 2D tensors to the same max width to prevent shape mismatches in
        # compute_advantages (e.g. ref_logp *= loss_mask).
        _max_seq = 0
        for v in local_subset.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 2:
                _max_seq = max(_max_seq, v.shape[1])
        if _max_seq > 0:
            for k, v in local_subset.items():
                if isinstance(v, torch.Tensor) and v.ndim >= 2 and v.shape[1] < _max_seq:
                    pad_w = _max_seq - v.shape[1]
                    local_subset[k] = torch.nn.functional.pad(v, (0, pad_w), value=0)

        # ── Unpack 1D packed tensors from compute_logp ──
        # compute_logp (actor & ref) returns 1D packed tensors [total_tokens]
        # via pack_tensor_dict on FSDP workers, but other fields are 2D
        # [batch, seq_len] from RTensor.localize().  Detect packed tensors
        # (size == attention_mask.sum()) and unpack them to 2D.
        attn = local_subset.get("attention_mask")
        if attn is not None and attn.ndim == 2:
            _bs, _ms = attn.shape
            _total_tokens = int(attn.sum().item())
            _lens = attn.sum(dim=1).int()
            for k in list(local_subset.keys()):
                v = local_subset[k]
                if (
                    isinstance(v, torch.Tensor)
                    and v.ndim == 1
                    and v.shape[0] == _total_tokens
                    and _total_tokens != _bs  # avoid false match on 1D [batch]
                ):
                    unpacked = torch.zeros(
                        _bs, _ms, *v.shape[1:], device=v.device, dtype=v.dtype
                    )
                    _off = 0
                    for _i in range(_bs):
                        _sl = _lens[_i].item()
                        unpacked[_i, :_sl] = v[_off : _off + _sl]
                        _off += _sl
                    local_subset[k] = unpacked

        # compute_advantages accesses input_ids.shape and .device only
        attn = local_subset["attention_mask"]
        local_subset["input_ids"] = torch.zeros(
            attn.shape[0], attn.shape[1], dtype=torch.int32, device=attn.device
        )

        if not hasattr(self, "_local_ppo_actor"):
            self._local_ppo_actor = PPOActor(config=self.config, engine=None)
        result = self._local_ppo_actor.compute_advantages(local_subset, *args, **kwargs)

        # Merge: keep original RTensors for unmodified fields,
        # leave modified fields as plain tensors — _ensure_rtensors in
        # the dispatch path will re-wrap them with fresh shard_ids.
        # DO NOT wrap with RTensor(shards=layout.shards, data=tensor):
        # layout.shards are stale shard_ids pointing to old data in the
        # Ray object store. Workers calling to_local() would fetch the
        # old padded data instead of the new computed values.
        merged: dict[str, Any] = {}
        for k, v in data.items():
            if k in _MODIFIED_KEYS and k in result:
                merged[k] = result[k]  # plain tensor
            else:
                merged[k] = v  # keep original RTensor or plain value

        # Add new keys produced by compute_advantages
        for k in _MODIFIED_KEYS:
            if k not in merged and k in result:
                merged[k] = result[k]  # plain tensor

        return merged

    def stage_batch(self, *args, **kwargs):
        """Stage batch data on workers for reuse across multiple passes."""
        self._custom_function_call("stage_batch", *args, **kwargs)

    def clear_staged_batch(self):
        """Clear staged batch data on workers."""
        self._custom_function_call("clear_staged_batch")

    def recompute_archer_advantages(self, data):
        """Recompute ArCHer advantages locally with updated critic values."""
        from areal.infra.rpc.rtensor import RTensor

        _NEEDED = {"loss_mask", "values", "episode_id", "archer_reward_score"}
        local = {}
        for k, v in data.items():
            if k in _NEEDED:
                local[k] = RTensor.localize(v) if isinstance(v, RTensor) else v

        if not hasattr(self, "_local_ppo_actor"):
            self._local_ppo_actor = PPOActor(config=self.config, engine=None)
        self._local_ppo_actor.recompute_archer_advantages(local)

        # Update advantages as plain tensor (will be re-wrapped by _ensure_rtensors)
        data["advantages"] = local["advantages"]

    def ppo_update(self, *args, **kwargs) -> None:
        self._custom_function_call("ppo_update", *args, **kwargs)


def _frpo_tail_sum_log_ratio(
    log_ratio: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Per-row tail sum of (log_ratio · loss_mask).

    For each row, out[t] = Σ_{t'≥t in same row} log_ratio[t']·loss_mask[t'].
    Handles both 2D [bs, seq] and 1D packed [n_tokens] (with cu_seqlens).
    """
    masked = log_ratio * loss_mask.to(log_ratio.dtype)
    if log_ratio.dim() == 2:
        return torch.flip(torch.cumsum(torch.flip(masked, dims=[1]), dim=1), dims=[1])
    n = masked.shape[0]
    g = torch.flip(torch.cumsum(torch.flip(masked, dims=[0]), dim=0), dims=[0])
    g = torch.cat([g, torch.zeros(1, device=g.device, dtype=g.dtype)])
    if cu_seqlens is None:
        return g[:n]
    seg_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    seg_end = torch.repeat_interleave(cu_seqlens[1:], seg_lens)
    return g[:n] - g[seg_end]


def _frpo_total_response_length_per_token(
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Per-token T = total #valid tokens in the row/segment containing this token.

    Same shape as loss_mask. All positions within the same row (or 1D-packed
    segment) share the same value (= that row's response length).
    """
    lm_f = loss_mask.to(out_dtype)
    if loss_mask.dim() == 2:
        T_per_row = lm_f.sum(dim=1, keepdim=True)
        return T_per_row.expand_as(lm_f).contiguous()
    if cu_seqlens is None:
        return torch.full_like(lm_f, lm_f.sum().item())
    seg_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    n_segs = seg_lens.numel()
    seg_idx = torch.repeat_interleave(
        torch.arange(n_segs, device=loss_mask.device), seg_lens
    )
    seg_T = torch.zeros(n_segs, device=loss_mask.device, dtype=out_dtype)
    seg_T.scatter_add_(0, seg_idx, lm_f)
    return seg_T[seg_idx]


def grpo_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    eps_clip: float,
    eps_clip_higher: float | None,
    c_clip: float | None,
    behav_imp_weight_cap: float | None,
    m2_threshold: float | None = None,
    importance_sampling_level: str = "token",
    current_version: int | None = None,
    prox_logp_method: str = PROX_LOGP_METHOD_RECOMPUTE,
    use_sapo_loss: bool = False,
    sapo_tau_pos: float = 1.0,
    sapo_tau_neg: float = 1.05,
    use_decoupled_loss: bool = False,
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
    kl_ctl: float = 0.0,
    use_frpo_loss_term: bool = False,
    frpo_clip_eps: float = 0.2,
    frpo_length_norm: bool = False,
    use_frpo_weight_clip: bool = False,
    frpo_weight_clip_eps: float = 0.2,
):
    """Loss function for actor step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    old_logp = input_data["logprobs"]
    advantages = input_data["advantages"]
    loss_mask = input_data["loss_mask"].bool()
    prox_logp_gt = input_data.get("prox_logp")  # Could be None if skipped

    entropy = entropy.detach()

    # Resolve proximal log-probabilities based on method
    prox_logp = _resolve_proximal_logp(
        prox_logp_gt=prox_logp_gt,
        prox_logp_method=prox_logp_method,
        old_logp=old_logp,
        logprobs=logprobs.detach(),
        versions=input_data.get("versions"),
        current_version=current_version,
    )

    # Apply M2PO masking if threshold is set
    if m2_threshold is not None:
        loss_mask = _apply_m2po_masking(old_logp, prox_logp, loss_mask, m2_threshold)

    # FRPO advantage modification: A_t -= β · Σ_{t'≥t} clamp(log(π_θ/π_prox)).
    # π_θ is detached so this is a stop-gradient scalar correction; only the
    # standard PPO ratio·A path carries gradients. Per-token clamp bounds each
    # token's contribution to ~frpo_clip_eps, capping tail-sum variance.
    # Optional total-response-length normalization: divide tail sum by T (the
    # row's #valid tokens), so the per-row correction budget is fixed and not
    # proportional to row length.
    # When frpo_clip_eps >= 1.0 the log-ratio clamp would underflow at
    # log(0); treat that as a deliberate "disable" signal and skip the
    # advantage subtraction entirely. (Used by the FRPO weight-clip mode,
    # which doesn't want the advantage modification at all.)
    if (use_frpo_loss_term and kl_ctl > 0 and prox_logp_gt is not None
            and frpo_clip_eps < 1.0):
        with torch.no_grad():
            log_ratio = (logprobs.detach() - prox_logp_gt).float()
            lo = math.log(1.0 - frpo_clip_eps)
            hi = math.log(1.0 + frpo_clip_eps)
            log_ratio_clipped = log_ratio.clamp(min=lo, max=hi)
            cu_seqlens = input_data.get("cu_seqlens")
            tail_sum = _frpo_tail_sum_log_ratio(
                log_ratio_clipped, loss_mask, cu_seqlens,
            )
            if frpo_length_norm:
                T_per_token = _frpo_total_response_length_per_token(
                    loss_mask, cu_seqlens, out_dtype=tail_sum.dtype,
                )
                tail_sum = tail_sum / T_per_token.clamp(min=1.0)
            frpo_subtraction = kl_ctl * tail_sum
            # Log the *exact* term being subtracted (β·tail_sum_clipped,
            # divided by T if length_norm enabled) and its ratio to the
            # ORIGINAL GRPO advantage (recorded before the in-place subtract).
            # Local denominators because `n_valid_tokens` is registered later
            # in this function — using it here would break zip-alignment.
            adv_eps = 1e-6
            ratio_mask = loss_mask & (advantages.abs() > adv_eps)
            adv_safe = torch.where(
                advantages.abs() > adv_eps,
                advantages,
                torch.ones_like(advantages),
            )
            stats_tracker.denominator(frpo_sub_denom=loss_mask.bool())
            stats_tracker.stat(
                frpo_subtraction=frpo_subtraction,
                frpo_subtraction_abs=frpo_subtraction.abs(),
                denominator="frpo_sub_denom",
            )
            stats_tracker.denominator(frpo_sub_ratio_valid=ratio_mask)
            stats_tracker.stat(
                frpo_subtraction_over_adv=frpo_subtraction / adv_safe,
                frpo_abs_sub_over_abs_adv=(
                    frpo_subtraction.abs()
                    / advantages.abs().clamp(min=adv_eps)
                ),
                denominator="frpo_sub_ratio_valid",
            )
            advantages = advantages - frpo_subtraction

    # Use SAPO or PPO loss
    if use_sapo_loss:
        if use_decoupled_loss:
            raise ValueError(
                "SAPO is not compatible with `use_decoupled_loss=True`. "
                "Please set `actor.use_decoupled_loss=false` in your configuration."
            )
        loss, stat = sapo_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logp,
            advantages=advantages,
            tau_pos=sapo_tau_pos,
            tau_neg=sapo_tau_neg,
            loss_mask=loss_mask,
            importance_sampling_level=importance_sampling_level,
            cu_seqlens=input_data.get("cu_seqlens"),
        )
    else:
        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logp,
            advantages=advantages,
            eps_clip=eps_clip,
            eps_clip_higher=eps_clip_higher,
            loss_mask=loss_mask,
            c_clip=c_clip,
            proximal_logprobs=prox_logp,
            behav_imp_weight_cap=behav_imp_weight_cap,
            importance_sampling_level=importance_sampling_level,
            cu_seqlens=input_data.get("cu_seqlens"),
            use_frpo_weight_clip=use_frpo_weight_clip,
            frpo_weight_clip_eps=frpo_weight_clip_eps,
        )

    # KL is included in per-token rewards in compute_advantages, so it
    # already flows through GAE into `advantages` — no separate loss term
    # needed here.

    # Log training statistics
    stats_tracker.denominator(
        # NOTE: n_tokens must have shape [batch, seq] to match vocab stats.
        # Using torch.ones_like(loss_mask) ensures correct shape when this function is called
        # standalone (e.g., by tests), not just from ppo_update() which already
        # registers n_tokens.
        n_tokens=torch.ones_like(loss_mask, dtype=torch.bool, device=logprobs.device),
        n_valid_tokens=loss_mask.bool(),
        clipped_tokens=stat["clip_mask"],
        dual_clipped_tokens=stat["dual_clip_mask"],
    )

    stats_tracker.stat(
        importance_weight=stat["importance_weight"],
        approx_kl=stat["approx_kl"],
        approx_kl_abs=stat["approx_kl"].abs(),
        new_logp=logprobs.detach(),
        old_logp=old_logp,
        entropy=entropy.float(),
        actor_loss=stat["loss"],
        clip_ratio=stat["clip_mask"].float(),
        dual_clip_ratio=stat["dual_clip_mask"].float(),
        denominator="n_valid_tokens",
    )
    if "behave_imp_weight" in stat:
        stats_tracker.denominator(unclipped_behave_tokens=stat["behave_mask"])
        stats_tracker.stat(
            behave_imp_weight=stat["behave_imp_weight"],
            behave_approx_kl=stat["behave_approx_kl"],
            behave_approx_kl_abs=stat["behave_approx_kl"].abs(),
            denominator="unclipped_behave_tokens",
        )

    if vocab_min_logits is not None and vocab_max_logits is not None:
        stats_tracker.stat(
            vocab_min_logits=vocab_min_logits,
            vocab_max_logits=vocab_max_logits,
            denominator="n_tokens",
        )

    # Log SAPO-specific statistics
    if use_sapo_loss:
        stats_tracker.stat(
            sapo_soft_gate=stat["sapo_soft_gate"],
            sapo_scaled_gate_pos=stat["sapo_scaled_gate_pos"],
            sapo_scaled_gate_neg=stat["sapo_scaled_gate_neg"],
            denominator="n_valid_tokens",
        )
    else:
        # Log clipping statistics (PPO only)
        clip_mask = stat["clip_mask"]
        clipped_new_logp = torch.where(clip_mask, logprobs.detach(), 0.0)
        clipped_old_logp = torch.where(clip_mask, old_logp, 0.0)
        stats_tracker.stat(
            clipped_new_logp=clipped_new_logp,
            clipped_old_logp=clipped_old_logp,
            denominator="clipped_tokens",
        )

    # Log proximal approximation metrics
    compute_logp_mask = stat.get("behave_mask", loss_mask)
    _log_proximal_approximation_stats(
        prox_logp_method=prox_logp_method,
        prox_logp_gt=prox_logp_gt,
        old_logp=old_logp,
        logprobs=logprobs.detach(),
        versions=input_data.get("versions"),
        current_version=current_version,
        compute_logp_mask=compute_logp_mask,
    )

    # FRPO comparison: term2 = β·Σ_{t'≥t} log(π_θ/π_prox), ratio = term2/Â_t.
    # Only meaningful when KL is enabled and π_prox is available (decoupled
    # PPO path). Uses logprobs/prox_logp_gt already in scope, no extra fwd.
    if kl_ctl > 0 and prox_logp_gt is not None:
        with torch.no_grad():
            log_ratio_th_prox = (logprobs.detach() - prox_logp_gt).float()
            tail_sum = _frpo_tail_sum_log_ratio(
                log_ratio_th_prox,
                loss_mask,
                input_data.get("cu_seqlens"),
            )
            frpo_term2 = kl_ctl * tail_sum
            adv_eps = 1e-6
            ratio_mask = loss_mask & (advantages.abs() > adv_eps)
            adv_safe = torch.where(
                advantages.abs() > adv_eps,
                advantages,
                torch.ones_like(advantages),
            )
            frpo_term2_over_term1 = frpo_term2 / adv_safe
            stats_tracker.stat(
                frpo_term2=frpo_term2,
                denominator="n_valid_tokens",
            )
            stats_tracker.denominator(frpo_ratio_valid_tokens=ratio_mask)
            stats_tracker.stat(
                frpo_term2_over_term1=frpo_term2_over_term1,
                denominator="frpo_ratio_valid_tokens",
            )

    # Log version staleness metrics
    if "versions" in input_data and current_version is not None:
        version_metrics_mask = stat.get("behave_mask", loss_mask)
        _log_version_staleness_stats(
            versions=input_data["versions"],
            current_version=current_version,
            version_metrics_mask=version_metrics_mask,
            loss_mask=loss_mask,
            episode_id=input_data.get("episode_id"),
        )

    return loss


# =============================================================================
# Core Functions
# =============================================================================


def compute_prox_logp_approximations(
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor,
    current_version: int,
    method: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute approximation(s) for proximal policy log-probabilities.

    This function approximates the log-probabilities of the proximal policy (one training step
    behind the current policy) using version-aware interpolation between the behavior policy
    (old_logp) and current policy (logprobs). This avoids the need for an expensive forward pass
    to compute the proximal policy's log-probabilities explicitly.

    Args:
        old_logp: log_p_behave from the rollout (behavior policy)
        logprobs: log_p_theta from current training forward pass
        versions: per-token policy versions from rollout (v_behave for each token)
        current_version: current training step version (v_theta)
        method: If specified, only compute this method. If None, compute all methods.

    Returns:
        Dictionary with approximation results. Single key if method specified, all methods otherwise.
    """
    # Assume proximal version is current_version - 1 (last broadcast)
    # In AReaL, proximal policy is the last updated/broadcast policy version
    v_proximal = current_version - 1

    # Extract version information
    v_behave = versions.float()
    v_theta = float(current_version)

    # CRITICAL: Only approximate generated tokens (version >= 0)
    # Prompt tokens (version < 0) must NOT be approximated - they have no generation version
    generated_tokens_mask = versions >= 0

    # Compute interpolation factor alpha
    # When v_behave == v_proximal: alpha=0 (use old_logp)
    # When v_behave == v_theta: alpha=1 (use logprobs)
    # For prompt tokens (version < 0): alpha=0 (no interpolation)
    version_diff = v_theta - v_behave
    version_gap = v_proximal - v_behave
    # Avoid division by zero AND exclude prompt tokens
    alpha = torch.where(
        (version_diff > 0) & generated_tokens_mask,
        version_gap / version_diff,
        torch.zeros_like(v_behave),
    )
    alpha = torch.clamp(alpha, 0.0, 1.0)

    approximations = {}

    # If method is specified, only compute that one
    # Otherwise compute all methods (for metrics comparison)
    methods_to_compute = [method] if method else PROX_APPROX_METHODS_ALL

    for m in methods_to_compute:
        if m == PROX_APPROX_METHOD_LOGLINEAR:
            # Method 1: Log-linear interpolation in log-space (geometric mean in probability space)
            # log(p_prox) = (1-α)·log(p_behave) + α·log(p_theta)
            approximations[PROX_APPROX_METHOD_LOGLINEAR] = old_logp + alpha * (
                logprobs - old_logp
            )

        elif m == PROX_APPROX_METHOD_LINEAR:
            # Method 2: Linear interpolation in probability space (arithmetic mean)
            # p_prox = (1-α)·p_behave + α·p_theta
            # Then convert back to log space: log(p_prox)
            p_behave = torch.exp(old_logp)
            p_theta = torch.exp(logprobs)
            p_arithmetic = (1 - alpha) * p_behave + alpha * p_theta
            approximations[PROX_APPROX_METHOD_LINEAR] = torch.log(p_arithmetic + 1e-10)

        elif m == PROX_APPROX_METHOD_ROLLOUT:
            # Method 3: Use behavior policy from rollout as-is (no approximation)
            # p_prox = p_behave
            # Used for metrics comparison
            approximations[PROX_APPROX_METHOD_ROLLOUT] = old_logp.clone()

    return approximations


def _resolve_proximal_logp(
    prox_logp_gt: torch.Tensor | None,
    prox_logp_method: str,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor | None,
    current_version: int | None,
) -> torch.Tensor:
    """
    Resolve the proximal policy log-probabilities based on the method.

    This function determines the final proximal log-probabilities to use for PPO training,
    either from ground truth (forward pass) or approximation methods.

    Args:
        prox_logp_gt: Ground truth proximal logp (from forward pass), or None if skipped.
        prox_logp_method: Method to use (recompute, loglinear, metrics).
        old_logp: Behavior policy log-probabilities.
        logprobs: Current policy log-probabilities (should be detached).
        versions: Per-token policy versions, or None.
        current_version: Current training version, or None.

    Returns:
        Resolved proximal log-probabilities tensor.

    Raises:
        ValueError: If configuration is invalid (e.g., missing required data).
        RuntimeError: If computation fails (None result, NaN, Inf).
    """
    prox_logp_is_none = prox_logp_gt is None

    # Validate configuration when prox_logp is None
    if prox_logp_is_none:
        if not ProxLogpMethod(prox_logp_method).skips_forward_pass():
            raise ValueError(
                f"prox_logp is None but prox_logp_method='{prox_logp_method}'. "
                "This indicates compute_logp() was skipped incorrectly."
            )
        if versions is None:
            raise ValueError(
                f"prox_logp is None with prox_logp_method='{prox_logp_method}' "
                "but versions not available. "
                "Cannot proceed without either ground truth or approximation."
            )

    # Determine prox_logp based on method
    prox_logp = prox_logp_gt  # Default to ground truth (could be None)

    if prox_logp_method == PROX_LOGP_METHOD_LOGLINEAR:
        # Use loglinear approximation (must compute if prox_logp is None)
        if prox_logp_is_none and versions is not None and current_version is not None:
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=PROX_APPROX_METHOD_LOGLINEAR,
            )
            prox_logp = approximations[PROX_APPROX_METHOD_LOGLINEAR]
    elif prox_logp_method == PROX_LOGP_METHOD_METRICS:
        # Metrics mode: use recomputed prox_logp for training,
        # but will also compute approximation metrics later
        pass  # Use prox_logp_gt as-is (should be recomputed)
    # else: PROX_LOGP_METHOD_RECOMPUTE - use prox_logp_gt as-is

    # Safety check: ensure we have prox_logp
    if prox_logp is None:
        raise RuntimeError(
            f"prox_logp is None after handling prox_logp_method='{prox_logp_method}'. "
            "This indicates configuration or computation error."
        )

    # Verify the value is valid
    if torch.isnan(prox_logp).any() or torch.isinf(prox_logp).any():
        raise RuntimeError(
            f"prox_logp contains NaN or Inf with prox_logp_method='{prox_logp_method}'. "
            "This indicates computation failed."
        )

    return prox_logp


def _apply_m2po_masking(
    old_logp: torch.Tensor,
    prox_logp: torch.Tensor,
    loss_mask: torch.Tensor,
    m2_threshold: float,
) -> torch.Tensor:
    """
    Apply M2PO (Second-Momentum PPO) masking to filter high-variance tokens.

    M2PO filters out tokens with high second-momentum (squared difference between
    old and proximal log-probabilities) to reduce gradient variance.

    Args:
        old_logp: Behavior policy log-probabilities.
        prox_logp: Proximal policy log-probabilities.
        loss_mask: Original loss mask [batch, seq_len].
        m2_threshold: Threshold for second-momentum filtering.

    Returns:
        Updated loss mask with M2PO filtering applied.
    """
    delta = old_logp - prox_logp
    m2 = delta * delta
    mask_flat = loss_mask.view(-1)
    m2_selected = m2.view(-1)[mask_flat]

    if m2_selected.numel() == 0:
        return loss_mask

    sorted_m2, indices = torch.sort(m2_selected, descending=True)
    restored_indices = torch.argsort(indices)
    sorted_m2_loss_mask = _get_m2po_loss_mask(
        sorted_m2=sorted_m2, m2_threshold=m2_threshold
    )
    m2_selected_mask = sorted_m2_loss_mask[restored_indices]

    m2_full_flat = torch.zeros_like(
        mask_flat, dtype=torch.bool, device=loss_mask.device
    )
    m2_full_flat[mask_flat] = m2_selected_mask

    return m2_full_flat.view_as(loss_mask)


def _get_m2po_loss_mask(
    sorted_m2: torch.Tensor,
    m2_threshold: float,
) -> torch.Tensor:
    """
    Get the mask for M2PO loss based on the second-momentum threshold.
    Mask the tokens whose second-momentum is the largest, until the average second-momentum is below the threshold.
    """
    n = sorted_m2.numel()
    if n == 0:
        return torch.ones_like(sorted_m2, dtype=torch.bool)

    # Suffix sums: S[i] = sum(sorted_m2[i:])
    suffix_sums = sorted_m2.flip(0).cumsum(0).flip(0)

    # Number of elements in suffix: N[i] = n - i
    counts = torch.arange(n, 0, -1, device=sorted_m2.device, dtype=sorted_m2.dtype)

    # Average of suffix: A[i] = S[i] / N[i]
    avg_m2_suffix = suffix_sums / counts

    # Find the first index `k` where the average of the rest is below threshold.
    below_threshold_indices = torch.where(avg_m2_suffix < m2_threshold)[0]

    if len(below_threshold_indices) > 0:
        num_to_mask = below_threshold_indices[0].item()
    else:
        # All suffix averages are >= threshold. Mask all but one to satisfy assertion.
        num_to_mask = n - 1

    loss_mask = torch.ones_like(sorted_m2, dtype=torch.bool)
    if num_to_mask > 0:
        loss_mask[:num_to_mask] = False

    if loss_mask.sum() == 0:
        raise RuntimeError("All tokens are masked out when getting the m2po loss mask.")

    return loss_mask


# =============================================================================
# Logging Helper Functions
# =============================================================================

_EPSILON = 1e-8  # Small constant for numerical stability in relative error calculations


def _compute_importance_weight(
    logp_numerator: torch.Tensor,
    logp_denominator: torch.Tensor,
) -> torch.Tensor:
    """Compute importance weight as exp(logp_num - logp_denom)."""
    return torch.exp(logp_numerator - logp_denominator).float()


def _compute_approximation_errors(
    ground_truth: torch.Tensor,
    approximation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute error metrics between ground truth and approximation.

    Returns:
        Dictionary with abs_error, rel_error, and squared_error tensors.
    """
    diff = ground_truth - approximation
    abs_error = torch.abs(diff).float()
    rel_error = torch.abs(diff / (torch.abs(ground_truth) + _EPSILON)).float()
    squared_error = (diff * diff).float()
    return {
        "abs_error": abs_error,
        "rel_error": rel_error,
        "squared_error": squared_error,
    }


def _tensor_scalar_stats(tensor: torch.Tensor) -> dict[str, float]:
    """
    Compute scalar statistics (avg, max, min) for a tensor.

    Args:
        tensor: Input tensor to compute statistics on.

    Returns:
        Dictionary with avg, max, min as Python floats.
    """
    t = tensor.float()
    return {
        "avg": t.mean().item(),
        "max": t.max().item(),
        "min": t.min().item(),
    }


def _log_approximation_metrics_for_method(
    method_name: str,
    approx_logp: torch.Tensor,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    prox_logp_gt: torch.Tensor | None = None,
) -> None:
    """
    Log metrics for a single approximation method.

    Args:
        method_name: Name of the approximation method (e.g., "loglinear").
        approx_logp: Approximated proximal log-probabilities.
        old_logp: Behavior policy log-probabilities.
        logprobs: Current policy log-probabilities.
        prox_logp_gt: Ground truth proximal logp, or None if unavailable.
    """
    # Compute importance weights from approximation
    behave_imp_weight = _compute_importance_weight(approx_logp, old_logp)
    importance_weight = _compute_importance_weight(logprobs, approx_logp)

    metrics = {
        f"{method_name}/approx_logp": approx_logp.float(),
        f"{method_name}/behave_imp_weight": behave_imp_weight,
        f"{method_name}/importance_weight": importance_weight,
    }

    # Add error metrics if ground truth is available
    if prox_logp_gt is not None:
        # Log-probability errors
        logp_errors = _compute_approximation_errors(prox_logp_gt, approx_logp)
        metrics.update(
            {
                f"{method_name}/abs_error": logp_errors["abs_error"],
                f"{method_name}/rel_error": logp_errors["rel_error"],
                f"{method_name}/squared_error": logp_errors["squared_error"],
            }
        )

        # Ground truth importance weights for comparison
        behave_imp_weight_gt = _compute_importance_weight(prox_logp_gt, old_logp)
        importance_weight_gt = _compute_importance_weight(logprobs, prox_logp_gt)

        # Importance weight errors
        behave_errors = _compute_approximation_errors(
            behave_imp_weight_gt, behave_imp_weight
        )
        imp_errors = _compute_approximation_errors(
            importance_weight_gt, importance_weight
        )

        metrics.update(
            {
                f"{method_name}/behave_imp_weight_abs_error": behave_errors[
                    "abs_error"
                ],
                f"{method_name}/behave_imp_weight_rel_error": behave_errors[
                    "rel_error"
                ],
                f"{method_name}/importance_weight_abs_error": imp_errors["abs_error"],
                f"{method_name}/importance_weight_rel_error": imp_errors["rel_error"],
            }
        )

    stats_tracker.stat(**metrics, denominator="n_valid_tokens")


def _log_proximal_approximation_stats(
    prox_logp_method: str,
    prox_logp_gt: torch.Tensor | None,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor | None,
    current_version: int | None,
    compute_logp_mask: torch.Tensor,
) -> None:
    """
    Log proximal policy approximation metrics based on the method.

    Args:
        prox_logp_method: The proximal logp method being used.
        prox_logp_gt: Ground truth proximal logp, or None if skipped.
        old_logp: Behavior policy log-probabilities.
        logprobs: Current policy log-probabilities (detached).
        versions: Per-token policy versions, or None.
        current_version: Current training version, or None.
        compute_logp_mask: Mask for valid tokens.
    """
    with stats_tracker.scope("compute_logp"):
        stats_tracker.denominator(n_valid_tokens=compute_logp_mask.bool())

        # Log ground truth when available
        if prox_logp_gt is not None:
            stats_tracker.stat(
                prox_logp_gt=prox_logp_gt.float(),
                denominator="n_valid_tokens",
            )

        # Skip if versions not available
        if versions is None or current_version is None:
            return

        if prox_logp_method == PROX_LOGP_METHOD_LOGLINEAR:
            # Loglinear mode: log approximation without error metrics
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=PROX_APPROX_METHOD_LOGLINEAR,
            )
            for method_name, approx_logp in approximations.items():
                _log_approximation_metrics_for_method(
                    method_name=method_name,
                    approx_logp=approx_logp,
                    old_logp=old_logp,
                    logprobs=logprobs,
                    prox_logp_gt=None,  # No ground truth in loglinear mode
                )

        elif prox_logp_method == PROX_LOGP_METHOD_METRICS and prox_logp_gt is not None:
            # Metrics mode: compute all methods with error metrics
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=None,  # Compute all methods
            )
            for method_name, approx_logp in approximations.items():
                _log_approximation_metrics_for_method(
                    method_name=method_name,
                    approx_logp=approx_logp,
                    old_logp=old_logp,
                    logprobs=logprobs,
                    prox_logp_gt=prox_logp_gt,
                )


def _log_version_staleness_stats(
    versions: torch.Tensor,
    current_version: int,
    version_metrics_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    episode_id: torch.Tensor | None = None,
) -> None:
    """
    Log sample staleness metrics based on policy versions (trainer side).

    Only logs batch-level metrics (offpolicy gap, per-batch unique versions).
    Accurate per-step and per-episode version stats are logged on the rollout
    side in rollout_controller.py where 2D tensors are still available.

    Args:
        versions: Per-token policy versions from rollout (1D packed or 2D padded).
        current_version: Current training version.
        version_metrics_mask: Mask for valid tokens (behave_mask, after imp weight cap).
        loss_mask: Original loss mask before behave filtering.
        episode_id: Per-row trajectory id ([bs]). When provided alongside a
            2D versions tensor, enables per-trajectory partition of weight
            updates into within-step (intra-row) vs cross-step (inter-row).
    """
    with stats_tracker.disable_scope():
        generated_mask = (versions >= 0) & loss_mask.bool()
        valid_generated_mask = version_metrics_mask & (versions >= 0)

        if not generated_mask.any():
            return

        with stats_tracker.scope("staleness/gap"):
            v_proximal = current_version - 1
            v_behave = versions.float()

            # Compute off-policy gap: v_proximal - v_behave
            offpolicy_gap = (v_proximal - v_behave)[valid_generated_mask]
            gap_stats = _tensor_scalar_stats(offpolicy_gap)

            stats_tracker.scalar(
                offpolicy_avg=gap_stats["avg"],
                offpolicy_max=gap_stats["max"],
                offpolicy_min=gap_stats["min"],
                current_version=current_version,
            )
        with stats_tracker.scope("ppo_actor"):
            n_generated = generated_mask.sum().item()
            n_behave = valid_generated_mask.sum().item()
            stats_tracker.scalar(
                behave_n_tokens=n_behave,
                behave_cap_ratio=1.0 - n_behave / max(n_generated, 1),
            )

        # Per-batch unique version count
        all_gen = versions[generated_mask]
        with stats_tracker.scope("staleness/versions_per_batch"):
            if all_gen.numel() > 0:
                stats_tracker.scalar(
                    n_unique=float(all_gen.unique().numel()),
                )

        # Per-step unique versions: each row in the 2D versions tensor is one
        # step. Count unique versions among generated tokens per row.
        if versions.dim() == 2:
            step_uniques = []
            for row_idx in range(versions.shape[0]):
                row_gen = versions[row_idx][generated_mask[row_idx]]
                if row_gen.numel() > 0:
                    step_uniques.append(float(row_gen.unique().numel()))
            if step_uniques:
                import torch as _torch

                su = _torch.tensor(step_uniques)
                with stats_tracker.scope("staleness/versions_per_step"):
                    stats_tracker.scalar(
                        avg=su.mean().item(),
                        max=su.max().item(),
                        min=su.min().item(),
                    )

        # Per-trajectory weight update partition: within-step vs cross-step.
        # Versions are monotonically non-decreasing within a trajectory so:
        #   total_updates_in_traj   = unique_versions_in_traj − 1
        #   within_step_in_traj     = Σ_rows (unique_versions_in_row − 1)
        #   cross_step_in_traj      = total − within_step
        # This decomposition is order-independent (doesn't require step order
        # within the microbatch).
        if versions.dim() == 2 and episode_id is not None:
            unique_eps = episode_id.unique()
            within_per_ep = []
            cross_per_ep = []
            for ep_id in unique_eps.tolist():
                ep_rows = (episode_id == ep_id).nonzero(as_tuple=True)[0]
                if ep_rows.numel() == 0:
                    continue
                ep_versions_all = []
                within_in_ep = 0
                for row_idx in ep_rows.tolist():
                    row_vers = versions[row_idx][generated_mask[row_idx]]
                    if row_vers.numel() == 0:
                        continue
                    unique_in_row = row_vers.unique().numel()
                    within_in_ep += max(unique_in_row - 1, 0)
                    ep_versions_all.append(row_vers)
                if not ep_versions_all:
                    continue
                all_vers = torch.cat(ep_versions_all)
                total_in_ep = max(all_vers.unique().numel() - 1, 0)
                within_per_ep.append(within_in_ep)
                cross_per_ep.append(max(total_in_ep - within_in_ep, 0))

            if within_per_ep:
                wsu = torch.tensor(within_per_ep, dtype=torch.float)
                csu = torch.tensor(cross_per_ep, dtype=torch.float)
                n_eps = float(len(within_per_ep))
                total_w = wsu.sum().item()
                total_c = csu.sum().item()
                total_u = total_w + total_c
                with stats_tracker.scope("staleness/updates_per_episode"):
                    stats_tracker.scalar(
                        within_step_avg=wsu.mean().item(),
                        within_step_max=wsu.max().item(),
                        cross_step_avg=csu.mean().item(),
                        cross_step_max=csu.max().item(),
                        n_eps_in_microbatch=n_eps,
                        n_eps_with_within_step=float((wsu > 0).sum().item()),
                        n_eps_with_cross_step=float((csu > 0).sum().item()),
                        n_eps_with_any_update=float(((wsu + csu) > 0).sum().item()),
                        frac_within_among_updates=(
                            total_w / total_u if total_u > 0 else 0.0
                        ),
                    )
