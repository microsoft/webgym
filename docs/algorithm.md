

# Algorithm Design

AsyncWebRL's algorithmic contribution is a one-line change to the
multi-step GRPO loss aggregation: replace the per-trajectory step-number
normalizer $1/|\tau_i|$ with a constant $1/k$, where $k$ is the
Easy-difficulty horizon (10 throughout the paper).

## The standard multi-step GRPO loss

The loss aggregated over a group of $G$ rollouts, with $|\tau_i|$ steps
per rollout and $|\tau_{i,j}|$ tokens per step, looks like:

$$
\mathcal J(\theta) = \mathbb E_{\tau \sim \pi_{\mathrm{behave}}}\!\left[
  \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|\tau_i|}
  \sum_{j=1}^{|\tau_i|} \sum_{t=1}^{|\tau_{i,j}|}
  \text{(per-token surrogate)}
\right]
$$

The $1/|\tau_i|$ factor normalizes each trajectory's total contribution
to the loss to 1, regardless of length. Equivalently: every trajectory
gets the same "vote", and a long trajectory's vote is split across more
tokens, so each token in a long trajectory carries a smaller share.

## The fix

Replace $1/|\tau_i|$ with a constant $1/k$:

$$
\mathcal J(\theta) = \mathbb E_{\tau \sim \pi_{\mathrm{behave}}}\!\left[
  \frac{1}{G \cdot k} \sum_{i=1}^{G}
  \sum_{j=1}^{|\tau_i|} \sum_{t=1}^{|\tau_{i,j}|}
  \text{(per-token surrogate)}
\right]
$$

Now each rollout enters the loss with weight $|\tau_i|/k$ (proportional
to length), restoring full per-token gradient weight on the long failures
the policy must learn to avoid.

## Why this matters in our setting

In WebGym, failed trajectories average **12.5 steps** against **5.1 steps**
for successes (a $\approx 2.4\times$ gap). Under the standard
$1/|\tau_i|$ normalizer, the gradient on every token in a failed
trajectory is attenuated by roughly $2.4\times$ relative to a successful
one, so the policy is implicitly told "long failures aren't very bad",
and it drifts toward longer rollouts. The constant $1/k$ replacement
removes that attenuation.

## How an episode reward becomes a per-token gradient

This is the exact sequence the code follows to take a single
binary success/fail signal at the end of an episode and turn it into the
per-token quantity that gets backpropagated. File paths are absolute
under `areal/` and line numbers are pinned to the current `main` branch.

### Step 0 — what arrives from the rollout

For each episode $e$ in a group of $G = 8$ rollouts (one prompt, $n=8$
sampled trajectories), the workflow stores a single scalar reward $r_e
\in \{0, 1\}$ plus a list of $T_e$ multi-step records (one per agent
turn). At consumer-batch assembly time those are flattened into one row
per step, all tagged with the same `episode_id` and `group_id`, so a
group becomes a contiguous block of $\sum_e T_e$ rows. The text-token
columns (`input_ids`, `loss_mask`, `logprobs`) carry the per-token
fields for the response tokens of that one step.

### Step 1 — group-normalize the episode reward

In `trainer/ppo/actor.py:174-179`, `reward_score = (rewards +
reward_bias) * reward_scaling` is applied to the raw $\{0,1\}$ vector
(identity when `reward_bias=0, reward_scaling=1`).

In `trainer/ppo/actor.py:261-297`, since `episode_id` is present and
`reward_norm.{mean_level,std_level} == group`, the code picks one row
per episode, calls `self.reward_norm(ep_rewards, group_ids=...)`, then
broadcasts the normalized scalar back to every row of that episode. The
normalizer lives in `utils/data.py:1215-1377`; the group branch reduces
to

$$
\hat r_e \;=\; \frac{r_e - \mu_g}{\sigma_g + \varepsilon},
\qquad
\mu_g = \frac{1}{8}\!\sum_{e \in g} r_e,\;
\sigma_g = \sqrt{\frac{1}{8}\!\sum_{e \in g} (r_e - \mu_g)^2}.
$$

For binary rewards with $k$ of $8$ successes in the group, this
simplifies to $\hat r = \pm \sqrt{(8-k)/k}$ for the successes and the
opposite sign for the failures. Groups with $k\!\in\!\{0,8\}$ collapse
to $\sigma_g = 0$ and are dropped upstream by the variance filter (they
contribute zero gradient anyway).

### Step 2 — flatten one episode-level $\hat r$ into per-step advantages

The step-advantage is computed by `_compute_step_level_advantages` in
`trainer/ppo/actor.py:518-557`. For each episode it sets

$$
\text{step\_adv}_{e,j} \;=\; \frac{\hat r_e}{T_{\text{const}}},
\quad j \in \{1,\dots,T_e\}
$$

when `step_adv_const_length` is set, and $\hat r_e / T_e$ otherwise
(line 551-552). With `step_adv_const_length=10.0`, every step of every
episode gets the same scalar $\hat r_e / 10$, regardless of episode
length. **This is the only line that consumes `step_adv_const_length`.**
The loss function never sees this knob — the rescaling is baked into
the advantage tensor before it ever reaches the surrogate.

### Step 3 — place the step advantage on a single response token

For each step row, the code constructs a per-token reward tensor and
writes the step advantage into exactly one slot:

```python
# trainer/ppo/actor.py:340-407 (paraphrased)
kl_rewards = 0.0                            # kl_ctl=0 → no KL term
rewards = torch.zeros_like(input_ids, ...)  # [batch, seq_len]
rewards[:, seq_len - 2] = step_adv          # line 383
tot_rewards = rewards                       # line 407 (logged as final_reward)
```

The `[:, seq_len - 2]` index targets the position immediately before
the EOS — i.e. the last *response* token of the step. Everywhere else
in the per-token reward tensor is exactly zero. This is the moment a
group-normalized, length-normalized episode-level scalar becomes a
single per-token reward.

### Step 4 — GAE spreads that single-token reward across the response

`trainer/ppo/actor.py:392-403` runs masked GAE over the response tokens
with `discount=1.0, gae_lambda=1.0`. With $\lambda = \gamma = 1$ the
GAE recursion degenerates to a cumulative-from-the-right of the
per-token reward tensor, masked by `loss_mask`:

$$
A_t \;=\; \sum_{t' \ge t} r_{t'}\; \mathbb{1}[\text{loss\_mask}_{t'} = 1].
$$

Since the reward tensor is zero everywhere except at the last response
position, **every response token in step $j$ of episode $e$ ends up
with the same advantage**

$$
A_{t} \;=\; \frac{\hat r_e}{T_{\text{const}}} \;=\; \frac{\hat r_e}{10}.
$$

`adv_norm` is `null`, so no second normalization fires (the
`if self.adv_norm is not None:` block at `actor.py:417` is skipped).
The advantage broadcast is uniform along the response — there is no
intra-response credit assignment.

### Step 5 — the per-token GRPO surrogate

The actor update runs `ppo_n_epochs × ppo_n_minibatches = 6` passes per
train step, each calling `grpo_loss_fn` → `ppo_actor_loss_fn` in
`utils/functional/functional.py:144-258`. Per token $t$ in the response
window:

$$
\rho_t \;=\; \exp\!\big(\log\pi_\theta(a_t|s_t) - \log\pi_{\text{prox}}(a_t|s_t)\big),
$$
$$
\rho_t^{\text{clip}} \;=\; \text{clip}(\rho_t,\, 1-\varepsilon,\, 1+\varepsilon),
$$
$$
\text{pg}_t \;=\; \min\!\Big(\max(-A_t\rho_t,\, -A_t\rho_t^{\text{clip}}),\;
\operatorname{sign}(A_t)\cdot c\cdot |A_t|\Big),
$$

with $\varepsilon = $ `eps_clip` $ = 0.2$ and $c = $ `c_clip` $ = 3.0$.
The $\min(\cdot,\, c|A_t|)$ branch is the dual clip (functional.py:218-228).

### Step 6 — decoupled off-policy correction and the importance cap

Because `use_decoupled_loss: true`, the surrogate is reweighted by the
behaviour-policy ratio
$w_t = \exp(\log\pi_{\text{prox}} - \log\pi_{\text{behave}})$ at
`functional.py:229-242`. Tokens with $w_t > $ `behav_imp_weight_cap`
($=5.0$) are masked out of the gradient (`behav_mask`, line 231-235);
the rest contribute

$$
\text{pg}_t' \;=\; w_t \cdot \text{pg}_t.
$$

This is the entire `behav_imp_weight_cap=5.0` knob: it does **not** clip
$w_t$ to 5, it **drops the token** when $w_t > 5$. The cap-engagement
rate is logged as `behave_cap_ratio` (0 means no tokens were dropped).

### Step 7 — reduce to a scalar and backpropagate

`functional.py:244` reduces to a **per-token mean over the global
batch**:

$$
\mathcal L \;=\; \frac{1}{|\mathcal T|}
\sum_{t \in \mathcal T} \text{pg}_t' \cdot \mathbb 1[\text{loss\_mask}_t = 1],
$$

where $\mathcal T$ is the set of all response tokens across the batch.
This scalar is what `engine.train_batch` backwards. It is also what
`update/actor_loss/avg` logs.

### Step 8 — what the dashboard metrics actually measure

| wandb key | source | what it really is |
|---|---|---|
| `rollout/reward` | raw $r_e$ averaged over accepted episodes | the user-visible success rate before any normalization |
| `ppo_actor/final_reward/avg` | per-token reward tensor mean | $\approx 0$ by construction — $\hat r_e$ is group-mean-centred and only one of $\sim T \cdot$ tokens per step is non-zero |
| `ppo_actor/advantages/avg` | post-GAE advantage tensor mean | $\approx 0$ for the same reason |
| `ppo_actor/update/actor_loss/avg` | per-token surrogate mean | **the scalar that is actually being minimised** |
| `ppo_actor/update/approx_kl_abs/avg` | $\|\log\pi_\theta - \log\pi_{\text{old}}\|$ averaged over tokens | per-update KL the optimizer is producing |
| `ppo_actor/update/behave_approx_kl_abs/avg` | $\|\log\pi_\theta - \log\pi_{\text{behave}}\|$ averaged over tokens | gap to the behaviour (rollout-time) policy; should stay small |
| `ppo_actor/behave_cap_ratio` | fraction of tokens dropped by `behav_imp_weight_cap` | should be $\ll 0.05$; spikes indicate off-policy data is too stale |

### Summary in one line

`rewards (binary, episode-level)` → **group-normalize** →
`r̂_e ∈ ℝ` → **divide by $k = 10$ (constT10)** →
`step_adv ∈ ℝ` → **place on last response token, GAE-broadcast** →
`A_t = r̂_e / 10` (constant along response) → **clipped, dual-clipped,
decoupled-IS-weighted PPO surrogate** → **mean over all response
tokens** → scalar loss.

## Implementation

Toggle via `step_adv_const_length` in `areal/api/cli_args.py` and applied
in `areal/trainer/ppo/actor.py`:

- Default (`step_adv_const_length: null`): standard $1/|\tau_i|$
  normalization.
- Set to a positive value (e.g. `step_adv_const_length: 10.0`): use that
  constant as $k$. Paper uses $k=10$ throughout.

Configs that enable the fix are named `*_constT10.yaml`.

## RAFT++ as a baseline

The repository also implements RAFT++ as a contrasting off-policy
baseline. RAFT++ can be viewed as vanilla multi-step GRPO with the same
$1/|\tau_i|$ normalizer, but with group normalization disabled and the
group-relative advantage replaced by a success filter ($r > 0$). Only
successful trajectories contribute gradient, so RAFT++ effectively
performs behavior cloning on a rolling buffer of positives, and provides
no contrastive signal on below-average trajectories.

Toggle via `actor.actor_loss: fbc`. Decoupled importance sampling is
still applied to keep the off-policy gradient unbiased.

## Dual-clip

For negative-advantage tokens with exploded importance ratios, we keep a
PPO-style dual clip on the per-token contribution:

$$
f_{\text{dual}}(\rho) =
\begin{cases}
  \max(\rho \cdot \hat A,\, c \cdot \hat A) & \hat A < 0 \\
  \min(\rho \cdot \hat A,\, \text{clip}(\rho)\cdot \hat A) & \hat A \ge 0
\end{cases}
$$

with $c > 1$ controlled by `actor.c_clip` (default 3.0 in the paper).
The dual clip provides an absolute lower bound on the per-token
contribution and eliminates the tail risk introduced by rare
low-probability tokens with exploded importance-sampling ratios.

## DAPO-style dynamic sampling

Following [DAPO](https://arxiv.org/abs/2503.14476), AsyncWebRL drops the
reference-KL term and applies dynamic sampling: skip groups whose
trajectories all succeed or all fail, and gather 128 mixed trajectories
(16 groups) before launching training. This is configured by
`actor.dynamic_sampling: true` and `actor.dynamic_sampling_batch_size: 128`.

## Pointer to AReaL

The training loop, FSDP/Megatron engines, and inference backends are all
inherited from AReaL. For details on the broader RL infrastructure (e.g.
how `PPOTrainer` orchestrates rollout/update/refresh, what each engine
supports, how checkpointing works), see the
[AReaL documentation](https://inclusionai.github.io/AReaL/).
