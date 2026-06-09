

# AsyncWebRL

**AsyncWebRL** is an asynchronous multi-step reinforcement-learning framework
for visual web agents. It pairs a fully async execution model with a
one-line GRPO loss fix that contracts trajectory length while preserving
task accuracy.

## What you get

- **Async system**: an everlasting rollout pool that keeps rollout workers
  alive across iteration boundaries and overlaps rollout, gradient update,
  and policy refresh end-to-end. **2.4–2.9× end-to-end training-throughput
  speedup** over the previously fastest open synchronous pipeline (WebGym).
- **Lightweight screenshot handling**: per-step image tensors stay out of
  the shared inter-worker data store; only lightweight references are
  routed through, avoiding the disk-spill path that hundreds of concurrent
  visual rollouts otherwise induce.
- **Decoupled off-policy correction**: the standard
  $\pi_\theta/\pi_{\mathrm{behave}}$ ratio is factored into a rollout-staleness
  factor and a current-update factor, with the PPO clip centered on the
  current-update factor only. Roughly halves the clip rate.
- **Constant $1/k$ loss aggregation**: replace the per-trajectory step-number
  normalizer $1/|\tau_i|$ in multi-step GRPO with a constant $1/k$
  (k = Easy-difficulty horizon = 10). Holding hardware and framework fixed,
  this loss-aggregation fix alone gives a **1.8× per-training-step speedup**
  over the standard $1/|\tau_i|$ loss.

## Where AsyncWebRL fits

AsyncWebRL is built on top of [AReaL](https://github.com/inclusionAI/AReaL),
the upstream distributed RL framework, and extends it with a
WebGym-specific async rollout pool, lightweight screenshot routing, the
decoupled-PPO loss path, and the constant-$1/k$ aggregation toggle. For
general framework concepts (FSDP/Megatron engines, SGLang/vLLM rollout
backends, checkpoint formats, alloc modes), refer to the
[AReaL documentation](https://inclusionai.github.io/AReaL/).
