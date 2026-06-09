

# Async System Design

AsyncWebRL changes two things on top of the synchronous multi-step rollout
pool used by [WebGym](https://arxiv.org/abs/2511.06715):

1. **An everlasting rollout pool** that keeps rollout workers continuously
   alive across iteration boundaries.
2. **Lightweight screenshot handling** that keeps per-step image tensors
   out of the shared inter-worker data store.

Together these two changes eliminate the GPU idle time spent waiting for
the slowest trajectory in a batch and the warm-up cost paid on hundreds
of browser sessions when a sync rollout pool is rebuilt each round.

## Process and GPU layout

AsyncWebRL runs as three **disaggregated** component groups, each a set of
separate processes pinned to its own GPUs. A single controller process
(`PPOTrainer` in `areal/trainer/rl_trainer.py`) launches all three through
the scheduler (`scheduler.type: ray`) and orchestrates the per-step loop;
the GPU workers themselves never share a device across roles.

1. **Train rollout** — SGLang inference servers that generate training
   trajectories. One server per GPU (`tp_size=1`), instantiated by
   `RolloutController` (`areal/infra/controller/rollout_controller.py`)
   under the scheduler role `rollout`.
2. **Update engine (actor)** — the FSDP trainer that consumes trajectories
   and applies gradient updates. The 8B policy is sharded across its GPUs
   (`FSDPPPOActor`, `areal/engine/fsdp_engine.py`), under the role `actor`.
3. **Test rollout** — a second, independent SGLang server group used only
   for evaluation, driven by a background daemon thread
   (`TestRolloutManager`, `areal/utils/test_rollout.py`) under the role
   `test-rollout`. It runs only when `test_rollout.enabled: true` and never
   blocks the train loop.

**GPU placement** is set by the `allocation_mode` string. The `+` operator
means *disaggregated* — each component gets its own dedicated GPUs (a `|`
would colocate them on shared GPUs). For example:

```yaml
allocation_mode: sglang[rollout]:d48+fsdp[actor]:d16
```

assigns **48 GPUs** to train-rollout SGLang servers (`dN` = data-parallel
server count) and **16 GPUs** to the FSDP update engine (`dM` = trainer
data-parallel ranks), totalling 64 GPUs (8 nodes × 8). Physical node
pinning is handled by `RayScheduler.setup_mixed_layout`
(`areal/infra/scheduler/ray.py`), which builds one Ray placement group per
role. Test-rollout GPUs are sized separately by `test_rollout.n_gpus` and
placed on free GPUs (`SchedulingStrategyType.separation`), distinct from
both train-rollout and the update engine.

| Component | Scheduler role | GPU-count field |
|---|---|---|
| Train rollout | `rollout` | `sglang[rollout]:dN` in `allocation_mode` |
| Update engine | `actor` | `fsdp[actor]:dM` in `allocation_mode` |
| Test rollout | `test-rollout` | `test_rollout.n_gpus` |

**How they communicate.** The controller fans out to all workers over RPC
(`scheduler.async_call_engine`). Trajectories flow from train-rollout to
the update engine through the rollout controller's pending-results queue
(completion signalled by an HTTP callback), with screenshots passed by
reference through the `PixelStore` Ray actor (see *Lightweight screenshot
handling* below). Refreshed weights reach the two rollout groups by two
different transports:

- **Update engine → train rollout**: an in-place **XCCL/NCCL broadcast**.
  FSDP rank 0 and all SGLang servers form one process group
  (`weight_update_mode: xccl`); rank 0 broadcasts each full parameter to
  the servers (`_update_weights_from_distributed`,
  `areal/engine/fsdp_engine.py`).
- **Update engine → test rollout**: **via disk**. Test rollout never joins
  the broadcast group; its thread scans for the latest completed
  checkpoint and hot-loads it (`update_weights_from_disk`), so evaluation
  always runs on a self-consistent saved snapshot.

**Async overlap across iterations.** Because the three groups occupy
distinct GPUs, they run concurrently:

- Train-rollout servers generate **continuously**, governed by the
  `StalenessManager` (`areal/infra/staleness_manager.py`): rollouts up to
  `rollout.max_head_offpolicyness` weight versions behind the current
  policy are accepted, so generation for steps *N+1*/*N+2* proceeds while
  step *N* trains. Stale trajectories are dropped at collection time.
- A weight refresh is bracketed by `rollout.pause()` / `rollout.resume()`
  around the broadcast, so the only serialized window is the parameter
  copy itself — rollout-vs-train *compute* fully overlaps.
- Test rollout overlaps the **entire** training loop on its own GPUs with
  no staleness coupling; it only asks the trainer to flush a checkpoint
  and otherwise never stalls training.

## Everlasting rollout pool

In a synchronous multi-step rollout pool, each training iteration:

1. Spawns N concurrent browser sessions
2. Waits until all of them finish (or hit the horizon)
3. Stops, refreshes policy weights, then respawns

The trailing tail of step (2) is GPU idle time; step (3) pays a per-round
warm-up cost on hundreds of browser sessions.

AsyncWebRL keeps workers continuously alive. When one episode ends, the
next begins immediately on the same worker without waiting for the rest
of the batch or for the next iteration. The parameter update of $\pi_\theta$
can happen at any time while rollout continues; new weights are broadcast
in place to the inference workers and the next rollout segment is sampled
under the updated policy.

The implementation lives in:

- `areal/infra/workflow_executor.py` — long-lived workflow executor
- `areal/infra/controller/rollout_controller.py` — `update_weights_xccl`
  callback and `update_weights_from_distributed`
- `areal/infra/remote_inf_engine.py` — inference engine that hot-swaps
  weights via NCCL/XCCL broadcast

The async mode is toggled by setting `cluster.n_inference_servers` to a
non-zero value (i.e. running rollout on dedicated inference workers, not
collocated with training).

## Lightweight screenshot handling

In a vanilla multi-step rollout pool, every screenshot generated by a
browser session is serialized through the shared RPC object store before
reaching the trainer. With tens of high-resolution screenshots per
trajectory and hundreds of concurrent rollouts, this saturates the store
and triggers a disk-spill fallback path, which erases the async benefit.

AsyncWebRL routes screenshots through a dedicated Ray actor instead. The
implementation lives in `areal/utils/pixel_store.py` (`PixelStore` actor,
`get_or_create_pixel_store`). Rollout workers store tensors keyed by an
opaque ID and pass only the key through the RPC object store
(`areal/workflow/webgym_workflow.py`). The trainer fetches the tensor
lazily from the `PixelStore` actor only when it needs to construct a
forward-pass batch (`areal/engine/fsdp_engine.py`).

The net effect: the RPC object store sees lightweight references
(strings + small metadata) instead of dozens of MB of pixel data per
step.

## Decoupled off-policy correction

Asynchronous execution shifts the sampling distribution: the policy that
generated a given token, $\pi_{\mathrm{behave}}$, is several updates behind
the policy $\pi_\theta$ the trainer is now updating. A standard PPO
importance-sampling ratio $\pi_\theta/\pi_{\mathrm{behave}}$ must capture
two quantities at once:

1. How much the policy has moved since the rollout was sampled
   (**rollout staleness**).
2. How much the optimizer has moved the policy during the current
   gradient update.

Clipping a single coupled ratio confounds the two, so rollout staleness
alone triggers many clip events and slows training.

AsyncWebRL adopts the decoupled-PPO factorization of
[Vanlioglu (2025)](https://arxiv.org/abs/2503.16252):

- Split the ratio into a **rollout-staleness factor**
  $\pi_{\mathrm{prox}}/\pi_{\mathrm{behave}}$ (enters the loss as an unclipped
  weight) and a **current-update factor** $\pi_\theta/\pi_{\mathrm{prox}}$
  (the only factor that gets clipped).
- $\pi_{\mathrm{prox}}$ is the policy snapshot at the start of the current
  update.

Implementation:

- `areal/utils/functional/functional.py` — `behav_imp_weight` and the
  decoupled-PPO loss path.
- `areal/utils/constants.py` — `ProxLogpMethod` enum.

Config toggles in `areal/api/cli_args.py`:

- `actor.recompute_logprob: true`
- `actor.use_decoupled_loss: true`
- `actor.prox_logp_method: recompute`

Empirically this roughly halves the clip rate relative to the coupled
formulation.
