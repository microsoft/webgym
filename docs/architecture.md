# How AReaL and WebGym Fit Together

AsyncWebRL is two layers with a clean seam between them. **AReaL** is the
distributed RL framework — it owns training, inference serving, and the RL
loop. **WebGym** is the web-agent layer — it owns the environment (real
browsers) and the logic that turns a task into a trajectory. They meet at a
single interface: AReaL's `RolloutWorkflow`.

The guiding idea: *AReaL never knows what a browser is, and WebGym never
manages GPUs or gradients.* Everything that crosses between them flows through
the workflow contract.

## Division of responsibilities

**AReaL provides (the framework):**

- `PPOTrainer` (`areal/trainer/rl_trainer.py`) — the controller that drives the
  whole loop: pull dataset items, submit them for rollout, collect trajectories,
  compute GRPO advantages, run the actor update, refresh inference weights.
- The **inference engine** (`RemoteSGLangEngine`) — SGLang servers that generate
  tokens, exposed to workflows as `engine.agenerate(...)`.
- The **actor engine** (`FSDPPPOActor`) — shards the policy and applies gradient
  updates.
- Supporting machinery — dataset loaders, the reward-function registry,
  checkpoint/recover, and stats logging.

**WebGym provides (the agent layer):**

- `WebGymWorkflow` (`areal/workflow/webgym_workflow.py`, a `RolloutWorkflow`) —
  runs one full multi-step browser episode per task.
- The **Omniboxes browser cluster** — the actual environment: headless Chromium
  instances behind an HTTP service (redis + master + node + instance servers),
  running on CPU nodes. Omniboxes is a separate service maintained at
  [`microsoft/webgym` (`webgym` branch)](https://github.com/microsoft/webgym/tree/webgym);
  see the
  [WebGym server quickstart](https://webgym.readthedocs.io/en/latest/server/quickstart_server.html)
  to deploy it.
- `webgym_reward_fn` and the judge model — score a finished episode.
- The entry point and config — `webgym/train.py` and `WebGymConfig(GRPOConfig)`,
  which wire the WebGym dataset, reward, and workflow into `PPOTrainer`.

## The contract: `RolloutWorkflow`

The single integration point is one method:

```python
class RolloutWorkflow(ABC):
    async def arun_episode(self, engine, data): ...
```

AReaL's rollout controller hands each task (`data`) to a `WebGymWorkflow`
instance together with a handle to the inference `engine`. The workflow's only
job is to turn that one task into one trajectory. Swapping in a different
environment — or a different agent loop — means writing a new workflow and
nothing else.

## What one rollout looks like

`WebGymWorkflow.arun_episode` runs an episode by alternating between the
**policy** (AReaL's SGLang engine, on GPU) and the **environment** (Omniboxes
browsers, on CPU):

1. **Allocate** a browser instance from the Omniboxes master (HTTP).
2. **Step loop**, until the agent answers, times out, or hits the step cap:
   - take a **screenshot** of the browser (the observation);
   - build a vision-language prompt (screenshot + history + task);
   - call `engine.agenerate(ModelRequest(...))` — the **policy** produces the
     next action as text;
   - parse the text into a browser command and **execute** it via Omniboxes
     (HTTP).
3. **Score** the finished episode with the judge, and broadcast that scalar
   reward across every token of the trajectory.
4. **Return** the multi-turn trajectory (token ids, log-probs, per-step reward)
   to AReaL.

```
                 ┌───────────────── one rollout (WebGymWorkflow) ─────────────────┐
   task ───▶     │  screenshot ──▶ build prompt ──▶ agenerate ──▶ parse ──▶ execute │ ──▶ trajectory
                 │     ▲  (Omniboxes / CPU)        (SGLang / GPU)      (Omniboxes)   │
                 │     └──────────────────── loop until answer ───────────────────┘ │
                 └─────────────────────────────────────────────────────────────────┘
   AReaL ◀── trajectories ── advantages ── FSDP actor update ── refresh SGLang weights ──▶ (repeat)
```

## Back in the trainer

AReaL collects the returned trajectories, forms GRPO groups, computes
advantages, runs the FSDP actor update, and pushes the refreshed weights back to
the SGLang servers. In the synchronous picture these stages run one after
another; AsyncWebRL overlaps them so rollout, update, and weight refresh proceed
concurrently — see [Async System Design](async_system.md).
