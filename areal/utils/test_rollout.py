"""Test rollout manager for WebGym training.

Runs test-set evaluation on dedicated SGLang servers in a background thread,
independent of the training loop. Test rollout servers load model weights from
disk between evaluation rounds (no per-step weight sync).

Before each round, the manager scans the HF checkpoint directory for the latest
saved checkpoint and loads it if newer than the currently loaded one. If no
checkpoint exists yet, the initial (base) model is used.
"""

import getpass
import os
import re
import threading

import ray
import time
from copy import deepcopy
from dataclasses import dataclass

from areal.api.cli_args import (
    InferenceEngineConfig,
    SchedulingStrategy,
    SchedulingStrategyType,
    SGLangConfig,
)
from areal.api.io_struct import WeightUpdateMeta
from areal.api.workflow_api import WorkflowLike
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils.logging import getLogger

logger = getLogger("TestRollout")


@dataclass
class TestRolloutConfig:
    """Configuration for test-set rollout on dedicated GPUs.

    Test rollout is completely independent from training rollout: it always
    evaluates each task exactly once (group_size=1, no n_samples concept).

    Attributes:
        enabled: Whether to enable test-set rollout.
        n_gpus: Number of GPUs to dedicate to test rollout SGLang servers.
        max_new_tokens: Maximum generation length for test rollout.
        greedy: Whether to use greedy decoding for test rollout.
        max_concurrent_per_gpu: Maximum number of concurrent tasks per test
            rollout GPU. The global in-flight limit is
            ``max_concurrent_per_gpu * n_gpus``.
    """

    enabled: bool = False
    n_gpus: int = 4
    max_new_tokens: int = 3072
    greedy: bool = True
    max_tasks: int | None = None
    max_concurrent_per_gpu: int = 24
    log_interval_frac: float = 0.1
    base_model_checkpoint: str | None = None
    """Local HF checkpoint directory for the base model. Required because
    SGLang update_weights_from_disk fails with HF Hub paths on shared-GPU
    worker nodes. If no training checkpoint exists, test rollout loads from
    this path. Must be prepared manually before training starts."""


class TestRolloutManager:
    """Background test-set rollout on dedicated SGLang servers.

    Launches separate SGLang servers on free GPUs and runs perpetual test evaluation:
    1. Round 0: evaluate using the initial HF model (no save needed)
    2. Round N (N>=1): request weight save from training → load from disk → evaluate
    3. Log scores to wandb after each round
    """

    def __init__(
        self,
        config: TestRolloutConfig,
        eval_dataloader,
        workflow: WorkflowLike,
        workflow_kwargs: dict,
        initial_model_path: str,
        sglang_config: SGLangConfig,
        rollout_config: InferenceEngineConfig,
        allocation_mode,
        scheduler,
        experiment_name: str,
        trial_name: str,
        fileroot: str,
    ):
        self._config = config
        self._eval_dataloader = eval_dataloader
        self._workflow = workflow
        self._workflow_kwargs = workflow_kwargs
        self._initial_model_path = initial_model_path
        self._sglang_config = sglang_config
        self._rollout_config = rollout_config
        self._allocation_mode = allocation_mode
        self._scheduler = scheduler
        self._experiment_name = experiment_name
        self._trial_name = trial_name
        self._fileroot = fileroot

        # Threading coordination
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Track the last loaded checkpoint's global_step to avoid redundant reloads
        self._loaded_global_step: int = -1

        # State
        self._eval_round = 0
        self._eval_controller = None

        # Signal training loop to save after test rollout round completes
        self._save_requested = threading.Event()

        # Running reward from current/latest test rollout round (thread-safe)
        self._lock = threading.Lock()
        self._running_reward: float | None = None
        self._running_n_tasks: int = 0
        self._running_n_failed: int = 0
        self._running_num_steps: float = 0.0
        self._running_progress: float = 0.0
        self._running_avg_resp_tokens: float = 0.0
        self._running_max_resp_tokens: float = 0.0
        self._running_entropy: float = 0.0
        # Success/failure subset stats — only num_steps / avg_resp / entropy
        # are split, matching what the workflow can report. Each is a dict
        # with keys: num_steps, avg_resp_tokens, entropy.
        self._running_succ: dict[str, float] = {}
        self._running_fail: dict[str, float] = {}
        self._running_diff: dict[str, dict] = {}  # cat -> {reward, num_steps, n_tasks}

        # Final reward from last completed round (for trajectory_reward)
        self._iteration_reward: float | None = None
        self._iteration_round: int = -1
        self._iteration_num_steps: float = 0.0
        self._iteration_n_tasks: int = 0
        self._iteration_n_failed: int = 0
        self._iteration_avg_resp_tokens: float = 0.0
        self._iteration_max_resp_tokens: float = 0.0
        self._iteration_entropy: float = 0.0
        self._iteration_succ: dict[str, float] = {}
        self._iteration_fail: dict[str, float] = {}
        self._iteration_diff: dict[str, dict] = {}  # cat -> {reward, num_steps, n_tasks}

        # Blocked website tracking: website -> rounds_remaining
        # Websites with is_blocked=True are skipped for the next 2 rounds.
        self._blocked_websites: dict[str, int] = {}

        # HF checkpoint directory: {fileroot}/checkpoints/{user}/{experiment}/{trial}/default/
        self._checkpoint_root = os.path.join(
            fileroot,
            "checkpoints",
            getpass.getuser(),
            experiment_name,
            trial_name,
            "default",
        )

    def start(self):
        """Launch the background test rollout thread."""
        self._thread = threading.Thread(
            target=self._eval_loop, name="test-rollout", daemon=True
        )
        self._thread.start()
        logger.info("Test rollout thread started")

    def stop(self):
        """Signal the test rollout thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=120)
            logger.info("Test rollout thread stopped")

    def should_save(self) -> bool:
        """Check if test rollout has completed a round and wants a checkpoint save."""
        return self._save_requested.is_set()

    def clear_save_request(self):
        """Clear the save-requested flag after the training loop has saved."""
        self._save_requested.clear()

    def get_step_metrics(self) -> dict | None:
        """Get running test rollout metrics for step_reward logging.

        Called by training loop after each step. Returns None if no test
        rollout data is available yet.
        """
        with self._lock:
            if self._running_reward is None:
                return None
            metrics = {
                "test_rollout/step/reward": self._running_reward,
                "test_rollout/step/n_tasks": self._running_n_tasks,
                "test_rollout/step/n_failed": self._running_n_failed,
                "test_rollout/step/num_steps": self._running_num_steps,
                "test_rollout/step/progress": self._running_progress,
                "test_rollout/step/avg_response_tokens_per_step": self._running_avg_resp_tokens,
                "test_rollout/step/max_response_tokens_per_step": self._running_max_resp_tokens,
                "test_rollout/step/entropy_avg": self._running_entropy,
            }
            for outcome, d in (("success", self._running_succ),
                               ("failure", self._running_fail)):
                if d:
                    metrics[f"test_rollout/step/{outcome}/num_steps"] = d.get("num_steps", 0.0)
                    metrics[f"test_rollout/step/{outcome}/avg_response_tokens_per_step"] = d.get("avg_resp_tokens", 0.0)
                    metrics[f"test_rollout/step/{outcome}/entropy_avg"] = d.get("entropy", 0.0)
            for cat, d in self._running_diff.items():
                metrics[f"test_rollout/step/{cat}/reward"] = d["reward"]
                metrics[f"test_rollout/step/{cat}/num_steps"] = d["num_steps"]
                metrics[f"test_rollout/step/{cat}/n_tasks"] = d["n_tasks"]
            return metrics

    def get_iteration_metrics(self) -> dict | None:
        """Get final metrics from the last completed test rollout round.

        Called by training loop when should_save() is True.
        Returns None if no round has completed.
        """
        with self._lock:
            if self._iteration_reward is None:
                return None
            metrics = {
                "test_rollout/trajectory/reward": self._iteration_reward,
                "test_rollout/trajectory/round": self._iteration_round,
                "test_rollout/trajectory/num_steps": self._iteration_num_steps,
                "test_rollout/trajectory/n_tasks": self._iteration_n_tasks,
                "test_rollout/trajectory/n_failed": self._iteration_n_failed,
                "test_rollout/trajectory/avg_response_tokens_per_step": self._iteration_avg_resp_tokens,
                "test_rollout/trajectory/max_response_tokens_per_step": self._iteration_max_resp_tokens,
                "test_rollout/trajectory/entropy_avg": self._iteration_entropy,
            }
            for outcome, d in (("success", self._iteration_succ),
                               ("failure", self._iteration_fail)):
                if d:
                    metrics[f"test_rollout/trajectory/{outcome}/num_steps"] = d.get("num_steps", 0.0)
                    metrics[f"test_rollout/trajectory/{outcome}/avg_response_tokens_per_step"] = d.get("avg_resp_tokens", 0.0)
                    metrics[f"test_rollout/trajectory/{outcome}/entropy_avg"] = d.get("entropy", 0.0)
            for cat, d in self._iteration_diff.items():
                metrics[f"test_rollout/trajectory/{cat}/reward"] = d["reward"]
                metrics[f"test_rollout/trajectory/{cat}/num_steps"] = d["num_steps"]
                metrics[f"test_rollout/trajectory/{cat}/n_tasks"] = d["n_tasks"]
            return metrics

    def _find_latest_checkpoint(self) -> tuple[str | None, int]:
        """Find the latest HF checkpoint in the default save directory.

        Only considers checkpoints that have a `.save_complete` marker file,
        which is written by FSDPEngine after all safetensors files are flushed.
        This prevents loading a partially-written checkpoint.

        Returns (path, global_step) or (None, -1) if no checkpoint exists.
        """
        if not os.path.isdir(self._checkpoint_root):
            return None, -1

        best_step = -1
        best_path = None
        pattern = re.compile(r"globalstep(\d+)$")

        for entry in os.listdir(self._checkpoint_root):
            m = pattern.search(entry)
            if m:
                candidate = os.path.join(self._checkpoint_root, entry)
                if not os.path.isfile(os.path.join(candidate, ".save_complete")):
                    continue  # Still being written
                step = int(m.group(1))
                if step > best_step:
                    best_step = step
                    best_path = candidate

        return best_path, best_step

    def _eval_loop(self):
        """Main test rollout loop running in background thread.

        Retries indefinitely on failure with exponential backoff (up to 5 min).
        This handles transient issues like Ray Placement Group timeouts on
        fresh pods without requiring a full training restart.
        """
        backoff = 30  # seconds
        max_backoff = 300
        while not self._stop_event.is_set():
            try:
                self._eval_loop_inner()
                return  # clean exit (stop_event set)
            except Exception:
                logger.exception(
                    "Test rollout thread crashed, retrying in %ds...", backoff
                )
                # Clean up the failed controller if it exists
                if self._eval_controller is not None:
                    try:
                        self._eval_controller.destroy()
                    except Exception:
                        logger.warning(
                            "Failed to destroy eval controller", exc_info=True
                        )
                    self._eval_controller = None
                # Also clear any partial "test-rollout" worker group left
                # in the scheduler. On the first crash (e.g. actor socket
                # closed during initialize) the controller handle above is
                # None but the scheduler has already registered the group,
                # so every retry trips "Worker group already exists".
                try:
                    self._scheduler.delete_workers("test-rollout")
                except Exception:
                    logger.debug(
                        "delete_workers('test-rollout') ignored", exc_info=True
                    )
                # Wait with backoff, checking stop_event periodically
                deadline = time.monotonic() + backoff
                while time.monotonic() < deadline:
                    if self._stop_event.wait(timeout=5):
                        return
                backoff = min(backoff * 2, max_backoff)

    def _eval_loop_inner(self):
        """Inner test rollout loop with actual logic."""
        # Step 1: Launch test rollout SGLang servers
        logger.info("Launching %d test rollout SGLang servers...", self._config.n_gpus)
        controller = self._launch_eval_controller()
        self._eval_controller = controller
        logger.info("Test rollout SGLang servers ready")

        while not self._stop_event.is_set():
            round_start = time.monotonic()
            logger.info("=== Test rollout round %d starting ===", self._eval_round)

            # Check for the latest HF checkpoint saved by training.
            # Retry with backoff: checkpoint save (DCP) may still be in progress
            # when a new round starts, causing a race condition where the round
            # misses the latest checkpoint and reuses stale weights.
            _CKPT_WAIT_MAX = 120  # seconds
            _CKPT_WAIT_INTERVAL = 10  # seconds
            ckpt_path, ckpt_step = self._find_latest_checkpoint()

            if (
                ckpt_path is None or ckpt_step <= self._loaded_global_step
            ) and self._loaded_global_step >= 0:
                # No new checkpoint yet but we've already loaded one before.
                # Training may still be saving — wait and retry.
                logger.info(
                    "No new checkpoint found (latest=%d, loaded=%d). "
                    "Waiting up to %ds for checkpoint save to complete...",
                    ckpt_step,
                    self._loaded_global_step,
                    _CKPT_WAIT_MAX,
                )
                _wait_start = time.monotonic()
                while time.monotonic() - _wait_start < _CKPT_WAIT_MAX:
                    if self._stop_event.is_set():
                        return
                    time.sleep(_CKPT_WAIT_INTERVAL)
                    ckpt_path, ckpt_step = self._find_latest_checkpoint()
                    if ckpt_path is not None and ckpt_step > self._loaded_global_step:
                        logger.info(
                            "New checkpoint found after %.0fs wait: step %d",
                            time.monotonic() - _wait_start,
                            ckpt_step,
                        )
                        break
                else:
                    logger.warning(
                        "No new checkpoint after %ds wait (latest=%d, loaded=%d). "
                        "Reusing current weights.",
                        _CKPT_WAIT_MAX,
                        ckpt_step,
                        self._loaded_global_step,
                    )

            if ckpt_path is not None and ckpt_step > self._loaded_global_step:
                # New checkpoint available — load it
                load_path = ckpt_path
                load_step = ckpt_step
            elif ckpt_path is None and self._loaded_global_step < 0:
                # No training checkpoint yet — load from pre-saved base model
                # checkpoint. SGLang update_weights_from_disk fails with HF Hub
                # paths on shared-GPU worker nodes, so a local copy is required.
                base_ckpt = self._config.base_model_checkpoint
                if base_ckpt is None or not os.path.isdir(base_ckpt):
                    raise RuntimeError(
                        f"No training checkpoint found and base_model_checkpoint "
                        f"is not set or does not exist: {base_ckpt!r}. "
                        f"SGLang update_weights_from_disk fails with HF Hub paths "
                        f"on shared-GPU worker nodes. Please save the base model "
                        f"as a local HF checkpoint and set "
                        f"test_rollout.base_model_checkpoint to its path. "
                        f"Example: python -c \"from transformers import "
                        f"AutoModelForCausalLM; m = AutoModelForCausalLM"
                        f".from_pretrained('{self._initial_model_path}', "
                        f"torch_dtype='auto'); m.save_pretrained('/path/to/save')\""
                    )
                load_path = base_ckpt
                load_step = 0
            else:
                load_path = None
                load_step = self._loaded_global_step

            if load_path is not None:
                logger.info(
                    "Loading weights from %s (step %d)", load_path, load_step
                )
                t0 = time.monotonic()
                meta = WeightUpdateMeta(
                    type="disk",
                    path=load_path,
                    clear_checkpoint_after_load=False,
                    skip_name_resolve=True,
                )
                controller._collective_rpc("update_weights_from_disk", meta=meta)
                self._loaded_global_step = load_step
                logger.info(
                    "Test rollout servers loaded weights (step %d): %.1fs",
                    load_step,
                    time.monotonic() - t0,
                )
            else:
                logger.info(
                    "Checkpoint step %d already loaded, reusing",
                    self._loaded_global_step,
                )

            # Submit all test tasks
            eval_step = max(self._loaded_global_step, 0)

            # Set version to loaded global_step so trajectory dumps go to
            # test-rollout/{step}/ — this survives resume without overwriting
            # previous rounds (unlike eval_round which resets to 0 on restart).
            controller.set_version(eval_step)
            logger.info(
                "Submitting test rollout tasks (round=%d, model_step=%d)...",
                self._eval_round,
                eval_step,
            )

            # Collect all tasks, sort hard→easy so hard tasks start early
            # and results arrive while easy tasks are still in-flight.
            all_items = []
            max_tasks = self._config.max_tasks
            for data in self._eval_dataloader:
                if self._stop_event.is_set():
                    break
                for item in data:
                    all_items.append(item)
                    if max_tasks is not None and len(all_items) >= max_tasks:
                        break
                if max_tasks is not None and len(all_items) >= max_tasks:
                    break

            # Decrement blocklist counters for this round
            self._blocked_websites = {
                w: n - 1
                for w, n in self._blocked_websites.items()
                if n > 1
            }
            blocked_now = set(self._blocked_websites.keys())
            if blocked_now:
                logger.info(
                    "Blocking %d websites for this round: %s",
                    len(blocked_now),
                    sorted(blocked_now)[:10],
                )

            # Filter out blocked websites, then sort hard→easy
            filtered_items = [
                item for item in all_items
                if item.get("website", "") not in blocked_now
            ]
            n_filtered = len(all_items) - len(filtered_items)
            if n_filtered > 0:
                logger.info("Filtered %d tasks from blocked websites", n_filtered)

            filtered_items.sort(
                key=lambda x: int(x.get("difficulty") or 0), reverse=True
            )

            # Map episode_id → website for blocked detection at collect time
            task_website_map: dict[str, str] = {
                item.get("task_id", ""): item.get("website", "")
                for item in filtered_items
            }

            # Submit tasks with concurrency control: keep at most
            # max_concurrent_per_gpu * n_gpus tasks in-flight at any time.
            max_inflight = (
                self._config.max_concurrent_per_gpu * self._config.n_gpus
            )
            total_tasks = len(filtered_items)
            logger.info(
                "Submitting %d test rollout tasks (max_inflight=%d)...",
                total_tasks, max_inflight,
            )

            from areal.infra.rpc.rtensor import RTensor

            log_interval = max(1, int(total_tasks * self._config.log_interval_frac))
            rewards = []
            num_steps_list = []
            avg_resp_list: list[float] = []
            max_resp_list: list[float] = []
            entropy_list: list[float] = []
            # Success / failure sub-buckets (success = reward > 0.5).
            succ_data: dict[str, list[float]] = {
                "num_steps": [], "avg_resp_tokens": [], "entropy": [],
            }
            fail_data: dict[str, list[float]] = {
                "num_steps": [], "avg_resp_tokens": [], "entropy": [],
            }
            diff_data: dict[str, dict[str, list]] = {
                "easy": {"rewards": [], "num_steps": []},
                "medium": {"rewards": [], "num_steps": []},
                "hard": {"rewards": [], "num_steps": []},
            }
            submitted = 0
            collected = 0

            # Initial burst: submit up to max_inflight
            while submitted < total_tasks and submitted - collected < max_inflight:
                if self._stop_event.is_set():
                    break
                controller.submit(
                    filtered_items[submitted],
                    self._workflow,
                    self._workflow_kwargs,
                    group_size=1,  # Test rollout always evaluates once per task
                    is_eval=True,
                )
                submitted += 1

            cnt = submitted  # for compatibility with downstream code

            if self._stop_event.is_set():
                break

            logger.info(
                "Waiting for %d test rollout tasks to complete "
                "(%d submitted so far)...",
                total_tasks, submitted,
            )

            # Collect results and submit more tasks as slots free up.
            # Grace period: when remaining tasks < _TAIL_THRESHOLD, start a
            # 2-min timer. If the timer expires before all results arrive,
            # treat uncollected tasks as failures and move on.
            _TAIL_THRESHOLD = 10
            _GRACE_PERIOD_SECS = 120
            grace_deadline: float | None = None

            while collected < submitted:
                if self._stop_event.is_set():
                    break

                remaining_tasks = submitted - collected
                # Enter grace period when we're in the tail
                if (
                    remaining_tasks <= _TAIL_THRESHOLD
                    and submitted >= total_tasks
                    and grace_deadline is None
                ):
                    grace_deadline = time.monotonic() + _GRACE_PERIOD_SECS
                    logger.info(
                        "Entering grace period: %d tasks remaining, "
                        "deadline in %ds",
                        remaining_tasks,
                        _GRACE_PERIOD_SECS,
                    )

                # Check grace period expiry
                if grace_deadline is not None and time.monotonic() > grace_deadline:
                    n_abandoned = submitted - collected
                    logger.warning(
                        "Grace period expired: abandoning %d remaining "
                        "tasks as failures",
                        n_abandoned,
                    )
                    collected = submitted  # force exit
                    break

                # Use short timeout during grace period so we can check expiry.
                # Always wait for just 1 result to keep the pipeline warm —
                # waiting for large batches drains workers before submitting
                # new tasks.
                wait_timeout = (
                    10.0 if grace_deadline is not None else 30.0
                )
                try:
                    batch = controller.wait(
                        1,
                        timeout=wait_timeout,
                        raise_timeout=True,
                    )
                except TimeoutError:
                    # Poll timeout — loop back to check grace period / stop
                    continue

                # Eagerly drain any additional ready results (non-blocking)
                while True:
                    try:
                        extra = controller.wait(1, timeout=0.01, raise_timeout=True)
                        batch.extend(extra)
                    except TimeoutError:
                        break

                # Submit new tasks to fill freed slots
                new_slots = len(batch)
                while (
                    submitted < total_tasks
                    and new_slots > 0
                    and not self._stop_event.is_set()
                ):
                    controller.submit(
                        filtered_items[submitted],
                        self._workflow,
                        self._workflow_kwargs,
                        group_size=1,  # Test rollout always evaluates once per task
                        is_eval=True,
                    )
                    submitted += 1
                    new_slots -= 1
                cnt = submitted  # update total count

                # Free pixel keys for this batch immediately — images must not
                # accumulate in PixelRefStore across the entire test round.
                batch_pixel_keys = []
                for traj in batch:
                    if traj is not None and "_image_store_keys" in traj:
                        batch_pixel_keys.extend(traj["_image_store_keys"])
                if batch_pixel_keys:
                    try:
                        from areal.utils.pixel_store import get_or_create_pixel_store

                        _store = get_or_create_pixel_store()
                        ray.get(_store.free.remote(batch_pixel_keys))
                    except Exception:
                        logger.warning(
                            "Failed to free pixel keys for batch", exc_info=True
                        )

                for traj in batch:
                    if traj is not None and "rewards" in traj:
                        # Check blocked status first (for blocklist + exclusion)
                        traj_blocked = False
                        if "is_blocked" in traj:
                            traj_blocked = bool(
                                RTensor.localize(traj["is_blocked"]).max().item()
                            )
                            if traj_blocked:
                                task_id = traj.get("_dataset_task_id", "")
                                website = task_website_map.get(str(task_id), "")
                                if website:
                                    self._blocked_websites[website] = 2

                        # Exclude blocked from reward stats (align with
                        # visualize_results.py: denominator = non-crashed
                        # AND non-blocked)
                        if not traj_blocked:
                            reward_tensor = RTensor.localize(traj["rewards"])
                            r = reward_tensor.float().max().item()
                            rewards.append(r)
                            # Per-difficulty breakdown
                            if "difficulty" in traj:
                                d = RTensor.localize(traj["difficulty"]).max().item()
                                cat = "easy" if d <= 3 else ("medium" if d <= 6 else "hard")
                                diff_data[cat]["rewards"].append(r)
                                if num_steps_list:
                                    diff_data[cat]["num_steps"].append(num_steps_list[-1])
                        if "horizon" in traj:
                            h = RTensor.localize(traj["horizon"])
                            ns = h.numel()
                        elif "attention_mask" in traj:
                            am = RTensor.localize(traj["attention_mask"])
                            ns = am.shape[0]
                        else:
                            ns = 1
                        num_steps_list.append(ns)
                        # Per-episode response-token + entropy stats (workflow
                        # wraps each in a single-element list).
                        _avg_resp_val: float | None = None
                        _avg_resp = traj.get("_avg_response_tokens_per_step")
                        if isinstance(_avg_resp, list) and _avg_resp:
                            _avg_resp_val = float(_avg_resp[0])
                            avg_resp_list.append(_avg_resp_val)
                        _max_resp = traj.get("_max_response_tokens_per_step")
                        if isinstance(_max_resp, list) and _max_resp:
                            max_resp_list.append(float(_max_resp[0]))
                        _entropy_val: float | None = None
                        _entropy = traj.get("_avg_entropy_per_token")
                        if isinstance(_entropy, list) and _entropy:
                            _entropy_val = float(_entropy[0])
                            entropy_list.append(_entropy_val)
                        # Success / failure bucket (only for non-blocked
                        # episodes that contributed to `rewards`).
                        if not traj_blocked:
                            bucket = succ_data if r > 0.5 else fail_data
                            bucket["num_steps"].append(float(ns))
                            if _avg_resp_val is not None:
                                bucket["avg_resp_tokens"].append(_avg_resp_val)
                            if _entropy_val is not None:
                                bucket["entropy"].append(_entropy_val)
                collected += len(batch)

                # Update running metrics (read by training loop for step_reward)
                if rewards:
                    inc_reward = sum(rewards) / len(rewards)
                    inc_steps = sum(num_steps_list) / len(num_steps_list)
                    n_failed = collected - len(rewards)
                    inc_diff = {
                        cat: {
                            "reward": sum(d["rewards"]) / len(d["rewards"]),
                            "num_steps": (
                                sum(d["num_steps"]) / len(d["num_steps"])
                                if d["num_steps"] else 0.0
                            ),
                            "n_tasks": len(d["rewards"]),
                        }
                        for cat, d in diff_data.items()
                        if d["rewards"]
                    }
                    inc_avg_resp = (
                        sum(avg_resp_list) / len(avg_resp_list) if avg_resp_list else 0.0
                    )
                    inc_max_resp = (
                        sum(max_resp_list) / len(max_resp_list) if max_resp_list else 0.0
                    )
                    inc_entropy = (
                        sum(entropy_list) / len(entropy_list) if entropy_list else 0.0
                    )

                    def _bucket_means(d: dict[str, list[float]]) -> dict[str, float]:
                        return {
                            k: (sum(v) / len(v) if v else 0.0)
                            for k, v in d.items()
                        }

                    inc_succ = _bucket_means(succ_data)
                    inc_fail = _bucket_means(fail_data)
                    with self._lock:
                        self._running_reward = inc_reward
                        self._running_n_tasks = len(rewards)
                        self._running_n_failed = n_failed
                        self._running_num_steps = inc_steps
                        self._running_progress = collected / cnt
                        self._running_avg_resp_tokens = inc_avg_resp
                        self._running_max_resp_tokens = inc_max_resp
                        self._running_entropy = inc_entropy
                        self._running_succ = inc_succ
                        self._running_fail = inc_fail
                        self._running_diff = inc_diff

                    elapsed = time.monotonic() - round_start
                    if collected % log_interval < len(batch):
                        logger.info(
                            "Test rollout round %d progress: reward=%.4f, "
                            "num_steps=%.1f, completed=%d/%d, failed=%d, "
                            "elapsed=%.1fs, model_step=%d",
                            self._eval_round,
                            inc_reward,
                            inc_steps,
                            len(rewards),
                            cnt,
                            n_failed,
                            elapsed,
                            eval_step,
                        )

            # Final summary
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            avg_num_steps = (
                sum(num_steps_list) / len(num_steps_list)
                if num_steps_list
                else 0.0
            )
            n_completed = len(rewards)
            n_failed = cnt - n_completed
            round_time = time.monotonic() - round_start

            logger.info(
                "Test rollout round %d complete: reward=%.4f, "
                "num_steps=%.1f, completed=%d/%d, failed=%d, "
                "time=%.1fs, model_step=%d",
                self._eval_round,
                avg_reward,
                avg_num_steps,
                n_completed,
                cnt,
                n_failed,
                round_time,
                eval_step,
            )

            # Store iteration reward and signal training loop
            final_diff = {
                cat: {
                    "reward": sum(d["rewards"]) / len(d["rewards"]),
                    "num_steps": (
                        sum(d["num_steps"]) / len(d["num_steps"])
                        if d["num_steps"] else 0.0
                    ),
                    "n_tasks": len(d["rewards"]),
                }
                for cat, d in diff_data.items()
                if d["rewards"]
            }
            final_avg_resp = (
                sum(avg_resp_list) / len(avg_resp_list) if avg_resp_list else 0.0
            )
            final_max_resp = (
                sum(max_resp_list) / len(max_resp_list) if max_resp_list else 0.0
            )
            final_entropy = (
                sum(entropy_list) / len(entropy_list) if entropy_list else 0.0
            )
            final_succ = {
                k: (sum(v) / len(v) if v else 0.0) for k, v in succ_data.items()
            }
            final_fail = {
                k: (sum(v) / len(v) if v else 0.0) for k, v in fail_data.items()
            }
            with self._lock:
                self._iteration_reward = avg_reward
                self._iteration_round = self._eval_round
                self._iteration_num_steps = avg_num_steps
                self._iteration_n_tasks = n_completed
                self._iteration_n_failed = n_failed
                self._iteration_avg_resp_tokens = final_avg_resp
                self._iteration_max_resp_tokens = final_max_resp
                self._iteration_entropy = final_entropy
                self._iteration_succ = final_succ
                self._iteration_fail = final_fail
                self._iteration_diff = final_diff
            self._save_requested.set()
            logger.info(
                "Requested checkpoint save from training loop (round %d)",
                self._eval_round,
            )

            self._eval_round += 1

        # Cleanup
        if self._eval_controller is not None:
            try:
                self._eval_controller.destroy()
            except Exception:
                logger.exception("Error destroying test rollout controller")

    def _launch_eval_controller(self):
        """Create a RolloutController with standalone test rollout SGLang servers."""
        # Build test-rollout-specific rollout config.
        # Test rollout is independent from training: always group_size=1,
        # no n_samples, no staleness control.
        eval_config = deepcopy(self._rollout_config)
        eval_config.max_head_offpolicyness = int(1e12)  # No staleness control
        global_concurrent = self._config.max_concurrent_per_gpu * self._config.n_gpus
        eval_config.max_concurrent_rollouts = global_concurrent
        eval_config.max_concurrent_per_worker = self._config.max_concurrent_per_gpu
        eval_config.n_samples = 1  # Decouple from training n_samples
        eval_config.consumer_batch_size = global_concurrent
        # Separation strategy: let Ray place on free GPUs
        eval_config.scheduling_strategy = SchedulingStrategy(
            type=SchedulingStrategyType.separation
        )
        eval_config.setup_timeout = 1800  # 30 min for first launch

        # Build server args for test rollout SGLang
        eval_sglang_config = deepcopy(self._sglang_config)
        server_args = SGLangConfig.build_args(
            sglang_config=eval_sglang_config,
            tp_size=1,  # 1 GPU per server for 8B model
            base_gpu_id=0,
        )

        # Create controller
        controller = RemoteSGLangEngine.as_controller(eval_config, self._scheduler)

        # Build a minimal alloc_mode for initialize()
        # We need gen.dp_size = n_gpus and gen_instance_size = 1
        eval_alloc = deepcopy(self._allocation_mode)
        eval_alloc.gen.data_parallel_size = self._config.n_gpus
        eval_alloc.gen.tensor_parallel_size = 1
        eval_alloc.gen.pipeline_parallel_size = 1

        controller.initialize(
            role="test-rollout",
            alloc_mode=eval_alloc,
            server_args=server_args,
        )

        return controller
