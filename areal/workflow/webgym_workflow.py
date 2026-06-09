"""WebGym multi-turn vision-language workflow for AReaL.

This workflow combines:
- Multi-turn token accumulation from ``MultiTurnWorkflow``
- Vision input handling from ``VisionRLVRWorkflow``
- Live browser environment interaction via Omniboxes ``MasterClient``

Each episode: allocate browser → (screenshot → generate action → execute) × N → reward.
"""

import asyncio
import itertools
import json
import logging as _logging
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests
import torch
import urllib3
from PIL import Image
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal import workflow_context
from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.image import image2base64

# Suppress SSL warnings for Omniboxes HTTPS connections
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger("WebGymWorkflow")


def _is_omniboxes_exception(exc: BaseException) -> bool:
    """True if exc came from an Omniboxes/browser-side call.

    Covers `requests` HTTP errors (covers 401/503/timeouts from any of
    `_allocate_instance`, `_navigate`, `_take_screenshot`, `_get_metadata`,
    `_execute_command`) plus the two `RuntimeError` strings raised after
    retry exhaustion. False for model-generation, parsing, and other
    Python-side errors raised inside `_arun_episode_impl`.
    """
    if isinstance(exc, requests.exceptions.RequestException):
        return True
    msg = str(exc)
    return (
        "Failed to allocate browser instance" in msg
        or "Screenshot failed:" in msg
    )


def release_all_instances(host_ip: str, master_port: int, api_key: str = "") -> None:
    """Release ALL browser instances on the Omniboxes server.

    Called before training starts to ensure a clean slate.
    """
    protocol = "https" if master_port == 443 else "http"
    base_url = f"{protocol}://{host_ip}:{master_port}"
    headers = {"x-api-key": api_key} if api_key else {}

    # 1. Get server info to discover all in-use instances
    try:
        resp = requests.get(
            f"{base_url}/info", headers=headers, verify=False, timeout=30
        )
        resp.raise_for_status()
        info = resp.json()
    except Exception:
        logger.warning(
            "Could not fetch server info, skipping instance release", exc_info=True
        )
        return

    # 2. Collect instances from the response
    instances: list[dict] = []
    if "nodes" in info:
        for node in info["nodes"]:
            node_hash = node.get("hash", node.get("url", ""))
            for inst_id in node.get("instances", []):
                instances.append({"instance_id": inst_id, "node": node_hash})
    elif "in_use" in info:
        for inst_id in info["in_use"]:
            instances.append({"instance_id": str(inst_id)})

    if not instances:
        logger.info("No instances to release.")
        return

    logger.info("Releasing %d instances...", len(instances))

    # 3. Release with controlled parallelism
    max_workers = min(20, len(instances))
    successful, failed = 0, 0

    def _reset(inst: dict) -> bool:
        try:
            r = requests.post(
                f"{base_url}/reset",
                params=inst,
                headers=headers,
                verify=False,
                timeout=30,
            )
            return r.status_code == 200
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_reset, inst): inst for inst in instances}
        for fut in as_completed(futures):
            if fut.result():
                successful += 1
            else:
                failed += 1

    logger.info("Instance release done: %d released, %d failed.", successful, failed)


class WebGymWorkflow(RolloutWorkflow):
    """Multi-turn vision-language workflow with live browser interaction.

    At each step the workflow:
    1. Takes a screenshot of the browser
    2. Builds a conversation prompt via WebGym's ``ContextManager``
    3. Generates action tokens via AReaL's inference engine
    4. Parses the generated text into a browser command
    5. Executes the command in the browser
    6. Repeats until the agent submits an answer or hits ``max_steps``

    The accumulated token sequence (with logprobs, versions, loss_mask)
    is returned for PPO training, along with multi-modal image data.
    """

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        processor: AutoProcessor | str,
        train_difficulty_max_steps: dict[str, int] | None = None,
        test_difficulty_max_steps: dict[str, int] | None = None,
        max_steps_schedule: dict[str, Any] | None = None,
        turn_discount: float = 1.0,
        env_config: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
        openai_config: dict[str, Any] | None = None,
    ):
        # --- Tokenizer / Processor ---
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer

        if isinstance(processor, str):
            processor = AutoProcessor.from_pretrained(processor)
        self.processor = processor

        # --- Generation config ---
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)

        # --- Reward ---
        self.reward_fn = reward_fn
        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)

        # --- Environment config ---
        env_config = env_config or {}
        self.host_ip = env_config.get("host_ip", "localhost")
        self.master_port = env_config.get("master_port", 7000)
        self.cpu_cluster_token = env_config.get("cpu_cluster_token", "")
        if not self.cpu_cluster_token:
            token_env_var = env_config.get(
                "cpu_cluster_token_env_var", "CPU_CLUSTER_TOKEN"
            )
            self.cpu_cluster_token = os.environ.get(token_env_var, "")
        self.interaction_mode = env_config.get("interaction_mode", "coordinates")
        self.operation_timeout = env_config.get("operation_timeout", 120)
        self.max_retries = env_config.get("max_retries", 2)
        self.instance_lifetime_mins = env_config.get("instance_lifetime_mins", 60)
        self.episode_timeout_minutes = env_config.get("episode_timeout_minutes", 45)
        self._episode_timeout_count = 0
        # pixel_cache_dir no longer needed: pixel_values are stored in
        # Ray object store (ray.put) instead of NFS disk.

        # --- Episode config ---
        self.train_difficulty_max_steps = train_difficulty_max_steps or {
            "easy": 10,
            "medium": 20,
            "hard": 30,
        }
        self.test_difficulty_max_steps = test_difficulty_max_steps or {
            "easy": 30,
            "medium": 50,
            "hard": 70,
        }
        self.turn_discount = turn_discount

        # --- Dynamic max_steps schedule ---
        # Increases train max_steps over training. Config format:
        #   base: [10, 20, 30]             # easy, medium, hard
        #   increase_interval: [30, 30, 30]  # every N training steps
        #   increase_amount: [2, 2, 2]       # add this many steps
        self.max_steps_schedule = max_steps_schedule

        # --- Model / prompt config ---
        model_config = model_config or {}
        self.model_config = {
            "model_type": model_config.get("model_type", "qwen3-instruct"),
            "prompt_version": model_config.get("prompt_version", "complete"),
            "interaction_mode": self.interaction_mode,
            "history_window": model_config.get("history_window", 4),
            "keep_thinking_in_history": model_config.get("keep_thinking_in_history", False),
        }
        self.enable_thinking = self.model_config["model_type"] == "qwen3-think"

        # --- WebGym ContextManager (lazy-init to avoid import issues in workers) ---
        self._context_manager = None

        # --- Evaluator (lazy-init) ---
        self.openai_config = openai_config
        self._evaluator = None

        # --- Per-worker identity ---
        self._worker_rank = env_config.get("worker_rank", 0)
        self._num_workers = env_config.get("num_rollout_workers", 1)

        # --- Task monitor (lazy-init, per-worker port) ---
        base_monitor_port = env_config.get("monitor_port", 0)
        self._monitor_port = (
            base_monitor_port + self._worker_rank if base_monitor_port > 0 else 0
        )
        self._monitor = None

        # --- Per-worker episode concurrency limit ---
        # Head-node workers get reduced concurrency to compensate for the
        # extra load on the master (trainer, Ray GCS, weight-sync, etc.).
        self._is_head_node = env_config.get("is_head_node", False)
        head_discount = env_config.get("head_node_concurrency_discount", 0.8)
        max_concurrent = env_config.get("max_concurrent_rollouts", 0)
        if max_concurrent > 0 and self._num_workers > 0:
            base_limit = max_concurrent // self._num_workers
            self._per_worker_limit = (
                max(1, int(base_limit * head_discount))
                if self._is_head_node
                else base_limit
            )
        else:
            self._per_worker_limit = 0
        self._episode_semaphore = (
            asyncio.Semaphore(self._per_worker_limit)
            if self._per_worker_limit > 0
            else None
        )

        # --- Episode ID counter for multi-step training ---
        # Each episode gets a unique ID so training can group steps by episode.
        # Use PID-based offset to avoid collision across workers (each worker
        # is a separate process with its own WebGymWorkflow instance).
        self._episode_id_counter = itertools.count(os.getpid() * 1_000_000)

        node_tag = "HEAD" if self._is_head_node else "worker"
        logger.info(
            "Worker rank=%d on %s node: per_worker_limit=%d (base=%s, discount=%s)",
            self._worker_rank,
            node_tag,
            self._per_worker_limit,
            max_concurrent // self._num_workers
            if self._num_workers > 0 and max_concurrent > 0
            else "N/A",
            head_discount if self._is_head_node else "1.0",
        )

        # --- Temp dir for screenshots ---
        # Use shared PVC (via screenshot_dir from env_config) so the master
        # node can access screenshots saved by workers when persisting
        # trajectories to the rollout directory.  Falls back to /tmp.
        _shared_base = env_config.get("screenshot_dir", "")
        if _shared_base:
            os.makedirs(_shared_base, exist_ok=True)
        else:
            _shared_base = "/tmp"
        self._tmpdir = tempfile.mkdtemp(prefix="webgym_workflow_", dir=_shared_base)
        import glob as _glob
        _cutoff = time.time() - 3600  # 1 hour ago (generous to avoid race with slow MoE model init)
        for _old in _glob.glob(os.path.join(_shared_base, "webgym_workflow_*")):
            if _old == self._tmpdir:
                continue
            try:
                if os.path.getmtime(_old) < _cutoff:
                    shutil.rmtree(_old, ignore_errors=True)
            except Exception:
                pass
        self._episode_count = 0  # for periodic memory logging

    # ------------------------------------------------------------------
    # Lazy initialisation helpers (called inside arun_episode)
    # ------------------------------------------------------------------

    @property
    def context_manager(self):
        if self._context_manager is None:
            from webgym.context import ContextManager

            context_config = {"interaction_mode": self.interaction_mode}
            self._context_manager = ContextManager(
                context_config, self.model_config, verbose=False
            )
        return self._context_manager

    @property
    def evaluator(self):
        if self._evaluator is None and self.openai_config is not None:
            try:
                from webgym.models.evaluator import Evaluator

                conv_builder = (
                    self.context_manager.get_model_interface().conversation_builder
                )
                self._evaluator = Evaluator(
                    openai_config=self.openai_config,
                    conversation_builder=conv_builder,
                    max_retries=self.max_retries,
                    verbose=False,
                )
            except Exception:
                logger.warning(
                    "Failed to init Evaluator, using simple reward", exc_info=True
                )
        return self._evaluator

    @property
    def monitor(self):
        if self._monitor is None and self._monitor_port > 0:
            from areal.workflow.task_monitor import TaskMonitor

            self._monitor = TaskMonitor(
                web_port=self._monitor_port,
                worker_rank=self._worker_rank,
            )
            if not self._monitor.running:
                self._monitor.start_monitoring()
        return self._monitor

    # ------------------------------------------------------------------
    # Difficulty-based max_steps
    # ------------------------------------------------------------------

    def _get_max_steps_for_difficulty(
        self,
        difficulty: int | str,
        split: str = "train",
        training_version: int = 0,
    ) -> int:
        """Get max_steps for a task based on its difficulty level.

        When ``max_steps_schedule`` is configured and ``split == "train"``,
        linearly interpolates between initial and max over training:

            cat = easy (d<=3) / medium (d<=6) / hard (d>6)
            n = min(version // interval, cap_step // interval)
            total_intervals = cap_step // interval
            max_steps = initial[cat] + n * (max[cat] - initial[cat]) / total_intervals

        Otherwise falls back to the 3-category static config.
        """
        difficulty_int = int(difficulty) if difficulty else 5  # default to medium

        if difficulty_int <= 3:
            cat_idx = 0
            category = "easy"
        elif difficulty_int <= 6:
            cat_idx = 1
            category = "medium"
        else:
            cat_idx = 2
            category = "hard"

        if split == "train" and self.max_steps_schedule is not None:
            sched = self.max_steps_schedule
            lo = sched["initial"][cat_idx]
            hi = sched["max"][cat_idx]
            interval = sched["increase_interval"]
            cap_step = sched["cap_step"]
            total_intervals = cap_step // interval if interval > 0 else 1
            n = min(training_version // interval, total_intervals) if interval > 0 else 0
            return lo + int(n * (hi - lo) / total_intervals)

        if split == "train":
            return self.train_difficulty_max_steps[category]
        return self.test_difficulty_max_steps[category]

    # ------------------------------------------------------------------
    # Screenshot comparison (post-processing)
    # ------------------------------------------------------------------

    @staticmethod
    def _add_screenshot_comparison_to_trajectory(
        trajectory: list[dict],
    ) -> list[dict]:
        """Add ``same_as_next_screenshot`` field to each step.

        True if the step's screenshot is pixel-identical to the next step's.
        Last step is always False.
        """
        import filecmp

        for j, step in enumerate(trajectory):
            if j == len(trajectory) - 1:
                step["same_as_next_screenshot"] = False
                continue

            cur_obs = step.get("observation")
            nxt_obs = trajectory[j + 1].get("observation")
            cur_path = getattr(cur_obs, "image_path", None) if cur_obs else None
            nxt_path = getattr(nxt_obs, "image_path", None) if nxt_obs else None

            same = False
            if cur_path and nxt_path:
                if cur_path == nxt_path:
                    same = True
                else:
                    try:
                        same = filecmp.cmp(cur_path, nxt_path, shallow=False)
                    except Exception:
                        same = False

            step["same_as_next_screenshot"] = same

        return trajectory

    # ------------------------------------------------------------------
    # HTTP helpers for Omniboxes browser interaction
    # ------------------------------------------------------------------

    def _base_url(self) -> str:
        protocol = "https" if self.master_port == 443 else "http"
        return f"{protocol}://{self.host_ip}:{self.master_port}"

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.cpu_cluster_token}

    def _allocate_instance(self) -> dict:
        """Allocate a browser instance from the Omniboxes master server.

        Retries with exponential backoff up to ~5 minutes total, giving
        time for other episodes to release their browsers.
        """
        url = f"{self._base_url()}/get"
        max_attempts = 15  # ~5 min total with backoff
        for attempt in range(max_attempts):
            try:
                params = {"lifetime_mins": self.instance_lifetime_mins}
                resp = requests.post(
                    url,
                    params=params,
                    headers=self._headers(),
                    verify=False,
                    timeout=self.operation_timeout,
                )
                if resp.status_code == 200:
                    return resp.json()
                if attempt % 5 == 0:  # Log every 5th attempt to reduce noise
                    logger.warning(
                        f"Instance alloc attempt {attempt + 1}/{max_attempts}: "
                        f"{resp.status_code} {resp.text[:200]}"
                    )
            except Exception as e:
                if attempt % 5 == 0:
                    logger.warning(
                        f"Instance alloc attempt {attempt + 1}/{max_attempts} error: {e}"
                    )
            if attempt < max_attempts - 1:
                time.sleep(min(20, 5 + attempt * 2))  # 5s, 7s, 9s, ... up to 20s
        raise RuntimeError("Failed to allocate browser instance")

    def _release_instance(self, instance: dict) -> None:
        """Release a browser instance back to the pool."""
        try:
            requests.post(
                f"{self._base_url()}/reset",
                params=instance,
                headers=self._headers(),
                verify=False,
                timeout=30,
            )
        except Exception:
            logger.warning("Failed to release instance", exc_info=True)

    def _navigate(self, instance: dict, url: str) -> None:
        """Navigate the browser to a URL."""
        cmd = {"visit_page": {"url": url}}
        try:
            requests.post(
                f"{self._base_url()}/execute",
                json={**cmd, **instance},
                headers=self._headers(),
                verify=False,
                timeout=self.operation_timeout,
            )
            # Wait for page load
            time.sleep(3)
        except Exception:
            logger.warning(f"Navigation to {url} failed", exc_info=True)

    def _take_screenshot(
        self, instance: dict, step: int, task_id: str = "", episode_suffix: str = ""
    ) -> tuple[str, Image.Image]:
        """Take a screenshot and save to a temp file.

        Retries up to 3 times on network errors (ChunkedEncodingError, timeouts).
        Returns (file_path, PIL.Image).
        """
        import io

        url = f"{self._base_url()}/screenshot"
        params = {**instance, "interaction_mode": self.interaction_mode}
        last_err: Exception | None = None

        for attempt in range(5):
            try:
                resp = requests.get(
                    url,
                    params=params,
                    headers=self._headers(),
                    verify=False,
                    timeout=self.operation_timeout,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"Screenshot failed: {resp.status_code}")

                img = Image.open(io.BytesIO(resp.content)).convert("RGB")

                # Include task_id and episode_suffix to prevent concurrent overwrites
                # within the same group (same task_id, different episodes).
                # Replace : with _ to avoid filesystem issues (instance_id contains port like :8023)
                suffix = f"_{episode_suffix.replace(':', '_')}" if episode_suffix else ""
                fname = (
                    f"task_{task_id}{suffix}_step_{step:03d}.png"
                    if task_id
                    else f"step_{step:03d}.png"
                )
                path = os.path.join(self._tmpdir, fname)
                img.save(path)
                return path, img
            except Exception as e:
                last_err = e
                if attempt < 4:
                    time.sleep(2 * (attempt + 1))  # 2s, 4s, 6s, 8s backoff

        raise last_err

    def _get_metadata(self, instance: dict) -> dict:
        """Get page metadata (title, url, viewport size)."""
        try:
            resp = requests.get(
                f"{self._base_url()}/metadata",
                params=instance,
                headers=self._headers(),
                verify=False,
                timeout=self.operation_timeout,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            logger.warning("Metadata request failed", exc_info=True)
        return {"title": "", "url": "", "width": 1280, "height": 720}

    def _execute_command(self, instance: dict, command: dict) -> None:
        """Execute a browser command."""
        try:
            requests.post(
                f"{self._base_url()}/execute",
                json={**command, **instance},
                headers=self._headers(),
                verify=False,
                timeout=self.operation_timeout,
            )
        except Exception:
            logger.warning("Command execution failed", exc_info=True)

    # ------------------------------------------------------------------
    # Message / image helpers
    # ------------------------------------------------------------------

    def _build_vllm_chat_messages(self, messages: list[dict]) -> list[dict]:
        """Convert ContextManager messages to vLLM-compatible chat format.

        Replaces ``file://`` image URLs with empty placeholders for vLLM's
        built-in image handling (images are sent via ``image_data`` field).
        """
        chat_msgs = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if item.get("type") == "image_url":
                        new_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": ""},
                            }
                        )
                    else:
                        new_content.append(item)
                chat_msgs.append({"role": msg["role"], "content": new_content})
            else:
                chat_msgs.append(msg)
        return chat_msgs

    def _collect_images_from_messages(
        self, messages: list[dict], all_images: list[Image.Image]
    ) -> list[Image.Image]:
        """Collect the images actually referenced in the messages.

        The ContextManager uses a 4-round sliding window, so only the last
        ~5 screenshots appear in the conversation.
        """
        image_count = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        image_count += 1
        # Return the last `image_count` images
        return all_images[-image_count:] if image_count > 0 else []

    def _build_multi_modal_input(
        self, images: list[Image.Image], text: str
    ) -> list[dict[str, Any]]:
        """Build ``multi_modal_input`` for AReaL training.

        Processes images through the model processor to obtain
        ``pixel_values`` (and ``image_grid_thw`` for Qwen-VL).
        """
        if not images:
            return []
        try:
            processed = self.processor(
                text=[text],
                images=images,
                padding=False,
                return_tensors="pt",
            )
            mmi = [{"pixel_values": processed["pixel_values"]}]
            if "image_grid_thw" in processed:
                mmi[0]["image_grid_thw"] = processed["image_grid_thw"]
            return mmi
        except Exception:
            logger.warning("Failed to build multi_modal_input", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Training data construction (post-rollout)
    # ------------------------------------------------------------------

    def _build_loss_mask_from_ids(self, input_ids: list[int]) -> list[int]:
        """Mark assistant-response tokens for training loss.

        Finds ``<|im_start|>assistant\\n … <|im_end|>`` spans and sets
        loss_mask = 1 for the content tokens (including ``<|im_end|>``
        so the model learns to stop).
        """
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_prefix = self.tokenizer.encode(
            "assistant\n", add_special_tokens=False
        )
        prefix_len = len(assistant_prefix)

        mask = [0] * len(input_ids)

        i = 0
        while i < len(input_ids):
            if input_ids[i] == im_start_id:
                start = i + 1
                if (
                    start + prefix_len <= len(input_ids)
                    and input_ids[start : start + prefix_len] == assistant_prefix
                ):
                    # Mark content tokens after "assistant\n"
                    j = start + prefix_len
                    while j < len(input_ids) and input_ids[j] != im_end_id:
                        mask[j] = 1
                        j += 1
                    # Include <|im_end|> itself so model learns to stop
                    if j < len(input_ids):
                        mask[j] = 1
                    i = j + 1
                    continue
            i += 1

        return mask

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_task_reward(
        self, trajectory: list[dict], task_data: dict[str, Any]
    ) -> tuple[float, list | None, bool, bool]:
        """Compute reward for a completed episode.

        Matches scaled_webgym behavior:
        - Answer trajectories: full evaluator pipeline (blocking + criteria).
        - Non-answer trajectories: reward=0, is_blocked=False.
        - Evaluator failure: reward=0 (never reward=1 on failure).

        Returns:
            (reward, evaluation, is_blocked, evaluator_failed)
            evaluator_failed is True only when the evaluator was attempted
            and raised; False otherwise (success or evaluator skipped).
        """
        last_is_answer = False
        if trajectory:
            last_action = trajectory[-1].get("action")
            if last_action is not None:
                last_is_answer = last_action.action.get("key") == "answer"

        if self.evaluator is not None and last_is_answer:
            try:
                reward_val, evaluation, is_blocked = (
                    self.evaluator.get_verifiable_reward(trajectory)
                )
                # Preserve evaluator reward as-is (matching scaled_webgym).
                return float(reward_val), evaluation, is_blocked, False
            except Exception as e:
                logger.warning("Evaluator failed, using simple reward", exc_info=True)
                return 0.0, [f"Evaluation failed: {e}"], False, True

        # No evaluator or non-answer trajectory: reward=0.
        if not last_is_answer:
            return 0.0, None, False, False
        return 0.0, None, False, False

    # ------------------------------------------------------------------
    # Observation text generation (screenshot comparison)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_observation_text(
        prev_screenshot: str,
        current_screenshot: str,
        page_metadata: dict,
    ) -> str:
        """Generate observation text by comparing consecutive screenshots.

        Mirrors scaled_webgym's ``_generate_observation_text`` so that
        ``_summarize_older_history`` in ``QwenConversationBuilder`` receives
        meaningful ``observation`` values for steps outside the sliding window.
        """
        import filecmp

        title = page_metadata.get("title", "Unknown") if page_metadata else "Unknown"
        url = page_metadata.get("url", "Unknown") if page_metadata else "Unknown"

        # Shorten long URLs
        if url != "Unknown" and len(url) > 60:
            try:
                import urllib.parse

                parsed = urllib.parse.urlparse(url)
                path_part = (
                    parsed.path[:30] + "..." if len(parsed.path) > 30 else parsed.path
                )
                url = f"{parsed.netloc}{path_part}"
            except Exception:
                url = url[:60] + "..."

        # Compare screenshots
        images_identical = False
        if prev_screenshot and current_screenshot:
            try:
                images_identical = filecmp.cmp(
                    prev_screenshot, current_screenshot, shallow=False
                )
            except Exception:
                images_identical = False

        if images_identical:
            return (
                "After the action above is executed by the environment, "
                "the webpage did not change (this means the last action "
                f"is not effective). The URL of the webpage after "
                f"executing the action: {url}"
            )
        return (
            "After the action above is executed by the environment, "
            "the webpage changed (this means the last action was "
            f"effective). The URL of the webpage after "
            f"executing the action: {url}"
        )

    # ------------------------------------------------------------------
    # Trajectory serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_trajectory_steps(trajectory: list[dict]) -> list[dict]:
        """Serialize trajectory steps to JSON-safe dicts for JSONL storage.

        Matches the per-step data structure from scaled_webgym so that
        ``view_trajs.py`` can fully reconstruct the rollout state.
        """
        steps = []
        for step in trajectory:
            obs = step.get("observation")
            act = step.get("action")
            resp = step.get("response")
            rew = step.get("reward")

            serialized: dict[str, Any] = {
                "observation": {
                    "image_path": obs.image_path if obs else "",
                    "page_metadata": obs.page_metadata if obs else {},
                    "ac_tree": obs.ac_tree if obs else "",
                }
                if obs
                else None,
                "action": {
                    "action": act.action if act else None,
                    "action_string": act.action_string if act else "",
                }
                if act
                else None,
                "response": {
                    "raw_response": resp.raw_response if resp else "",
                    "raw_prompt": resp.raw_prompt if resp else "",
                    "answering_tokens": resp.answering_tokens if resp else {},
                }
                if resp
                else None,
                "reward": {
                    "reward": rew.reward if rew else 0,
                    "evaluation": str(rew.evaluation) if rew and rew.evaluation else "",
                    "is_blocked": rew.is_blocked if rew else False,
                    "submit": getattr(rew, "submit", False) if rew else False,
                    "submission_judgment": getattr(rew, "submission_judgment", None),
                }
                if rew
                else None,
            }
            steps.append(serialized)
        return steps

    # ------------------------------------------------------------------
    # Main episode loop
    # ------------------------------------------------------------------

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Run a single WebGym episode, respecting per-worker concurrency limit."""
        if self._episode_semaphore is not None:
            async with self._episode_semaphore:
                return await self._arun_episode_impl(engine, data)
        return await self._arun_episode_impl(engine, data)

    async def _arun_episode_impl(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Run a single WebGym episode.

        Parameters
        ----------
        engine : InferenceEngine
            AReaL inference engine for action generation.
        data : dict
            Dataset sample with task metadata.

        Returns
        -------
        dict or None
            Trajectory tensors for PPO training, or None to reject.
        """
        # --- Load reward function dynamically if given as string ---
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        # --- Extract task info ---
        from webgym.data.components import Action, Observation, Response, Reward, Task

        task = Task(
            task_name=data.get("task_name", ""),
            domain=data.get("domain", ""),
            subdomain=data.get("subdomain", ""),
            website=data.get("website", ""),
            difficulty=data.get("difficulty", ""),
            evaluator_reference=json.loads(data["evaluator_reference"])
            if isinstance(data.get("evaluator_reference"), str)
            else data.get("evaluator_reference", []),
            reference_answer=data.get("reference_answer", ""),
            task_id=data.get("task_id", ""),
        )

        # --- Compute per-task max_steps based on difficulty ---
        _split = "test" if workflow_context.get().is_eval else "train"
        _version = engine.get_version() if hasattr(engine, "get_version") else 0
        max_steps = self._get_max_steps_for_difficulty(
            task.difficulty, split=_split, training_version=_version
        )
        task.max_steps = max_steps

        if not task.website:
            logger.warning(f"Task {task.task_id} has no website, skipping")
            return None

        # --- Task monitor ---
        # Use a unique episode key (auto-incrementing) so that re-rollouts of
        # the same task_id don't overwrite finished/failed cells in the grid.
        monitor = self.monitor
        episode_key = ""
        if monitor:
            episode_key = monitor.register_task(task.task_id, task.task_name, max_steps)
            monitor.start_allocation_wait(episode_key, task.task_name)

        # --- Accumulators ---
        all_images: list[Image.Image] = []
        screenshot_paths: list[str] = []
        trajectory: list[dict] = []
        # Per-step data for multi-step training.  Each step's sliding-window
        # context is saved independently so training sees exactly what the
        # model saw at rollout time.
        steps_data: list[dict[str, Any]] = []
        instance = None
        # Episode failed → no caller will dump these screenshots, so the
        # finally block below must nuke them itself (otherwise the workflow
        # tempdir grows monotonically with every 504 / blocked / timeout).
        _episode_failed = False

        try:
            # --- Allocate browser instance ---
            instance = await asyncio.to_thread(self._allocate_instance)

            # --- Navigate to website ---
            if monitor:
                monitor.start_task(episode_key, task.task_name)
                monitor.set_task_navigating(episode_key, task.website)
            await asyncio.to_thread(self._navigate, instance, task.website)

            # --- Get screen dimensions ---
            if monitor:
                monitor.set_task_getting_metadata(episode_key)
            metadata = await asyncio.to_thread(self._get_metadata, instance)
            screen_w = metadata.get("width", 1280)
            screen_h = metadata.get("height", 720)

            _episode_start_time = time.time()
            _episode_timed_out = False

            for step in range(max_steps):
                # Check per-episode wall-clock timeout
                elapsed_min = (time.time() - _episode_start_time) / 60
                if elapsed_min > self.episode_timeout_minutes:
                    logger.warning(
                        "[EPISODE_TIMEOUT] task=%s step=%d/%d elapsed=%.1fmin "
                        "limit=%dmin — terminating episode",
                        task.task_id,
                        step,
                        max_steps,
                        elapsed_min,
                        self.episode_timeout_minutes,
                    )
                    _episode_timed_out = True
                    self._episode_timeout_count += 1
                    break

                if monitor:
                    monitor.update_task_step(episode_key, step)

                # 1. Take screenshot
                if monitor:
                    monitor.set_task_taking_screenshot(episode_key, step)
                # Use instance_id as episode suffix to distinguish screenshots
                # from different episodes of the same task within a group.
                _ep_suffix = instance.get("instance_id", "") if isinstance(instance, dict) else ""
                screenshot_path, screenshot_img = await asyncio.to_thread(
                    self._take_screenshot, instance, step, task.task_id, _ep_suffix
                )
                all_images.append(screenshot_img)
                screenshot_paths.append(screenshot_path)

                # 2. Get page metadata
                page_metadata = await asyncio.to_thread(self._get_metadata, instance)

                # 2b. Generate observation text for PREVIOUS step
                # (compare prev screenshot vs current to tell model what happened)
                if len(trajectory) >= 1:
                    prev_step = trajectory[-1]
                    prev_obs = prev_step.get("observation")
                    prev_resp = prev_step.get("response")
                    if prev_obs and prev_resp:
                        obs_text = self._generate_observation_text(
                            prev_screenshot=prev_obs.image_path,
                            current_screenshot=screenshot_path,
                            page_metadata=page_metadata,
                        )
                        prev_resp.answering_tokens["observation"] = obs_text

                # 3. Build observation and add to trajectory
                observation = Observation(
                    task=task,
                    image_path=screenshot_path,
                    ac_tree="",
                    page_metadata=page_metadata,
                )
                trajectory.append(
                    {
                        "observation": observation,
                        "action": None,
                        "response": None,
                        "reward": Reward(reward=0, evaluation="In progress"),
                    }
                )

                # 4. Build conversation messages via ContextManager
                current_obs = {
                    "image_path": screenshot_path,
                    "ac_tree": "",
                    "page_metadata": page_metadata,
                }
                messages = self.context_manager.build_conversation(
                    task=task.task_name,
                    trajectory=trajectory,
                    current_observation=current_obs,
                    website=task.website,
                )

                # 5. Process messages to get input_ids + pixel_values
                messages_text = self.processor.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                )
                # For thinking models: strip trailing <think>\n from the
                # generation prompt so the model spontaneously decides
                # whether to think.  The chat template for Qwen3-VL-Thinking
                # unconditionally appends "<think>\n"; we remove it here.
                if self.enable_thinking and messages_text.endswith("<think>\n"):
                    messages_text = messages_text[: -len("<think>\n")]
                # Collect only the images referenced in the conversation
                conv_images = self._collect_images_from_messages(messages, all_images)
                processed_input = self.processor(
                    text=[messages_text],
                    images=conv_images if conv_images else None,
                    padding=False,
                    return_tensors="pt",
                )
                input_ids: list[int] = processed_input["input_ids"].tolist()[0]

                # Debug: log prompt overflow with per-message breakdown
                _prompt_len = len(input_ids)
                _max_tok = self.gconfig.max_tokens
                _max_new = self.gconfig.max_new_tokens
                if _prompt_len + _max_new > _max_tok:
                    _msg_summary = []
                    for _mi, _m in enumerate(messages):
                        _role = _m.get("role", "?")
                        _content = _m.get("content", "")
                        if isinstance(_content, str):
                            _tlen = len(self.processor.tokenizer.encode(_content))
                            _preview = repr(_content[:120])
                        elif isinstance(_content, list):
                            _tlen = sum(
                                len(self.processor.tokenizer.encode(c["text"]))
                                if c.get("type") == "text" else 960
                                for c in _content
                            )
                            _preview = str([c.get("type") for c in _content])
                        else:
                            _tlen = 0
                            _preview = "?"
                        _msg_summary.append(
                            f"  msg[{_mi}] {_role}: ~{_tlen} tok, {_preview}"
                        )
                    logger.warning(
                        "[PROMPT_OVERFLOW] task=%s step=%d/%d "
                        "prompt_len=%d + max_new=%d > max_tok=%d\n%s",
                        task.task_id, step, max_steps,
                        _prompt_len, _max_new, _max_tok,
                        "\n".join(_msg_summary),
                    )

                # 6. Build ModelRequest for AReaL engine
                byte_images = image2base64(conv_images) if conv_images else []
                chat_messages = self._build_vllm_chat_messages(messages)
                req = ModelRequest(
                    rid=uuid.uuid4().hex,
                    input_ids=input_ids,
                    image_data=byte_images,
                    vision_msg_vllm=[chat_messages],
                    gconfig=self.gconfig.new(n_samples=1),
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                )

                # 7. Generate action via AReaL engine
                if monitor:
                    monitor.set_task_getting_action(episode_key)
                try:
                    resp = await engine.agenerate(req)
                except RuntimeError as e:
                    # Context overflow (prompt_len >= max_tokens) — kill
                    # this trajectory cleanly: end as failure with reward 0
                    # instead of crashing the whole episode worker.
                    if "max_new_tokens" in str(e) and "non-positive" in str(e):
                        logger.warning(
                            "[task=%s step=%d] context overflow "
                            "(prompt_len=%d), terminating trajectory early",
                            task.task_id, step, len(input_ids),
                        )
                        # Drop this step's incomplete trajectory entry
                        # (observation appended but no action/response yet).
                        if trajectory and trajectory[-1].get("action") is None:
                            trajectory.pop()
                            if all_images:
                                all_images.pop()
                            if screenshot_paths:
                                screenshot_paths.pop()
                        break
                    raise

                # 8. Save this step's data for multi-step training
                # Extract only needed tensors; drop the full HF processor
                # output to avoid holding attention_mask, input_ids copies,
                # etc. across await suspension points.
                steps_data.append(
                    {
                        "input_ids": input_ids,
                        "pixel_values": processed_input.get("pixel_values"),
                        "image_grid_thw": processed_input.get("image_grid_thw"),
                        "conv_images": conv_images,
                        "resp": resp,
                    }
                )
                del processed_input

                # 9. Parse action from generated text
                generated_text = self.tokenizer.decode(resp.output_tokens)
                try:
                    parsed = self.context_manager.parse_response(generated_text)
                    action_info = self.context_manager.extract_action(parsed)
                except Exception:
                    logger.warning(
                        f"Failed to parse action at step {step}", exc_info=True
                    )
                    action_info = {"key": "wait", "arguments": {"time": 1}}
                    parsed = {"action": "wait"}

                action = Action(
                    action=action_info,
                    action_string=parsed.get("action", str(action_info)),
                )

                # 10. Update trajectory step with action & response
                from webgym.utils.image_utils import convert_messages_to_path_format

                raw_prompt_json = convert_messages_to_path_format(messages)
                response_obj = Response(
                    raw_response=generated_text,
                    answering_tokens=parsed,
                    raw_prompt=raw_prompt_json,
                )
                trajectory[-1]["action"] = action
                trajectory[-1]["response"] = response_obj

                # 11. Check for terminal action (answer)
                if action_info.get("key") == "answer":
                    break

                # 12. Execute action in browser
                if monitor:
                    monitor.set_task_executing_action(
                        episode_key, action.action_string[:30]
                    )
                browser_cmd = self.context_manager.parse_action_to_browser_command(
                    action,
                    screen_dimensions=(screen_w, screen_h),
                    homepage_url=task.website,
                )
                await asyncio.to_thread(self._execute_command, instance, browser_cmd)
                if monitor:
                    monitor.set_task_normal_phase(episode_key)
                # Brief wait for page to update
                await asyncio.sleep(1)

            # --- Compute reward ---
            evaluator_failed = False
            if _episode_timed_out:
                reward, evaluation, is_blocked = 0.0, None, False
            else:
                if monitor:
                    monitor.set_task_computing_reward(episode_key)
                reward, evaluation, is_blocked, evaluator_failed = await asyncio.to_thread(
                    self._compute_task_reward, trajectory, data
                )

                # Apply turn discount
                discount = self.turn_discount ** max(0, len(trajectory) - 1)
                reward = float(reward * discount)

            if monitor:
                monitor.finish_task(episode_key, True)
        except Exception as exc:
            import traceback

            tb_str = traceback.format_exc()
            msg = (
                f"[TRAJ_NONE] task={task.task_id} step={len(trajectory)}/{max_steps} "
                f"{type(exc).__name__}: {str(exc)[:500]}\n{tb_str[-1500:]}"
            )
            logger.warning(msg)
            print(msg, flush=True)
            if monitor:
                monitor.finish_task(episode_key, False, "Episode exception")
            _episode_failed = True
            # Per-episode failure indicators paired with the success-path
            # blocking_rate/evaluator_failed/omniboxes_failed scalars below.
            # Evaluator wasn't reached on this branch, so always 0 there.
            try:
                stats_tracker.get(workflow_context.stat_scope()).scalar(
                    omniboxes_failed=float(_is_omniboxes_exception(exc)),
                    evaluator_failed=0.0,
                )
            except Exception:
                pass
            return None
        finally:
            # Always release the browser instance
            if instance is not None:
                await asyncio.to_thread(self._release_instance, instance)
            # Sweep this episode's screenshots from the workflow tempdir.
            # Use instance_id in the prefix to avoid deleting sibling
            # episodes' screenshots (same task_id but different instance_id
            # in a group).
            #   Success path: keep trajectory screenshots so workflow_executor
            #   can copy them to PVC; only retry artifacts get unlinked here.
            #   Failure path (exception): no caller will dump these, so we
            #   nuke ALL task-prefix files including the kept ones —
            #   otherwise they leak forever.
            _inst_id = instance.get("instance_id", "") if isinstance(instance, dict) else ""
            _inst_suffix = _inst_id.replace(":", "_")
            # Require an instance_id to sweep — without it the prefix
            # collapses to "task_<id>_" which would match sibling
            # episodes' screenshots in the same group (n_samples=8 share
            # this tempdir). If we never got an instance, no screenshot
            # was ever taken by this call so there is nothing to clean.
            if _inst_suffix and data.get("task_id", ""):
                _task_prefix = f"task_{data.get('task_id', '')}_{_inst_suffix}_"
                _keep: set[str] = set() if _episode_failed else set(screenshot_paths)
                for _f in os.listdir(self._tmpdir):
                    _fpath = os.path.join(self._tmpdir, _f)
                    if _f.startswith(_task_prefix) and _fpath not in _keep:
                        try:
                            os.remove(_fpath)
                        except OSError:
                            pass

        # --- Episode counter (gc.collect moved to after result is built) ---
        self._episode_count += 1

        # --- Log stats ---
        # Align with visualize_results.py: exclude blocked episodes from
        # reward denominator (only count non-crashed, non-blocked episodes).
        scope = workflow_context.stat_scope()
        n_steps = len(trajectory)
        # Always log blocking rate (1=blocked, 0=not); average = blocking rate.
        # evaluator_failed: 1 only when the evaluator was attempted and threw;
        #   averaged over all reached episodes (denominator includes non-answer
        #   trajectories where the evaluator was never called).
        # omniboxes_failed: 0 here because the rollout succeeded; the exception
        #   path emits 1 when the failure was an Omniboxes/HTTP error.
        stats_tracker.get(scope).scalar(
            blocking_rate=float(is_blocked),
            evaluator_failed=float(evaluator_failed),
            omniboxes_failed=0.0,
        )
        # Outcome split: align with response-length / entropy splits below.
        # Convention: success = reward > 0.5, failure = reward <= 0.5 (matches
        # the response_algorithm_curve.py / collect_key_counts filter).
        outcome = "success" if (reward is not None and reward > 0.5) else "failure"
        if not is_blocked:
            stats_tracker.get(scope).scalar(reward=reward, num_steps=n_steps)
            stats_tracker.get(f"{scope}/{outcome}").scalar(num_steps=n_steps)

            # Per-difficulty stats: easy (1-3), medium (4-6), hard (7+)
            difficulty_int = int(task.difficulty) if task.difficulty else 5
            if difficulty_int <= 3:
                cat = "easy"
            elif difficulty_int <= 6:
                cat = "medium"
            else:
                cat = "hard"
            stats_tracker.get(f"{scope}/{cat}").scalar(reward=reward, num_steps=n_steps)

            # Per-difficulty-level stats (e.g. rollout/diff_1, rollout/diff_5)
            stats_tracker.get(f"{scope}/diff_{difficulty_int}").scalar(
                reward=reward, num_steps=n_steps
            )

            # Per-action-type counts
            if trajectory:
                action_counts: dict[str, int] = {}
                for traj_step in trajectory:
                    act = traj_step.get("action")
                    if act is not None:
                        act_dict = act.action if hasattr(act, "action") else act
                        if isinstance(act_dict, dict):
                            key = act_dict.get("key", "")
                            if key:
                                action_counts[key] = action_counts.get(key, 0) + 1
                action_stats = {
                    f"action_{k}_rate": v / n_steps
                    for k, v in action_counts.items()
                }
                stats_tracker.get(scope).scalar(**action_stats)

            # Screenshot comparison stats
            if trajectory and n_steps > 1:
                # Run comparison here for stats; will run again before
                # building training data (idempotent — just sets the field).
                self._add_screenshot_comparison_to_trajectory(trajectory)
                _n_unchanged = sum(
                    1 for s in trajectory if s.get("same_as_next_screenshot", False)
                )
                stats_tracker.get(scope).scalar(
                    unchanged_screenshot_rate=_n_unchanged / (n_steps - 1),
                    filtered_steps=_n_unchanged,
                )

                # Stuck rate: screenshot unchanged AND action != wait.
                # Reported as a diagnostic only; wait actions legitimately
                # produce no visual change and are excluded from the count.
                _n_stuck = 0
                for traj_step in trajectory:
                    if not traj_step.get("same_as_next_screenshot", False):
                        continue
                    act = traj_step.get("action")
                    if act is None:
                        continue
                    act_dict = act.action if hasattr(act, "action") else act
                    if isinstance(act_dict, dict) and act_dict.get("key") != "wait":
                        _n_stuck += 1
                stats_tracker.get(scope).scalar(stuck_rate=_n_stuck / n_steps)

        if not trajectory or not steps_data:
            msg = (
                f"[TRAJ_NONE] Empty/no-steps task={task.task_id} "
                f"len(traj)={len(trajectory)} len(steps)={len(steps_data)}"
            )
            logger.warning(msg)
            print(msg, flush=True)
            del trajectory, steps_data
            import gc as _gc
            _gc.collect()
            try:
                import ctypes as _ctypes
                _ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
            return None

        # --- Screenshot comparison (before building training data) ---
        # Mark steps where the page didn't change so we can zero their loss_mask.
        self._add_screenshot_comparison_to_trajectory(trajectory)
        n_filtered_steps = sum(
            1 for s in trajectory if s.get("same_as_next_screenshot", False)
        )

        # --- Build multi-step training data ---
        # Each step becomes an independent training sample with its own
        # sliding-window context, preserving exactly what the model saw
        # at rollout time.  One episode → T samples.
        all_input_ids: list[torch.Tensor] = []
        all_logprobs: list[torch.Tensor] = []
        all_loss_masks: list[torch.Tensor] = []
        all_versions: list[torch.Tensor] = []
        all_mmis: list[dict[str, Any]] = []

        # Per-image dedup: HISTORY_WINDOW=4 means each screenshot appears
        # in ~4 consecutive steps.  Deduplicate by PIL object identity
        # (the conversation builder reuses the same Image objects).
        image_store: list[torch.Tensor] = []  # unique per-image pixel_values
        pil_id_to_idx: dict[int, int] = {}

        step_response_lens: list[int] = []
        # Flat list of output-token logprobs across all steps; used to emit a
        # rollout-time MC entropy estimate (H ≈ -E[log p(x)]).
        flat_output_logprobs: list[float] = []
        all_is_screenshot_unchanged: list[int] = []
        for step_idx, step in enumerate(steps_data):
            inp = step["input_ids"]
            out = list(step["resp"].output_tokens)
            seq = inp + out
            n_in, n_out = len(inp), len(out)
            step_response_lens.append(n_out)
            try:
                flat_output_logprobs.extend(
                    float(lp) for lp in step["resp"].output_logprobs
                )
            except Exception:
                pass

            # Detect screenshot-unchanged steps for the FBC filter.
            is_screenshot_unchanged_step = (
                step_idx < len(trajectory)
                and trajectory[step_idx].get("same_as_next_screenshot", False)
            )
            all_is_screenshot_unchanged.append(int(is_screenshot_unchanged_step))

            all_input_ids.append(torch.tensor(seq, dtype=torch.int32))
            all_loss_masks.append(
                torch.tensor([0] * n_in + [1] * n_out, dtype=torch.int32)
            )
            all_logprobs.append(
                torch.tensor(
                    [0.0] * n_in + list(step["resp"].output_logprobs),
                    dtype=torch.float32,
                )
            )
            all_versions.append(
                torch.tensor(
                    [-1] * n_in + list(step["resp"].output_versions),
                    dtype=torch.int32,
                )
            )

            mmi_entry: dict[str, Any] = {}
            if step["conv_images"]:
                pv = step["pixel_values"]
                thw = step["image_grid_thw"]

                # Split pixel_values into per-image tensors
                patches_per_img = thw.prod(dim=-1)
                per_img_pv = pv.split(patches_per_img.tolist())

                step_indices = []
                for i, img in enumerate(step["conv_images"]):
                    pil_id = id(img)
                    if pil_id not in pil_id_to_idx:
                        pil_id_to_idx[pil_id] = len(image_store)
                        image_store.append(per_img_pv[i])
                    step_indices.append(pil_id_to_idx[pil_id])

                mmi_entry["image_indices"] = step_indices
                mmi_entry["image_grid_thw"] = thw
            # Release heavy tensors now that pixel_values are extracted
            step.pop("pixel_values", None)
            step.pop("image_grid_thw", None)
            step.pop("conv_images", None)
            all_mmis.append(mmi_entry)

        T = len(steps_data)
        del steps_data, all_images  # Release heavy accumulators early
        max_step_len = max(t.shape[0] for t in all_input_ids)

        # Pad and stack into [T, max_step_len]
        padded_input_ids = torch.zeros(T, max_step_len, dtype=torch.int32)
        padded_logprobs = torch.zeros(T, max_step_len, dtype=torch.float32)
        padded_loss_mask = torch.zeros(T, max_step_len, dtype=torch.int32)
        padded_versions = torch.zeros(T, max_step_len, dtype=torch.int32)
        padded_attn_mask = torch.zeros(T, max_step_len, dtype=torch.bool)

        for i, (ids, lp, lm, ver) in enumerate(
            zip(all_input_ids, all_logprobs, all_loss_masks, all_versions)
        ):
            L = ids.shape[0]
            padded_input_ids[i, :L] = ids
            padded_logprobs[i, :L] = lp
            padded_loss_mask[i, :L] = lm
            padded_versions[i, :L] = ver
            padded_attn_mask[i, :L] = True

        ep_id = next(self._episode_id_counter)

        # Per-trajectory rollout-time summary stats — computed here so they
        # can be stashed in the `res` dict (used by TestRolloutManager) and
        # also re-used in the stats_tracker block below.
        avg_resp = (
            sum(step_response_lens) / len(step_response_lens)
            if step_response_lens else 0.0
        )
        max_resp = max(step_response_lens) if step_response_lens else 0
        entropy_avg = (
            -sum(flat_output_logprobs) / len(flat_output_logprobs)
            if flat_output_logprobs else 0.0
        )

        # Store deduplicated pixel tensors directly in PixelRefStore actor.
        # This avoids images flowing through the Ray RPC return path (which
        # lands them in the object store and causes spilling at scale).
        # Rollout workers stream images to the actor as each episode completes,
        # spreading the load naturally.  Only keys are included in the result.
        image_store_keys: list[str] = []
        if image_store:
            import numpy as _np

            import ray

            from areal.utils.pixel_store import get_or_create_pixel_store

            _keys = []
            _tensors = []
            for idx, pv in enumerate(image_store):
                k = f"{ep_id}:{idx}"
                _keys.append(k)
                _tensors.append(pv.numpy() if isinstance(pv, torch.Tensor) else pv)
            _pixel_store = get_or_create_pixel_store()
            ray.get(_pixel_store.put_batch.remote(_keys, _tensors))
            image_store_keys = _keys
            del _tensors

        res = dict(
            input_ids=padded_input_ids,
            logprobs=padded_logprobs,
            loss_mask=padded_loss_mask,
            versions=padded_versions,
            rewards=torch.full((T,), reward, dtype=torch.float32),
            is_screenshot_unchanged=torch.tensor(
                all_is_screenshot_unchanged, dtype=torch.int32
            ),
            difficulty=torch.full((T,), int(task.difficulty) if task.difficulty else 5, dtype=torch.int32),
            is_blocked=torch.full((T,), int(is_blocked), dtype=torch.int32),
            attention_mask=padded_attn_mask,
            multi_modal_input=all_mmis,
            _image_store_keys=image_store_keys,
            # Per-episode response-token-length + entropy stats (used by
            # TestRolloutManager to emit test_rollout/trajectory/{avg,max}
            # _response_tokens_per_step and entropy_avg; ignored for train
            # rollouts which log them directly via stats_tracker).
            _avg_response_tokens_per_step=[avg_resp],
            _max_response_tokens_per_step=[max_resp],
            _avg_entropy_per_token=[entropy_avg],
            # Wrap per-episode metadata in single-element lists so that
            # concat_padded_tensors (which extends lists) produces a list
            # of N values indexable by ep_idx in _dump_trajectory.
            _episode_timed_out=[_episode_timed_out],
            episode_id=torch.full((T,), ep_id, dtype=torch.int64),
            horizon=torch.full((T,), max_steps, dtype=torch.int32),
            _screenshot_paths=[screenshot_paths],
            _evaluation=[evaluation],
            _is_blocked=[is_blocked],
            _dataset_task_id=data.get("task_id", ""),
            _trajectory_steps=self._serialize_trajectory_steps(trajectory),
            _task_info={
                "task_name": task.task_name,
                "domain": task.domain,
                "subdomain": task.subdomain,
                "website": task.website,
                "difficulty": task.difficulty,
                "task_id": task.task_id,
                "evaluator_reference": task.evaluator_reference,
                "reference_answer": task.reference_answer,
            },
        )
        # Per-step response-token stats + per-trajectory rollout-time entropy
        # estimate. Split each by outcome (success / failure) so wandb shows
        # how response length and token-level entropy differ between
        # trajectories that solved vs failed the task. Values were computed
        # before the `res` dict above.
        if step_response_lens and not is_blocked:
            stats_tracker.get(scope).scalar(
                avg_response_tokens_per_step=avg_resp,
                max_response_tokens_per_step=max_resp,
            )
            stats_tracker.get(f"{scope}/{outcome}").scalar(
                avg_response_tokens_per_step=avg_resp,
            )
            if flat_output_logprobs:
                stats_tracker.get(scope).scalar(entropy_avg=entropy_avg)
                stats_tracker.get(f"{scope}/{outcome}").scalar(entropy_avg=entropy_avg)

        del trajectory, all_input_ids, all_logprobs, all_loss_masks, all_versions
        del image_store, pil_id_to_idx

        # Memory reclamation after each episode.
        import gc as _gc
        _gc.collect()
        try:
            import ctypes as _ctypes
            _ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        return res
