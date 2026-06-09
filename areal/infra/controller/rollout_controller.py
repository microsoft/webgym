from __future__ import annotations

import asyncio
import shutil
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any

from flask import Flask, jsonify, request
from torchdata.stateful_dataloader import StatefulDataLoader
from werkzeug.serving import make_server

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    InferenceEngineConfig,
    PerfTracerConfig,
    SchedulingSpec,
)
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import (
    LocalInfServerInfo,
    ModelRequest,
    ModelResponse,
    ParamSpec,
    WeightUpdateMeta,
)
from areal.api.scheduler_api import Job, Scheduler, Worker
from areal.api.workflow_api import RolloutWorkflow, WorkflowLike
from areal.infra.rpc.serialization import deserialize_value
from areal.infra.utils.concurrent import run_async_task
from areal.utils import logging, perf_tracer
from areal.utils.data import concat_padded_tensors, cycle_dataloader
from areal.utils.dynamic_import import import_from_string
from areal.utils.network import find_free_ports, gethostip
from areal.utils.perf_tracer import trace_perf

from ..staleness_manager import StalenessManager
from ..workflow_executor import BatchTaskDispatcher, TaskIdGenerator

logger = logging.getLogger("RolloutController")


# NOTE: remote task input has a slightly different
# type annotation, which disallows workflow object or types
@dataclass
class _RemoteRolloutTaskInput:
    task_id: int
    data: dict[str, Any]
    workflow: str
    workflow_kwargs: dict[str, Any]
    should_accept_fn: str | None
    is_eval: bool = False
    group_size: int = 1
    proxy_addr: str | None = None


@dataclass
class _RemoteRolloutResult:
    task_id: int
    trajectory: dict[str, Any]


class RolloutController:
    def __init__(
        self,
        inf_engine: type[InferenceEngine],
        config: InferenceEngineConfig,
        scheduler: Scheduler,
    ):
        self.inf_engine = inf_engine
        self.config = config
        self.scheduler = scheduler

        # Worker management
        self.workers: list[Worker] = []  # List of Worker objects from scheduler
        self.server_infos: list[LocalInfServerInfo] = []
        self._worker_role: str

        # Round-robin scheduling
        self._current_worker_idx = 0

        # State
        self._version_lock = Lock()
        self._version = 0

        self._task_id_generator = TaskIdGenerator()

        # Use provided staleness manager or create a default one
        # The manager will be properly initialized in initialize()
        self._staleness_manager: StalenessManager | None = None

        # Dispatcher will be initialized in initialize() after staleness_manager is ready
        self._dispatcher: (
            BatchTaskDispatcher[_RemoteRolloutTaskInput, _RemoteRolloutResult] | None
        ) = None

        # HTTP callback server
        self._callback_app: Flask | None = None
        self._callback_server = None
        self._callback_server_thread: threading.Thread | None = None
        self._callback_port: int | None = None
        self._callback_host: str | None = None
        self._callback_loop: asyncio.AbstractEventLoop | None = None
        self._callback_loop_ready = threading.Event()

        # Task completion futures
        self._pending_futures: dict[int, asyncio.Future] = {}
        self._futures_lock = threading.Lock()

        # Proxy worker management (for AgentWorkflow support)
        self.proxy_workers: list[Worker] = []
        self.proxy_addrs: list[str] = []
        self._proxy_started = False

        # Tracking for stats
        self._last_consumed_count: int = 0

    def _engine_name(self, rank: int) -> str:
        """Generate engine name for a worker rank.

        Engine names follow the "role/index" format (e.g., "rollout/0", "rollout/1").
        """
        return f"{self._worker_role}/{rank}"

    def initialize(
        self,
        role: str,
        alloc_mode: AllocationMode,
        server_args: dict[str, Any] | None = None,
        server_infos: list[LocalInfServerInfo] | None = None,
        *args,
        **kwargs,
    ):
        # Get scheduling config from kwargs or use defaults
        # Schedule inference engines in the granularity of instance sizes,
        # usually TP x PP.
        self._worker_role = role

        # The first element of `self.config.scheduling_spec` is the resource spec
        # of workers, aka the RPC server process. Since a worker exactly matches
        # to a single engine instance in the local environment, we can dirrectly
        # use the spec of engines  as the spec of workers here. Engine scheduling
        # specs are ignored.
        sch_spec = SchedulingSpec(**asdict(self.config.scheduling_spec[0]))
        sch_spec.cpu *= alloc_mode.gen_instance_size
        sch_spec.mem *= alloc_mode.gen_instance_size
        if sch_spec.gpu > 0:
            sch_spec.gpu = alloc_mode.gen_instance_size
        job = Job(
            replicas=alloc_mode.gen.dp_size,
            tasks=[sch_spec for _ in range(alloc_mode.gen.dp_size)],
            scheduling_strategy=self.config.scheduling_strategy,
            role=self._worker_role,
        )

        # Call async scheduler methods synchronously
        run_async_task(
            self._async_initialize, job, server_args, server_infos, *args, **kwargs
        )

        # Initialize staleness manager for global capacity control.
        # All values in TASK units (1 task = n_samples episodes).
        # max_concurrent_rollouts is already in task units (converted in train.py).
        # consumer_batch_size is in episode units; convert here.
        _n_samples = getattr(self.config, "n_samples", 1) or 1
        max_concurrent_rollouts = (
            self.config.max_concurrent_rollouts or self.config.consumer_batch_size
        )
        consumer_batch_size = self.config.consumer_batch_size // _n_samples
        self._staleness_manager = StalenessManager(
            version_provider=self,
            max_concurrent_rollouts=max_concurrent_rollouts,
            consumer_batch_size=consumer_batch_size,
            max_staleness=self.config.max_head_offpolicyness,
        )

        # Create and initialize the dispatcher
        qsize = self.config.queue_size or max_concurrent_rollouts * 16
        self._dispatcher = BatchTaskDispatcher[
            _RemoteRolloutTaskInput, _RemoteRolloutResult
        ](
            max_queue_size=qsize,
            task_factory=self._create_submit_callback,
            staleness_manager=self._staleness_manager,
            enable_tracing=self.config.enable_rollout_tracing,
        )
        # Initialize the dispatcher's async task runner
        self._dispatcher.initialize(logger=logger)

        # Start callback server for weight sync coordination
        self._start_callback_server()

    async def _async_initialize(
        self,
        job: Job,
        server_args: dict[str, Any],
        server_infos: list[LocalInfServerInfo] | None = None,
        *args,
        **kwargs,
    ):
        # Create workers via scheduler
        logger.info("Creating workers via scheduler...")
        worker_ids = self.scheduler.create_workers(job=job)
        logger.info(f"Workers created: {worker_ids}")

        # Wait for workers to be ready
        logger.info("Waiting for workers to be ready...")
        self.workers = self.scheduler.get_workers(role=job.role)
        logger.info(f"Workers ready: {[w.id for w in self.workers]}")

        # Get engine class path for dynamic import on workers
        engine_class = self.inf_engine

        # Create and initialize engines on workers
        logger.info("Creating engines...")
        tasks = [
            self.scheduler.create_engine(
                worker_id=worker.id,
                engine=f"{engine_class.__module__}.{engine_class.__name__}",
                engine_name=self._engine_name(rank),
                config=self.config,
            )
            for rank, worker in enumerate(self.workers)
        ]
        await asyncio.gather(*tasks)
        logger.info("Engine created on all workers!")

        logger.info("Calling engine initialization...")
        n_workers = len(self.workers)
        if server_infos is not None:
            # Connecting to existing local servers for evaluation
            self.server_infos = server_infos
            assert len(self.server_infos) == len(self.workers), (
                len(self.server_infos),
                len(self.workers),
            )
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="initialize",
                    engine_name=self._engine_name(rank),
                    # args in `engine_api`
                    engine_id=str(rank),
                    addr=f"{info.host}:{info.port}",
                    num_rollout_workers=n_workers,
                    *args,
                    **kwargs,
                )
                for rank, (worker, info) in enumerate(
                    zip(self.workers, self.server_infos)
                )
            ]
            await asyncio.gather(*tasks)
        else:
            self.server_infos = await self._collective_rpc_async(
                "launch_server", server_args=server_args
            )
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="initialize",
                    engine_name=self._engine_name(rank),
                    # args in `engine_api`
                    engine_id=str(rank),
                    num_rollout_workers=n_workers,
                    *args,
                    **kwargs,
                )
                for rank, worker in enumerate(self.workers)
            ]
            await asyncio.gather(*tasks)

        logger.info("All engines are initialized...")

    def destroy(self):
        # Stop background threads and shutdown the async task runner
        if self._dispatcher is not None:
            self._dispatcher.destroy()

        self._stop_callback_server()

        self._collective_rpc("destroy", http_timeout=60.0)

        # Delete workers via scheduler
        if hasattr(self, "_worker_role"):
            try:
                self.scheduler.delete_workers(role=self._worker_role)
                logger.info("Workers deleted")
            except Exception as e:
                logger.error(f"Error deleting workers: {e}")

        # Delete proxy workers if initialized
        if self._proxy_started:
            try:
                self.scheduler.delete_workers(role="proxy")
                logger.info("Proxy workers deleted")
            except Exception as e:
                logger.error(f"Error deleting proxy workers: {e}")
            self.proxy_workers.clear()
            self.proxy_addrs.clear()
            self._proxy_started = False

        self.workers.clear()

    def start_proxy(self) -> None:
        """Initialize proxy workers for AgentWorkflow support.

        Creates proxy workers colocated with rollout workers. Each proxy worker
        runs a ProxyRolloutServer that connects to the same inference server
        as its corresponding rollout worker.
        """
        if self._proxy_started:
            logger.warning("Proxy workers already initialized")
            return

        if not self.server_infos:
            raise RuntimeError(
                "Cannot initialize proxy workers: rollout not initialized. "
                "Call initialize() first."
            )

        run_async_task(self._async_start_proxy)
        self._proxy_started = True

    async def _async_start_proxy(self) -> None:
        """Async implementation of proxy worker initialization."""
        command = "areal.experimental.openai.proxy.proxy_rollout_server"
        worker_ids = self.scheduler.fork_workers(
            role="proxy",
            target_role=self._worker_role,
            command=command,
        )
        logger.info(f"Proxy workers forked: {worker_ids}")

        self.proxy_workers = self.scheduler.get_workers(role="proxy")
        logger.info(f"Proxy workers: {[w.id for w in self.proxy_workers]}")

        engine_class = f"{self.inf_engine.__module__}.{self.inf_engine.__name__}"

        create_tasks = []
        for rank, worker in enumerate(self.proxy_workers):
            create_tasks.append(
                self.scheduler.create_engine(
                    worker_id=worker.id,
                    engine=engine_class,
                    engine_name=f"proxy/{rank}",
                    config=self.config,
                )
            )
        await asyncio.gather(*create_tasks)
        logger.info("Proxy engines created")

        init_tasks = []
        for rank, (worker, server_info) in enumerate(
            zip(self.proxy_workers, self.server_infos)
        ):
            init_tasks.append(
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="initialize",
                    engine_name=f"proxy/{rank}",
                    addr=f"{server_info.host}:{server_info.port}",
                )
            )
            self.proxy_addrs.append(f"http://{worker.ip}:{worker.worker_ports[0]}")
        await asyncio.gather(*init_tasks)

        logger.info(f"Proxy servers initialized. Addresses: {self.proxy_addrs}")

    def get_proxy_addr(self, rank: int) -> str:
        """Get the proxy server address for a given rollout worker rank.

        Parameters
        ----------
        rank : int
            The rank of the rollout worker

        Returns
        -------
        str
            The HTTP address of the corresponding proxy server
        """
        if not self._proxy_started:
            raise RuntimeError(
                "Proxy workers not initialized. Call start_proxy() first."
            )
        if rank >= len(self.proxy_addrs):
            raise IndexError(
                f"Invalid rank {rank}, only {len(self.proxy_addrs)} proxy workers"
            )
        return self.proxy_addrs[rank]

    def _start_callback_server(self):
        """Start Flask HTTP server to receive callbacks from RolloutCallback."""
        if self._callback_server is not None:
            logger.warning("Callback server already running")
            return

        app = Flask(__name__)
        app.logger.disabled = True

        @app.route("/callback/init_weights_group", methods=["POST"])
        def init_weights_group():
            payload = request.get_json() or {}
            meta = deserialize_value(payload.get("meta"))
            self._callback_loop.run_until_complete(self.init_weights_update_group(meta))
            return jsonify({"status": "ok"})

        @app.route("/callback/update_weights_xccl", methods=["POST"])
        def update_weights():
            payload = request.get_json() or {}
            meta = deserialize_value(payload.get("meta"))
            param_specs = deserialize_value(payload.get("param_specs"))
            self._callback_loop.run_until_complete(
                self.update_weights_from_distributed(meta, param_specs)
            )
            return jsonify({"status": "ok"})

        @app.route("/callback/update_weights_disk", methods=["POST"])
        def update_weights_disk():
            payload = request.get_json() or {}
            meta = deserialize_value(payload.get("meta"))
            self._callback_loop.run_until_complete(self.update_weights_from_disk(meta))
            return jsonify({"status": "ok"})

        @app.route("/callback/pause_generation", methods=["POST"])
        def pause_generation():
            self._callback_loop.run_until_complete(self.pause_generation())
            return jsonify({"status": "ok"})

        @app.route("/callback/continue_generation", methods=["POST"])
        def continue_generation():
            self._callback_loop.run_until_complete(self.continue_generation())
            return jsonify({"status": "ok"})

        @app.route("/callback/rollout_complete", methods=["POST"])
        def rollout_complete():
            payload = request.get_json() or {}
            task_id = payload.get("task_id")
            try:
                self._resolve_task_future(task_id)
                return jsonify({"status": "ok"})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @app.errorhandler(Exception)
        def handle_error(e):
            logger.error(f"Callback handler error: {e}")
            return jsonify({"error": str(e)}), 500

        # Configure Werkzeug logging
        import logging as stdlib_logging

        werkzeug_logger = stdlib_logging.getLogger("werkzeug")
        werkzeug_logger.setLevel(stdlib_logging.WARNING)

        self._callback_port = find_free_ports(1)[0]
        self._callback_host = gethostip()
        self._callback_app = app
        self._callback_server = make_server(
            self._callback_host, self._callback_port, app, threaded=False
        )

        def serve_forever():
            # Create and set event loop for this thread
            self._callback_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._callback_loop)
            # Signal that the loop is ready
            self._callback_loop_ready.set()
            logger.info(
                f"Callback server started on {self._callback_host}:{self._callback_port}"
            )
            self._callback_server.serve_forever()

        self._callback_server_thread = threading.Thread(
            target=serve_forever, daemon=True
        )
        self._callback_server_thread.start()
        # Wait for loop to be created
        self._callback_loop_ready.wait()

    def _stop_callback_server(self):
        """Stop the callback server if running."""
        if self._callback_server is not None:
            logger.info("Stopping callback server...")
            self._callback_server.shutdown()
            if self._callback_loop is not None:
                self._callback_loop.close()
            self._callback_server = None
            self._callback_app = None
            self._callback_server_thread = None
            self._callback_port = None
            self._callback_host = None
            self._callback_loop = None
            self._callback_loop_ready.clear()

    @property
    def callback_addr(self) -> str:
        """Return callback server address as 'host:port'."""
        if self._callback_host is None or self._callback_port is None:
            raise RuntimeError("Callback server not started")
        return f"{self._callback_host}:{self._callback_port}"

    def _resolve_task_future(self, task_id: int):
        """Resolve a pending future with the task result."""
        with self._futures_lock:
            future = self._pending_futures.pop(task_id, None)
        if future:
            future.get_loop().call_soon_threadsafe(future.set_result, None)

    def _collective_rpc(self, method: str, *args, **kwargs) -> list[Any]:
        return run_async_task(self._collective_rpc_async, method, *args, **kwargs)

    async def _collective_rpc_async(self, method: str, *args, **kwargs) -> list[Any]:
        tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id,
                method=method,
                engine_name=self._engine_name(rank),
                *args,
                **kwargs,
            )
            for rank, worker in enumerate(self.workers)
        ]
        return await asyncio.gather(*tasks)

    def _choose_worker(self) -> tuple[Worker, int]:
        """Choose a worker for the next request using round-robin scheduling.

        Returns
        -------
        tuple[Worker, int]
            The chosen worker object and its rank
        """
        if not self.workers:
            raise RuntimeError("No workers available to choose from.")
        worker = self.workers[self._current_worker_idx]
        rank = self._current_worker_idx
        self._current_worker_idx = (self._current_worker_idx + 1) % len(self.workers)
        return worker, rank

    def _resolve_workflow_str(self, workflow: WorkflowLike) -> str:
        """Resolve workflow to a string import path.

        Handles RolloutWorkflow, agent workflow instances/classes, and string paths.
        """

        # String paths - return as-is
        if isinstance(workflow, str):
            return workflow

        # RolloutWorkflow classes
        elif isinstance(workflow, type) and issubclass(workflow, RolloutWorkflow):
            return f"{workflow.__module__}.{workflow.__name__}"

        # RolloutWorkflow instances
        elif isinstance(workflow, RolloutWorkflow):
            return f"{workflow.__module__}.{workflow.__class__.__name__}"

        # Agent-like workflow classes
        elif isinstance(workflow, type):
            return f"{workflow.__module__}.{workflow.__name__}"

        # Agent-like workflow instances
        else:
            return f"{workflow.__module__}.{workflow.__class__.__name__}"

    def _resolve_should_accept_fn(
        self, should_accept_fn: Callable[[dict[str, Any]], bool] | str | None
    ):
        if callable(should_accept_fn):
            raise RuntimeError(
                "If given, `should_accept_fn` must be an importable string path, e.g., 'my_module.filter_func'."
            )
        if should_accept_fn is not None:
            try:
                import_from_string(should_accept_fn)
            except Exception:
                raise RuntimeError(
                    f"Failed to import `should_accept_fn` from string path: {should_accept_fn}"
                )
        return should_accept_fn

    def _rollout_stats(self) -> str:
        stats = self._staleness_manager.get_stats()
        return (
            f"enqueued: {stats.enqueued}, "
            f"running: {stats.running}, "
            f"accepted: {stats.accepted}, "
            f"rejected: {stats.rejected}."
        )

    def _create_submit_callback(self, pending_task: _RemoteRolloutTaskInput):
        async def _submit_then_wait() -> _RemoteRolloutResult | None:
            # Choose worker via round-robin
            worker, rank = self._choose_worker()
            engine_name = self._engine_name(rank)

            # NOTE: No need to call `on_rollout_submitted` here.
            # This function will be passed to `BatchTaskDispather` where
            # `on_rollout_submitted` will be called upon dispatching
            task_id = pending_task.task_id

            manager = self.staleness_manager

            try:
                # Set future for this task
                future = asyncio.get_event_loop().create_future()
                with self._futures_lock:
                    self._pending_futures[task_id] = future

                proxy_addr = pending_task.proxy_addr
                if self._proxy_started and proxy_addr is None:
                    proxy_addr = self.get_proxy_addr(rank)
                engine_task_id = await self.scheduler.async_call_engine(
                    worker.id,
                    "submit",
                    engine_name=engine_name,
                    data=pending_task.data,
                    workflow=pending_task.workflow,
                    workflow_kwargs=pending_task.workflow_kwargs,
                    should_accept_fn=pending_task.should_accept_fn,
                    http_timeout=self.config.request_timeout,
                    is_eval=pending_task.is_eval,
                    group_size=pending_task.group_size,
                    task_id=task_id,
                    callback_addr=f"http://{self.callback_addr}/callback/rollout_complete",
                    proxy_addr=proxy_addr,
                )

                assert task_id == engine_task_id, (task_id, engine_task_id)

                # Wait for callback to resolve the future
                await asyncio.wait_for(future, timeout=self.config.request_timeout)

                # Fetch the result
                result = await self.scheduler.async_call_engine(
                    worker.id,
                    "wait_for_task",
                    engine_name=engine_name,
                    task_id=engine_task_id,
                    timeout=0.1,  # A short time to prevent blocking other requests
                    raise_timeout=False,
                    http_timeout=self.config.request_timeout,
                )

                traj = result
                if traj is not None:
                    # --- Version-based staleness rejection ---
                    max_staleness = self.config.max_head_offpolicyness
                    if (
                        max_staleness is not None
                        and isinstance(traj, dict)
                        and "versions" in traj
                    ):
                        import torch

                        versions_t = traj["versions"]
                        if isinstance(versions_t, torch.Tensor):
                            gen_mask = versions_t >= 0
                            if gen_mask.any():
                                traj_version = versions_t[gen_mask].min().item()
                                current_version = manager.version_provider.get_version()
                                staleness = current_version - traj_version
                                if staleness > max_staleness:
                                    manager.on_rollout_rejected()
                                    # Free pixel store images for stale trajectories
                                    _img_keys = traj.get("_image_store_keys")
                                    if _img_keys:
                                        try:
                                            from areal.utils.pixel_store import (
                                                get_or_create_pixel_store,
                                            )

                                            import ray

                                            _store = get_or_create_pixel_store()
                                            ray.get(_store.free.remote(_img_keys))
                                        except Exception:
                                            pass
                                    return None

                    manager.on_rollout_accepted()

                    # Update adaptive task sampling tracker
                    _tracker = getattr(self, "_task_success_tracker", None)
                    if _tracker is not None and isinstance(traj, dict):
                        _dtid = traj.get("_dataset_task_id", "")
                        _rew = traj.get("rewards")
                        _ep_ids = traj.get("episode_id")
                        if _dtid and _rew is not None:
                            import torch as _torch

                            from areal.infra.rpc.rtensor import RTensor as _RTensor
                            if isinstance(_rew, _RTensor):
                                _rew = _rew.to_local()
                            if isinstance(_ep_ids, _RTensor):
                                _ep_ids = _ep_ids.to_local()
                            if isinstance(_rew, _torch.Tensor) and isinstance(_ep_ids, _torch.Tensor):
                                # Per-episode reward: first token of each episode
                                _unique_eps = _ep_ids.unique()
                                _ep_rewards = []
                                for _eid in _unique_eps:
                                    _mask = _ep_ids == _eid
                                    _ep_rewards.append(_rew[_mask][0].item())
                                _n_success = sum(1 for r in _ep_rewards if r > 0)
                                # Format task text for BERT training
                                _task_text = ""
                                _ds_index = getattr(self, "_dataset_index", None)
                                if _ds_index is None and hasattr(self, "_full_dataset"):
                                    # Build index on first use (key by both str and int)
                                    self._dataset_index = {}
                                    for t in self._full_dataset:
                                        _tid_val = t.get("task_id")
                                        if _tid_val is not None:
                                            self._dataset_index[str(_tid_val)] = t
                                            self._dataset_index[_tid_val] = t
                                    _ds_index = self._dataset_index
                                if _ds_index and _dtid in _ds_index:
                                    try:
                                        from areal.utils.difficulty_predictor import (
                                            DifficultyPredictor,
                                        )
                                        _task_text = DifficultyPredictor.format_input(
                                            _ds_index[_dtid]
                                        )
                                    except ImportError:
                                        pass
                                _tracker.update(
                                    _dtid, _n_success, len(_ep_rewards),
                                    task_text=_task_text,
                                )

                    return _RemoteRolloutResult(task_id=task_id, trajectory=traj)

                manager.on_rollout_rejected()
                return None

            except TimeoutError:
                if task_id is not None:
                    with self._futures_lock:
                        self._pending_futures.pop(task_id, None)
                manager.on_rollout_rejected()
                logger.error(f"Rollout timed out after {self.config.request_timeout}s")
                return None
            except Exception as exc:
                if task_id is not None:
                    with self._futures_lock:
                        self._pending_futures.pop(task_id, None)
                manager.on_rollout_rejected()
                logger.error("Workflow execution failed: %s", exc, exc_info=True)
                return None

        return _submit_then_wait

    def get_capacity(self):
        return self.staleness_manager.get_capacity()

    def submit(
        self,
        data: dict[str, Any],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: str | None = None,
        task_id: int | None = None,
        is_eval: bool = False,
        group_size: int = 1,
        proxy_addr: str | None = None,
    ) -> int:
        workflow_str = self._resolve_workflow_str(workflow)
        should_accept_fn = self._resolve_should_accept_fn(should_accept_fn)
        if workflow_kwargs is None:
            workflow_kwargs = {}

        # NOTE: RolloutController does not support `should_accept_fn`
        # If the workflow's result should be aborted,
        # `arun_episode` should return None instead.
        if task_id is None:
            task_id = self._task_id_generator.next()
        task_input = _RemoteRolloutTaskInput(
            data=data,
            workflow=workflow_str,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            task_id=task_id,
            is_eval=is_eval,
            group_size=group_size,
            proxy_addr=proxy_addr,
        )

        # Delegate to dispatcher
        self.dispatcher.submit_task_input(task_input)
        return task_id

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[dict[str, Any] | None]:
        # Delegate to dispatcher and extract trajectories
        results = self.dispatcher.wait_results(count, timeout, raise_timeout)
        # Log and trace
        if self.config.enable_rollout_tracing:
            logger.info("Rollout results are ready!")

        return [r.trajectory if r is not None else None for r in results]

    @trace_perf("rollout_controller.rollout_batch", category="scheduler")
    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: str | None = None,
        group_size: int = 1,
    ) -> dict[str, Any]:
        perf_tracer.instant(
            "rollout_controller.rollout_batch",
            category="scheduler",
            args={"data": len(data)},
        )
        for item in data:
            self.submit(
                data=item,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                should_accept_fn=should_accept_fn,
                group_size=group_size,
            )
        results = self.wait(count=len(data))
        # Concatenate into batch tensor format
        return concat_padded_tensors([r for r in results if r is not None])

    @trace_perf("rollout_controller.prepare_batch", category="scheduler")
    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ) -> dict[str, Any]:
        """Prepare a batch with controlled staleness.

        Continuously submits from dataloader and waits for results, ensuring at least
        two batches are pending to maximize overlap.

        See :meth:`~areal.api.engine_api.InferenceEngine.prepare_batch` for parameters.
        """

        workflow_str = self._resolve_workflow_str(workflow)
        if workflow_kwargs is None:
            workflow_kwargs = {}

        _tracker = getattr(self, "_task_success_tracker", None)
        _predictor = getattr(self, "_difficulty_predictor", None)
        _full_dataset = getattr(self, "_full_dataset", None)
        _target_p = float(getattr(self, "_difficulty_target_p", 0.3))

        def task_input_generator():
            import random as _random

            _skip_count = 0

            # If BERT predictor is available, use it to select tasks.
            # Every 6 tasks (= 1 training batch worth of GRPO groups),
            # re-sample 1024 random tasks, BERT-score them, and pick
            # the 6 closest to p=_target_p. This balances on-policy
            # freshness with amortized inference cost.
            _tasks_per_batch = max(
                1, getattr(self.config, "consumer_batch_size", 48) // max(group_size, 1)
            )
            if _predictor is not None and _full_dataset is not None:
                while True:
                    selected = _predictor.select_tasks(
                        _full_dataset,
                        sample_size=1024,
                        select_top_k=_tasks_per_batch,
                        epsilon=_tracker.epsilon if _tracker else 0.1,
                        target_p=_target_p,
                    )
                    if not selected:
                        # Fallback: random tasks from dataloader
                        for data in cycle_dataloader(dataloader):
                            for item in data:
                                _tid = self._task_id_generator.next()
                                pass  # task data looked up via _dataset_index
                                yield _RemoteRolloutTaskInput(
                                    data=item,
                                    workflow=workflow_str,
                                    workflow_kwargs=workflow_kwargs,
                                    should_accept_fn=should_accept_fn,
                                    task_id=_tid,
                                    group_size=group_size,
                                )
                            break  # yield one dataloader batch then re-try BERT
                        continue
                    for item in selected:
                        _tid = self._task_id_generator.next()
                        yield _RemoteRolloutTaskInput(
                            data=item,
                            workflow=workflow_str,
                            workflow_kwargs=workflow_kwargs,
                            should_accept_fn=should_accept_fn,
                            task_id=_tid,
                            group_size=group_size,
                        )

            # Fallback: EMA-based filtering + rejection sampling
            for data in cycle_dataloader(dataloader):
                for item in data:
                    if _tracker is not None:
                        dataset_tid = item.get("task_id", "")
                        if dataset_tid:
                            if not _tracker.should_sample(dataset_tid):
                                _skip_count += 1
                                if _skip_count % 100 == 0:
                                    logger.info(
                                        "[adaptive_sampling] skipped %d tasks so far",
                                        _skip_count,
                                    )
                                continue
                            priority = _tracker.sampling_priority(dataset_tid)
                            if priority < 1.0 and _random.random() > priority:
                                continue
                    _tid = self._task_id_generator.next()
                    yield _RemoteRolloutTaskInput(
                        data=item,
                        workflow=workflow_str,
                        workflow_kwargs=workflow_kwargs,
                        should_accept_fn=should_accept_fn,
                        task_id=_tid,
                        group_size=group_size,
                    )

        if not hasattr(self, "data_generator"):
            self.data_generator = task_input_generator()

        # Delegate to dispatcher
        assert dataloader.batch_size is not None
        import time as _time

        _pending_before = self.pending_results_count
        # When group_size > 1 (e.g. GRPO n_samples=8), each accepted result
        # is a concatenated dict of group_size trajectories. The number of
        # results to collect is consumer_batch_size / group_size, not
        # dataloader.batch_size.
        effective_batch_size = (
            self.config.consumer_batch_size // group_size
            if group_size > 1
            else dataloader.batch_size
        )
        _t0 = _time.perf_counter()
        results = self.dispatcher.active_submit_and_wait(
            self.data_generator, batch_size=effective_batch_size, dynamic_bs=dynamic_bs
        )
        _t1 = _time.perf_counter()

        # Extract trajectories and concatenate
        trajectories = [r.trajectory if r is not None else None for r in results]
        valid_trajectories = [t for t in trajectories if t is not None]
        self._last_consumed_count = len(valid_trajectories)

        # Localize RTensors before concat — trajectories from Ray RPC arrive
        # with RTensor wrappers; concat_padded_tensors expects plain torch.Tensors.
        # With the InferenceEngine RTensor fix, rollout results are plain numpy
        # arrays (no RTensors), so batch_localize_dicts hits the fast path.
        from areal.infra.rpc.rtensor import RTensor

        valid_trajectories = RTensor.batch_localize_dicts(valid_trajectories)
        _t_localize = _time.perf_counter()

        batch = concat_padded_tensors(valid_trajectories)
        _t2 = _time.perf_counter()

        _batch_shape = list(batch["input_ids"].shape) if "input_ids" in batch else "N/A"
        _ep_ids = (
            batch["episode_id"].unique().tolist() if "episode_id" in batch else "N/A"
        )
        _mmi_len = (
            len(batch["multi_modal_input"]) if "multi_modal_input" in batch else "N/A"
        )
        logger.info(
            "prepare_batch/%s: pending_before=%d, consumed=%d, "
            "wait=%.1fs, localize+concat=%.1fs, total=%.1fs, "
            "batch_shape=%s, n_episodes=%s, mmi_len=%s",
            self._worker_role,
            _pending_before,
            self._last_consumed_count,
            _t1 - _t0,
            _t2 - _t1,
            _t2 - _t0,
            _batch_shape,
            len(_ep_ids) if isinstance(_ep_ids, list) else _ep_ids,
            _mmi_len,
        )
        return batch

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request.

        This method provides direct access to the inference engine's generation capabilities
        for single requests, bypassing the workflow system.

        Parameters
        ----------
        req : ModelRequest
            The model request containing input data and generation parameters

        Returns
        -------
        ModelResponse
            The generated response from the model
        """
        # Choose worker and delegate
        worker, rank = self._choose_worker()

        # Call agenerate on engine via scheduler
        return await self.scheduler.async_call_engine(
            worker_id=worker.id,
            method="agenerate",
            engine_name=self._engine_name(rank),
            req=req,
        )

    async def init_weights_update_group(self, meta: WeightUpdateMeta) -> None:
        tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id,
                method="init_weights_update_group",
                engine_name=self._engine_name(rank),
                meta=meta,
                xccl_group_ranks=[rank],
            )
            for rank, worker in enumerate(self.workers)
        ]
        await asyncio.gather(*tasks)

    async def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ):
        await self._collective_rpc_async(
            "update_weights_from_distributed", meta=meta, param_specs=param_specs
        )

    async def update_weights_from_disk(self, meta: WeightUpdateMeta):
        meta.clear_checkpoint_after_load = False
        await self._collective_rpc_async("update_weights_from_disk", meta=meta)
        shutil.rmtree(meta.path, ignore_errors=True)

    async def pause_generation(self):
        await self._collective_rpc_async("pause_generation")

    async def continue_generation(self):
        await self._collective_rpc_async("continue_generation")

    def set_version(self, version: int) -> None:
        with self._version_lock:
            self._version = version
            self._collective_rpc("set_version", version=version, http_timeout=60.0)

    def seed_staleness_for_resume(self, version: int) -> None:
        """Seed the staleness manager's ``accepted`` counter to ``version * cbs``.

        Called from the recover handler so the capacity formula in
        ``StalenessManager.get_capacity()`` reproduces steady-state behaviour
        from the very first iteration of a resumed run instead of allowing a
        burst of stale rollouts.
        """
        if self._staleness_manager is not None:
            self._staleness_manager.seed_for_resume(version)

    def get_version(self) -> int:
        with self._version_lock:
            return self._version

    def pause(self):
        self.dispatcher.pause()
        self._collective_rpc("pause", http_timeout=60.0)

    def resume(self):
        self._collective_rpc("resume", http_timeout=60.0)
        self.dispatcher.resume()

    def export_stats(self) -> dict[str, float]:
        all_raw_stats = self._collective_rpc(method="export_stats", http_timeout=60.0)
        stats = defaultdict(float)
        counts = defaultdict(int)

        for raw_stats in all_raw_stats:
            for k, v in raw_stats.items():
                if k.endswith("__count"):
                    counts[k] += v
                else:
                    stats[k] += v * raw_stats.get(k + "__count", 0)

        # Average non-count stats
        final_stats = {}
        for k, v in stats.items():
            count_key = k + "__count"
            if count_key in counts and counts[count_key] > 0:
                final_stats[k] = v / counts[count_key]
        return final_stats

    def config_perf_tracer(self, config: PerfTracerConfig, role: str) -> None:
        async def _call():
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="config_perf_tracer",
                    engine_name=self._engine_name(rank),
                    rank=rank,
                    role=role,
                    config=config,
                )
                for rank, worker in enumerate(self.workers)
            ]
            return await asyncio.gather(*tasks)

        run_async_task(_call)

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        self._collective_rpc("save_perf_tracer", step=step, force=force)

    @property
    def staleness_manager(self):
        return self._staleness_manager

    @property
    def pending_results_count(self) -> int:
        """Number of completed trajectories waiting to be consumed by training."""
        if self._dispatcher is None:
            return 0
        return self._dispatcher.get_pending_results_count()

    @property
    def dispatcher(
        self,
    ) -> BatchTaskDispatcher[_RemoteRolloutTaskInput, _RemoteRolloutResult]:
        """Get the task dispatcher, ensuring initialization has been called."""
        if self._dispatcher is None:
            raise RuntimeError(
                "RolloutController.initialize() must be called before scheduling rollouts."
            )
        return self._dispatcher

    @property
    def runner(self):
        """For backward compatibility. The runner is now owned by the dispatcher."""
        return self.dispatcher.runner
