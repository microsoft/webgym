"""Real-time web dashboard for monitoring WebGym episode status.

Adapted from ``scaled_webgym/webgym/environment/task_monitor.py`` for AReaL's
async workflow executor, which creates episodes continuously (rolling window)
rather than pre-allocating all task slots upfront.

The dashboard is a single-page Flask app served on a configurable port with
SSE live updates. Each episode is shown as a colour-coded grid cell indicating
its current sub-operation (screenshot, inference, action, reward, …).
"""

import json
import logging
import socket
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any

from flask import Flask, Response, jsonify, render_template_string, request


class TaskStatus(Enum):
    NOT_STARTED = "N"
    WAITING_ALLOCATION = "W"
    RUNNING = "R"
    FINISHED = "F"
    FAILED = "X"


class SubStatus(Enum):
    NORMAL = ""
    NAVIGATING = "n"
    METADATA = "m"
    SCREENSHOT = "s"
    AC_TREE = "t"
    ACTION = "a"
    EXECUTING = "e"
    REWARD = "r"


@dataclass
class TaskProgress:
    task_id: str
    status: TaskStatus
    current_step: int
    max_steps: int
    sub_status: SubStatus = SubStatus.NORMAL
    start_time: float | None = None
    end_time: float | None = None
    task_name: str = ""
    current_operation: str = ""


class TaskMonitor:
    """Real-time task monitor with Flask web dashboard.

    All episodes are kept in the grid (no eviction). Finished and failed
    cells remain visible so the user can see the full history.

    Each rollout worker creates its own instance on a unique port
    (base_port + worker_rank).
    """

    def __init__(
        self,
        web_port: int = 5000,
        worker_rank: int = 0,
    ):
        self.web_port = web_port
        self.worker_rank = worker_rank

        # OrderedDict preserves insertion order
        self.tasks: OrderedDict[str, TaskProgress] = OrderedDict()
        self.lock = threading.Lock()

        # Cumulative counters
        self.total_registered = 0
        self.total_finished = 0
        self.total_failed = 0

        self.running = False
        self.monitor_start_time: float | None = None

        # Flask components
        self.web_app: Flask | None = None
        self.web_thread: threading.Thread | None = None
        self._setup_web_dashboard()

    # =========================
    # Lifecycle
    # =========================

    def start_monitoring(self) -> None:
        self.monitor_start_time = time.time()
        self.running = True

        self._try_start_web_server()

    def _try_start_web_server(self) -> None:
        """Attempt to bind the web server. Starts a retry thread if port is busy."""
        if self.web_thread is not None:
            return  # Already started or starting

        if self._port_in_use():
            # Port is held by another process (possibly stale). Retry periodically.
            retry = threading.Thread(target=self._retry_web_server_loop, daemon=True)
            retry.start()
            return

        self.web_thread = threading.Thread(target=self._run_web_server, daemon=True)
        self.web_thread.start()

        if self._wait_for_web_server(timeout=10):
            print(f"TaskMonitor dashboard running at http://0.0.0.0:{self.web_port}")

    def _retry_web_server_loop(self) -> None:
        """Periodically check if the port becomes free, then start the web server."""
        for _ in range(60):  # Retry for up to 5 minutes
            time.sleep(5)
            if not self.running:
                return
            if self.web_thread is not None:
                return  # Another thread started it
            if not self._port_in_use():
                self.web_thread = threading.Thread(
                    target=self._run_web_server, daemon=True
                )
                self.web_thread.start()
                if self._wait_for_web_server(timeout=10):
                    print(
                        f"TaskMonitor dashboard running at http://0.0.0.0:{self.web_port}"
                    )
                return

    def stop_monitoring(self) -> None:
        self.running = False

    # =========================
    # Task registration & status updates
    # =========================

    def register_task(self, task_id: str, task_name: str = "", max_steps: int = 10) -> str:
        """Register a new task and return a unique episode key.

        The episode key (e.g. ``"ep42"``) is auto-incrementing so that
        re-rollouts of the same ``task_id`` get separate grid cells.
        """
        if not self.lock.acquire(timeout=1.0):
            return f"ep{self.total_registered}"
        try:
            self.total_registered += 1
            episode_key = f"ep{self.total_registered}"
            self.tasks[episode_key] = TaskProgress(
                task_id=episode_key,
                status=TaskStatus.NOT_STARTED,
                current_step=0,
                max_steps=max_steps,
                task_name=task_name[:30],
            )
            return episode_key
        finally:
            self.lock.release()

    def start_allocation_wait(self, task_id: str, task_name: str = "") -> None:
        if not self.lock.acquire(blocking=False):
            return
        try:
            if task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.WAITING_ALLOCATION
                self.tasks[task_id].task_name = task_name[:30]
                self.tasks[task_id].current_operation = "Waiting for instance"
        finally:
            self.lock.release()

    def start_task(self, task_id: str, task_name: str = "") -> None:
        if not self.lock.acquire(timeout=1.0):
            return
        try:
            if task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.RUNNING
                self.tasks[task_id].current_step = 0
                self.tasks[task_id].sub_status = SubStatus.NORMAL
                self.tasks[task_id].start_time = time.time()
                self.tasks[task_id].task_name = task_name[:30]
                self.tasks[task_id].current_operation = "Task started"
        finally:
            self.lock.release()

    def update_task_step(self, task_id: str, step: int) -> None:
        if not self.lock.acquire(blocking=False):
            return
        try:
            if task_id in self.tasks:
                self.tasks[task_id].current_step = step
                self.tasks[task_id].status = TaskStatus.RUNNING
                self.tasks[task_id].sub_status = SubStatus.NORMAL
                self.tasks[task_id].current_operation = f"Step {step + 1}"
        finally:
            self.lock.release()

    def set_task_navigating(self, task_id: str, url: str = "") -> None:
        if not self.lock.acquire(blocking=False):
            return
        try:
            if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.RUNNING:
                self.tasks[task_id].sub_status = SubStatus.NAVIGATING
                self.tasks[task_id].current_operation = (
                    f"Navigating to {url[:40]}" if url else "Navigating"
                )
        finally:
            self.lock.release()

    def set_task_getting_metadata(self, task_id: str) -> None:
        if not self.lock.acquire(blocking=False):
            return
        try:
            if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.RUNNING:
                self.tasks[task_id].sub_status = SubStatus.METADATA
                self.tasks[task_id].current_operation = "Getting page metadata"
        finally:
            self.lock.release()

    def set_task_taking_screenshot(self, task_id: str, step: int | None = None) -> None:
        if not self.lock.acquire(blocking=False):
            return
        try:
            if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.RUNNING:
                self.tasks[task_id].sub_status = SubStatus.SCREENSHOT
                step_info = f" (step {step + 1})" if step is not None else ""
                self.tasks[task_id].current_operation = f"Taking screenshot{step_info}"
        finally:
            self.lock.release()

    def set_task_getting_ac_tree(self, task_id: str) -> None:
        if not self.lock.acquire(blocking=False):
            return
        try:
            if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.RUNNING:
                self.tasks[task_id].sub_status = SubStatus.AC_TREE
                self.tasks[task_id].current_operation = "Getting accessibility tree"
        finally:
            self.lock.release()

    def set_task_getting_action(self, task_id: str) -> None:
        if not self.lock.acquire(blocking=False):
            return
        try:
            if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.RUNNING:
                self.tasks[task_id].sub_status = SubStatus.ACTION
                self.tasks[task_id].current_operation = "Getting action from model"
        finally:
            self.lock.release()

    def set_task_executing_action(self, task_id: str, action: str = "") -> None:
        if not self.lock.acquire(blocking=False):
            return
        try:
            if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.RUNNING:
                self.tasks[task_id].sub_status = SubStatus.EXECUTING
                action_desc = f": {action[:30]}" if action else ""
                self.tasks[task_id].current_operation = f"Executing action{action_desc}"
        finally:
            self.lock.release()

    def set_task_computing_reward(self, task_id: str) -> None:
        if not self.lock.acquire(blocking=False):
            return
        try:
            if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.RUNNING:
                self.tasks[task_id].sub_status = SubStatus.REWARD
                self.tasks[task_id].current_operation = "Computing reward"
        finally:
            self.lock.release()

    def set_task_normal_phase(self, task_id: str) -> None:
        if not self.lock.acquire(blocking=False):
            return
        try:
            if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.RUNNING:
                self.tasks[task_id].sub_status = SubStatus.NORMAL
                self.tasks[task_id].current_operation = "Processing"
        finally:
            self.lock.release()

    def finish_task(self, task_id: str, success: bool = True, error_message: str = "") -> None:
        if not self.lock.acquire(timeout=1.0):
            return
        try:
            if task_id in self.tasks:
                self.tasks[task_id].status = (
                    TaskStatus.FINISHED if success else TaskStatus.FAILED
                )
                self.tasks[task_id].sub_status = SubStatus.NORMAL
                self.tasks[task_id].end_time = time.time()
                if success:
                    self.total_finished += 1
                    self.tasks[task_id].current_operation = "Completed"
                else:
                    self.total_failed += 1
                    self.tasks[task_id].current_operation = f"Failed: {error_message[:80]}"
        finally:
            self.lock.release()

    # =========================
    # Internal helpers
    # =========================

    def _elapsed_seconds(self) -> float:
        if self.monitor_start_time is None:
            return 0.0
        return max(0.0, time.time() - self.monitor_start_time)

    def _format_duration(self, seconds: float) -> str:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    # =========================
    # Snapshot
    # =========================

    def get_status_snapshot(self) -> dict[str, Any]:
        lock_acquired = self.lock.acquire(timeout=0.5)
        if not lock_acquired:
            return self._get_fallback_snapshot()

        try:
            task_items = list(self.tasks.items())
            elapsed = self._elapsed_seconds()
            counts = self._get_summary_counts_locked()
        finally:
            self.lock.release()

        tasks_data: dict[str, dict] = {}
        for tid, t in task_items:
            tasks_data[tid] = {
                "task_id": t.task_id,
                "status": t.status.value,
                "current_step": t.current_step,
                "max_steps": t.max_steps,
                "sub_status": t.sub_status.value,
                "task_name": t.task_name[:25] if t.task_name else "",
                "current_operation": t.current_operation[:50],
                "start_time": t.start_time,
                "end_time": t.end_time,
            }

        running_tasks = [t for _, t in task_items if t.status == TaskStatus.RUNNING]
        avg_step = (
            sum(t.current_step for t in running_tasks) / len(running_tasks)
            if running_tasks
            else 0.0
        )
        finished_tasks = [
            t
            for _, t in task_items
            if t.status == TaskStatus.FINISHED and t.start_time and t.end_time
        ]
        avg_duration = (
            sum(t.end_time - t.start_time for t in finished_tasks) / len(finished_tasks)
            if finished_tasks
            else 0.0
        )

        summary = {
            **counts,
            "avg_step": round(avg_step, 1),
            "avg_duration": round(avg_duration, 1),
            "total_registered": self.total_registered,
            "total_finished": self.total_finished,
            "total_failed": self.total_failed,
        }

        return {
            "tasks": tasks_data,
            "summary": summary,
            "total_tasks": self.total_registered,
            "elapsed_seconds": elapsed,
            "downsampled": False,
            "sample_size": len(tasks_data),
        }

    def _get_summary_counts_locked(self) -> dict[str, int]:
        not_started = waiting = running = finished = failed = 0
        navigating = metadata = screenshot = ac_tree = action = executing = reward = 0

        for t in self.tasks.values():
            if t.status == TaskStatus.NOT_STARTED:
                not_started += 1
            elif t.status == TaskStatus.WAITING_ALLOCATION:
                waiting += 1
            elif t.status == TaskStatus.RUNNING:
                running += 1
                if t.sub_status == SubStatus.NAVIGATING:
                    navigating += 1
                elif t.sub_status == SubStatus.METADATA:
                    metadata += 1
                elif t.sub_status == SubStatus.SCREENSHOT:
                    screenshot += 1
                elif t.sub_status == SubStatus.AC_TREE:
                    ac_tree += 1
                elif t.sub_status == SubStatus.ACTION:
                    action += 1
                elif t.sub_status == SubStatus.EXECUTING:
                    executing += 1
                elif t.sub_status == SubStatus.REWARD:
                    reward += 1
            elif t.status == TaskStatus.FINISHED:
                finished += 1
            elif t.status == TaskStatus.FAILED:
                failed += 1

        return {
            "not_started": not_started,
            "waiting_allocation": waiting,
            "running": running,
            "finished": finished,
            "failed": failed,
            "navigating": navigating,
            "metadata": metadata,
            "screenshot": screenshot,
            "ac_tree": ac_tree,
            "action": action,
            "executing": executing,
            "reward": reward,
        }

    def _get_fallback_snapshot(self) -> dict[str, Any]:
        return {
            "tasks": {},
            "summary": {
                "not_started": 0,
                "waiting_allocation": 0,
                "running": 0,
                "finished": 0,
                "failed": 0,
                "navigating": 0,
                "metadata": 0,
                "screenshot": 0,
                "ac_tree": 0,
                "action": 0,
                "executing": 0,
                "reward": 0,
                "avg_step": 0.0,
                "avg_duration": 0.0,
                "total_registered": self.total_registered,
                "total_finished": self.total_finished,
                "total_failed": self.total_failed,
            },
            "total_tasks": self.total_registered,
            "elapsed_seconds": self._elapsed_seconds(),
            "downsampled": False,
            "sample_size": 0,
            "lock_timeout": True,
        }

    # =========================
    # Flask web dashboard
    # =========================

    def _setup_web_dashboard(self) -> None:
        self.web_app = Flask(__name__)

        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

        @self.web_app.route("/")
        def dashboard():
            return render_template_string(
                _DASHBOARD_TEMPLATE, worker_rank=self.worker_rank
            )

        @self.web_app.route("/api/status")
        def get_status():
            return jsonify(self.get_status_snapshot())

        @self.web_app.route("/stream")
        def stream():
            try:
                interval_ms = int(request.args.get("interval", "1000"))
            except Exception:
                interval_ms = 1000
            interval_ms = max(100, min(interval_ms, 5000))

            def event_stream():
                payload = self.get_status_snapshot()
                yield _sse_message(payload)

                last_emit = 0.0
                while self.running:
                    now = time.time()
                    if (now - last_emit) * 1000.0 >= interval_ms:
                        payload = self.get_status_snapshot()
                        yield _sse_message(payload)
                        last_emit = now
                    time.sleep(0.01)

            headers = {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
            return Response(event_stream(), headers=headers)

    def _run_web_server(self) -> None:
        try:
            self.web_app.run(
                host="0.0.0.0",
                port=self.web_port,
                debug=False,
                use_reloader=False,
                threaded=True,
            )
        except OSError as e:
            if "Address already in use" in str(e) or "Errno 98" in str(e):
                # Another worker already owns the port — silently back off.
                pass
            else:
                print(f"TaskMonitor: OS error starting dashboard: {e}")
        except Exception as e:
            print(f"TaskMonitor: dashboard failed to start: {e}")
            import traceback

            traceback.print_exc()

    def _port_in_use(self) -> bool:
        """Check if the port is already bound by another process."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", self.web_port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def _wait_for_web_server(self, timeout: int = 10) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(("127.0.0.1", self.web_port))
                sock.close()
                if result == 0:
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        return False


def _sse_message(payload: dict) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


# ---------------------------------------------------------------------------
# HTML / CSS / JS dashboard template
# ---------------------------------------------------------------------------

_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Worker {{ worker_rank }} — AReaL WebGym Task Monitor</title>
  <style>
    :root{
      --cell: 40px;
      --gap: 6px;
      --cell-mobile: 32px;
      --gap-mobile: 4px;
      --border: 1px;
      --radius: 10px;
      --stroke: rgba(255,255,255,0.08);
    }
    *{margin:0;padding:0;box-sizing:border-box}
    body{
      font-family:'Monaco','Menlo','Ubuntu Mono',monospace;
      background:linear-gradient(135deg,#0c0c0c 0%,#1a1a1a 100%);
      color:#e0e0e0;min-height:100vh;padding:20px;
      -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;
    }
    .container{max-width:1400px;margin:0 auto}
    .header{
      background:linear-gradient(90deg,#2563eb 0%,#3b82f6 100%);
      padding:20px;border-radius:12px;margin-bottom:20px;
      box-shadow:0 8px 32px rgba(37,99,235,.3);
      display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;
    }
    .header h1{font-size:24px;margin-bottom:5px;text-shadow:0 2px 4px rgba(0,0,0,.3)}
    .header .info{font-size:14px;opacity:.9}
    .controls{display:flex;gap:10px;align-items:center}
    .refresh-btn,.mode-badge{
      background:rgba(255,255,255,.2);border:none;color:#fff;padding:10px 12px;
      border-radius:8px;font-family:inherit;font-size:14px;transition:all .3s;
    }
    .refresh-btn:hover{background:rgba(255,255,255,.3);transform:translateY(-2px)}
    .refresh-btn:disabled{opacity:.5;cursor:not-allowed}
    .mode-badge{cursor:default}

    .grid-container{
      background:rgba(30,30,30,.9);padding:25px;border-radius:12px;margin-bottom:20px;
      backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,.1);
      overflow:auto;min-height:200px;position:relative;
    }
    .grid-title{font-size:16px;margin-bottom:15px;color:#60a5fa;font-weight:bold}

    /* GRID */
    .task-grid{
      display:grid;gap:var(--gap);justify-content:center;margin-bottom:15px;width:100%;
      position:relative;grid-auto-flow:row;
    }

    /* CELLS */
    .task-cell{
      width:var(--cell);height:var(--cell);aspect-ratio:1/1;
      border:var(--border) solid var(--stroke);border-radius:var(--radius);
      display:grid;place-items:center;
      line-height:1;
      cursor:pointer;position:relative;overflow:visible;
      font-weight:bold;font-size:13px;
      font-variant-numeric:tabular-nums;
      transition:box-shadow .15s ease;
      user-select:none;
    }
    .task-cell:hover{ box-shadow:0 0 0 2px rgba(255,255,255,.18), 0 6px 18px rgba(0,0,0,.35); z-index:20 }
    .task-cell > span{display:block;line-height:1;pointer-events:none}

    /* States */
    .task-cell.not-started{ background:#374151; color:#9ca3af }
    .task-cell.waiting{ background:linear-gradient(45deg,#fbbf24,#f59e0b); color:#1f2937 }
    .task-cell.running{ background:linear-gradient(45deg,#3b82f6,#1d4ed8); color:#fff }
    .task-cell.running.navigating{ background:linear-gradient(45deg,#8b5cf6,#7c3aed) }
    .task-cell.running.metadata{ background:linear-gradient(45deg,#06b6d4,#0891b2) }
    .task-cell.running.screenshot{ background:linear-gradient(45deg,#10b981,#059669) }
    .task-cell.running.ac-tree{ background:linear-gradient(45deg,#f59e0b,#d97706) }
    .task-cell.running.action{ background:linear-gradient(45deg,#d946ef,#c026d3) }
    .task-cell.running.executing{ background:linear-gradient(45deg,#ec4899,#db2777) }
    .task-cell.running.reward{ background:linear-gradient(45deg,#eab308,#ca8a04) }
    .task-cell.finished{ background:linear-gradient(45deg,#22c55e,#16a34a); color:#fff }
    .task-cell.failed{ background:linear-gradient(45deg,#ef4444,#dc2626); color:#fff }

    /* TOOLTIP */
    .task-cell .tooltip{
      position:absolute; display:none;
      bottom:110%; left:50%; transform:translateX(-50%);
      padding:8px 10px;border-radius:6px;background:rgba(0,0,0,.92);color:#fff;
      font-size:12px;line-height:1.3;white-space:pre-line;pointer-events:none;
      max-width:300px; z-index:1000; box-shadow:0 8px 28px rgba(0,0,0,.45);
    }
    .task-cell .tooltip::after{
      content:""; position:absolute; top:100%; left:50%; transform:translateX(-50%);
      border-width:6px; border-style:solid; border-color:rgba(0,0,0,.92) transparent transparent transparent;
    }
    .task-cell:hover .tooltip{ display:block }

    /* SUMMARY */
    .summary{
      background:rgba(30,30,30,.9); padding:25px; border-radius:12px;
      backdrop-filter:blur(10px); border:1px solid rgba(255,255,255,.1);
    }
    .summary h2{ color:#60a5fa; margin-bottom:15px; font-size:16px }
    .summary-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:15px; margin-bottom:15px }
    .summary-item{ background:rgba(55,65,81,.5); padding:12px; border-radius:6px; border-left:3px solid #60a5fa }
    .summary-item .label{ font-size:11px; color:#9ca3af; margin-bottom:3px }
    .summary-item .value{ font-size:18px; font-weight:bold; color:#e5e7eb }

    .legend{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:8px; margin-top:15px }
    .legend-item{ display:flex; align-items:center; gap:6px; font-size:12px }
    .legend-color{ width:16px; height:16px; border-radius:3px; border:var(--border) solid var(--stroke) }

    @media (max-width:768px){
      .task-cell{ width:var(--cell-mobile); height:var(--cell-mobile) }
      .task-grid{ gap:var(--gap-mobile) }
      .header{ flex-direction:column; gap:10px }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h1>Worker {{ worker_rank }} — AReaL WebGym Task Monitor</h1>
        <div class="info">
          <span id="elapsedTime">Elapsed: 00:00</span> |
          <span id="totalTasks">Episodes: 0</span> |
          <span id="lastUpdate">Last update: --</span>
        </div>
      </div>
      <div class="controls">
        <span class="mode-badge">Live mode: SSE</span>
        <button class="refresh-btn" onclick="manualRefresh()" id="refreshBtn" title="Fallback pull refresh">
          Pull Refresh
        </button>
      </div>
    </div>

    <div class="grid-container">
      <div class="grid-title">Episode Progress Grid</div>
      <div class="task-grid" id="taskGrid"></div>
    </div>

    <div class="summary">
      <h2>Summary Statistics</h2>

      <div style="margin-bottom:8px;color:#9ca3af;font-size:12px;font-weight:bold;">Episode Status</div>
      <div class="summary-grid">
        <div class="summary-item"><div class="label">Waiting</div><div class="value" id="waitingAllocation">0</div></div>
        <div class="summary-item"><div class="label">Running</div><div class="value" id="running">0</div></div>
        <div class="summary-item"><div class="label">Finished (total)</div><div class="value" id="finished">0</div></div>
        <div class="summary-item"><div class="label">Failed (total)</div><div class="value" id="failed">0</div></div>
        <div class="summary-item"><div class="label">Avg Duration</div><div class="value" id="avgDuration">--</div></div>
      </div>

      <div style="margin:15px 0 8px 0;color:#9ca3af;font-size:12px;font-weight:bold;">Current Operations</div>
      <div class="summary-grid">
        <div class="summary-item"><div class="label">Navigating</div><div class="value" id="navigating">0</div></div>
        <div class="summary-item"><div class="label">Screenshots</div><div class="value" id="screenshot">0</div></div>
        <div class="summary-item"><div class="label">Inference</div><div class="value" id="action">0</div></div>
        <div class="summary-item"><div class="label">Executing</div><div class="value" id="executing">0</div></div>
        <div class="summary-item"><div class="label">Reward</div><div class="value" id="reward">0</div></div>
      </div>

      <div class="legend">
        <div class="legend-item"><div class="legend-color" style="background:linear-gradient(45deg,#fbbf24,#f59e0b);"></div><span>W: Waiting</span></div>
        <div class="legend-item"><div class="legend-color" style="background:linear-gradient(45deg,#3b82f6,#1d4ed8);"></div><span>0-9: Running</span></div>
        <div class="legend-item"><div class="legend-color" style="background:linear-gradient(45deg,#8b5cf6,#7c3aed);"></div><span>Xn: Navigate</span></div>
        <div class="legend-item"><div class="legend-color" style="background:linear-gradient(45deg,#06b6d4,#0891b2);"></div><span>Xm: Metadata</span></div>
        <div class="legend-item"><div class="legend-color" style="background:linear-gradient(45deg,#10b981,#059669);"></div><span>Xs: Screenshot</span></div>
        <div class="legend-item"><div class="legend-color" style="background:linear-gradient(45deg,#d946ef,#c026d3);"></div><span>Xa: Inference</span></div>
        <div class="legend-item"><div class="legend-color" style="background:linear-gradient(45deg,#ec4899,#db2777);"></div><span>Xe: Execute</span></div>
        <div class="legend-item"><div class="legend-color" style="background:linear-gradient(45deg,#eab308,#ca8a04);"></div><span>Xr: Reward</span></div>
        <div class="legend-item"><div class="legend-color" style="background:linear-gradient(45deg,#22c55e,#16a34a);"></div><span>F: Finished</span></div>
        <div class="legend-item"><div class="legend-color" style="background:linear-gradient(45deg,#ef4444,#dc2626);"></div><span>X: Failed</span></div>
      </div>
    </div>
  </div>

  <script>
    function formatDuration(seconds){
      const h=Math.floor(seconds/3600);
      const m=Math.floor((seconds%3600)/60);
      const s=Math.floor(seconds%60);
      if(h>0) return h+':'+String(m).padStart(2,'0')+':'+String(s).padStart(2,'0');
      return String(m).padStart(2,'0')+':'+String(s).padStart(2,'0');
    }

    function calculateGridDimensions(taskCount){
      const containerWidth = window.innerWidth < 768 ? window.innerWidth - 60 : 1400;
      const cell = window.innerWidth < 768 ? 32 : 40;
      const gap = window.innerWidth < 768 ? 4 : 6;
      const maxCols = Math.floor((containerWidth + gap) / (cell + gap));
      const cols = Math.min(maxCols, taskCount, 30);
      const rows = Math.ceil(taskCount / cols);
      return { cols, rows, cell, gap };
    }

    function getDisplayCharacter(task){
      if(task.status==='N') return 'N';
      if(task.status==='W') return 'W';
      if(task.status==='F') return 'F';
      if(task.status==='X') return 'X';
      if(task.status==='R'){
        const baseChar = String(task.current_step % 10);
        const subChar = task.sub_status;
        return subChar ? baseChar + subChar : baseChar;
      }
      return '?';
    }

    function getTaskClass(task){
      const classes=['task-cell'];
      if(task.status==='N') classes.push('not-started');
      else if(task.status==='W') classes.push('waiting');
      else if(task.status==='F') classes.push('finished');
      else if(task.status==='X') classes.push('failed');
      else if(task.status==='R'){
        classes.push('running');
        if(task.sub_status==='n') classes.push('navigating');
        else if(task.sub_status==='m') classes.push('metadata');
        else if(task.sub_status==='s') classes.push('screenshot');
        else if(task.sub_status==='t') classes.push('ac-tree');
        else if(task.sub_status==='a') classes.push('action');
        else if(task.sub_status==='e') classes.push('executing');
        else if(task.sub_status==='r') classes.push('reward');
      }
      return classes.join(' ');
    }

    function createTooltip(task){
      let tip='Task: '+task.task_id;
      if(task.task_name) tip+='\\nName: '+task.task_name;
      tip+='\\nStatus: '+task.status;
      if(task.current_operation) tip+='\\nOp: '+task.current_operation;
      if(task.status==='R'){
        tip+='\\nStep: '+task.current_step+'/'+task.max_steps;
        const names={n:'Navigating',m:'Metadata',s:'Screenshot',t:'AC tree',a:'Inference',e:'Executing',r:'Reward'};
        if(task.sub_status && names[task.sub_status]) tip+='\\n'+names[task.sub_status];
      }
      if(task.start_time){
        const dur=(task.end_time||Date.now()/1000)-task.start_time;
        tip+='\\nDuration: '+dur.toFixed(1)+'s';
      }
      return tip;
    }

    function updateDisplay(data){
      document.getElementById('elapsedTime').textContent='Elapsed: '+formatDuration(data.elapsed_seconds);
      document.getElementById('totalTasks').textContent='Episodes: '+data.total_tasks;
      document.getElementById('lastUpdate').textContent='Last update: '+new Date().toLocaleTimeString();

      const s=data.summary;
      document.getElementById('waitingAllocation').textContent=s.waiting_allocation;
      document.getElementById('running').textContent=s.running;
      document.getElementById('finished').textContent=s.total_finished||s.finished;
      document.getElementById('failed').textContent=s.total_failed||s.failed;
      document.getElementById('avgDuration').textContent=s.avg_duration>0?s.avg_duration.toFixed(1)+'s':'--';

      document.getElementById('navigating').textContent=s.navigating||0;
      document.getElementById('screenshot').textContent=s.screenshot||0;
      document.getElementById('action').textContent=s.action||0;
      document.getElementById('executing').textContent=s.executing||0;
      document.getElementById('reward').textContent=s.reward||0;

      const taskGrid=document.getElementById('taskGrid');
      const tasks=Object.values(data.tasks);
      if(tasks.length===0){taskGrid.innerHTML='<div style="color:#666;padding:20px;">No episodes yet</div>';return;}
      const {cols,rows,cell,gap}=calculateGridDimensions(tasks.length);

      taskGrid.style.gridTemplateColumns='repeat('+cols+', '+cell+'px)';
      taskGrid.style.gridTemplateRows='repeat('+rows+', '+cell+'px)';
      taskGrid.style.gap=gap+'px';
      taskGrid.innerHTML='';

      tasks.forEach(function(task){
        const cellDiv=document.createElement('div');
        cellDiv.className=getTaskClass(task);
        const charSpan=document.createElement('span');
        charSpan.textContent=getDisplayCharacter(task);
        cellDiv.appendChild(charSpan);
        const tooltip=document.createElement('div');
        tooltip.className='tooltip';
        tooltip.textContent=createTooltip(task);
        cellDiv.appendChild(tooltip);
        taskGrid.appendChild(cellDiv);
      });

      window.lastDisplayData=data;
    }

    let evtSource,pullTimer;
    function startSSE(){
      try{
        evtSource=new EventSource('/stream?interval=2000');
        evtSource.onmessage=function(event){updateDisplay(JSON.parse(event.data));};
        evtSource.onerror=function(){try{evtSource.close();}catch(e){}startPullFallback();};
      }catch(e){startPullFallback();}
    }
    function startPullFallback(){
      clearInterval(pullTimer);
      pullTimer=setInterval(function(){
        fetch('/api/status').then(function(r){return r.json();}).then(updateDisplay).catch(function(){});
      },2000);
    }
    function manualRefresh(){
      var btn=document.getElementById('refreshBtn');
      btn.disabled=true;btn.textContent='Loading...';
      fetch('/api/status').then(function(r){return r.json();}).then(function(d){
        updateDisplay(d);btn.disabled=false;btn.textContent='Pull Refresh';
      }).catch(function(){btn.disabled=false;btn.textContent='Pull Refresh';});
    }

    document.addEventListener('DOMContentLoaded',function(){startSSE();});

    var resizeTimeout;
    window.addEventListener('resize',function(){
      clearTimeout(resizeTimeout);
      resizeTimeout=setTimeout(function(){if(window.lastDisplayData)updateDisplay(window.lastDisplayData);},250);
    });
  </script>
</body>
</html>
"""
