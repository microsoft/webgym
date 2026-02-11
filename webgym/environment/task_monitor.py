import asyncio
import threading
import time
import json
import signal
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Web dashboard imports
from flask import Flask, Response, render_template_string, jsonify, request
import logging

class TaskStatus(Enum):
    NOT_STARTED = 'N'
    WAITING_ALLOCATION = 'W'  # Waiting for browser allocation
    RUNNING = 'R'  # We'll use numbers 0-9 for actual steps
    FINISHED = 'F'
    FAILED = 'X'

class SubStatus(Enum):
    NORMAL = ''  # Normal step processing (default)
    NAVIGATING = 'n'  # Xn: navigating to URL
    METADATA = 'm'   # Xm: getting metadata
    SCREENSHOT = 's'  # Xs: taking screenshot  
    AC_TREE = 't'    # Xt: getting accessibility tree
    ACTION = 'a'     # Xa: getting action from vLLM
    EXECUTING = 'e'  # Xe: executing browser action
    REWARD = 'r'     # Xr: computing reward

@dataclass
class TaskProgress:
    task_id: str
    status: TaskStatus
    current_step: int
    max_steps: int
    sub_status: SubStatus = SubStatus.NORMAL
    start_time: float = None
    end_time: float = None
    task_name: str = ""
    current_operation: str = ""  # Description of current operation

class TaskMonitor:
    def __init__(self, total_tasks: int, max_steps: int = 10, enable_web_dashboard: bool = True, web_port: int = 5000, max_tracked_tasks: int = None):
        self.total_tasks = total_tasks
        self.max_steps = max_steps
        self.enable_web_dashboard = enable_web_dashboard
        self.web_port = web_port

        self.tasks: Dict[str, TaskProgress] = {}
        self.running = False
        self.lock = threading.Lock()
        self.monitor_start_time: Optional[float] = None

        # Web dashboard components
        self.web_app = None
        self.web_thread = None
        self.web_server_process = None  # Store server process ID for shutdown

        # Track all tasks for statistics (no sampling)
        for i in range(total_tasks):
            task_id = f"task_{i:04d}"
            self.tasks[task_id] = TaskProgress(
                task_id=task_id,
                status=TaskStatus.NOT_STARTED,
                current_step=0,
                max_steps=max_steps,
                sub_status=SubStatus.NORMAL
            )

        print(f"📊 TaskMonitor: Tracking all {len(self.tasks)} tasks")

        if self.enable_web_dashboard:
            self._setup_web_dashboard()

        self.task_error_details: Dict[str, str] = {}  # task_id -> error message
        
    
    def start_monitoring(self):
        """Start the monitoring system (web only, no terminal)"""
        self.monitor_start_time = time.time()
        self.running = True

        # Start web dashboard if enabled
        if self.enable_web_dashboard:
            print(f"🌐 Starting web dashboard on port {self.web_port}...")
            self.web_thread = threading.Thread(target=self._run_web_server, daemon=True)
            self.web_thread.start()

            # Wait for server to actually start with verification
            if not self._wait_for_web_server(timeout=10):
                print(f"⚠️ Web dashboard may not have started successfully on port {self.web_port}")
            else:
                print(f"✅ Web dashboard running at http://0.0.0.0:{self.web_port}")

    def _wait_for_web_server(self, timeout=10):
        """Wait for web server to actually start responding"""
        import socket
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to connect to the port
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', self.web_port))
                sock.close()
                if result == 0:
                    return True
            except:
                pass
            time.sleep(0.5)
        return False

    def stop_monitoring(self):
        """Stop the monitoring system including Flask web server"""
        print("📡 Stopping monitoring system...")
        self.running = False

        # Set flag to stop SSE streams - Flask daemon thread will exit with main program
        if self.web_thread and self.web_thread.is_alive():
            print("🌐 Flask web server running as daemon (will exit with main program)")

        print("✅ Monitoring system stopped")
    
    # =========================
    # Enhanced Task Status Methods
    # =========================
    
    def start_allocation_wait(self, task_id: str, task_name: str = ""):
        """Mark a task as waiting for browser allocation. Non-blocking - skips update if busy."""
        if not self.lock.acquire(blocking=False):  # Try immediate, skip if busy
            return
        try:
            if task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.WAITING_ALLOCATION
                self.tasks[task_id].task_name = task_name[:30]
                self.tasks[task_id].current_operation = "Waiting for instance allocation"
        finally:
            self.lock.release()

    def start_task(self, task_id: str, task_name: str = ""):
        """Mark a task as started. Uses short timeout for important state change."""
        if not self.lock.acquire(timeout=1.0):  # Short timeout, this is important
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

    def update_task_step(self, task_id: str, step: int):
        """Update the current step of a task. Non-blocking - skips if busy."""
        if not self.lock.acquire(blocking=False):
            return
        try:
            if task_id in self.tasks:
                self.tasks[task_id].current_step = min(step, self.max_steps - 1)
                self.tasks[task_id].status = TaskStatus.RUNNING
                self.tasks[task_id].sub_status = SubStatus.NORMAL
                self.tasks[task_id].current_operation = f"Processing step {step + 1}"
        finally:
            self.lock.release()

    # Fine-grained operation tracking methods - all non-blocking for minimal contention
    def set_task_navigating(self, task_id: str, url: str = ""):
        """Mark task as navigating to URL. Non-blocking."""
        if not self.lock.acquire(blocking=False):
            return
        try:
            if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.RUNNING:
                self.tasks[task_id].sub_status = SubStatus.NAVIGATING
                self.tasks[task_id].current_operation = f"Navigating to {url[:40]}..." if url else "Navigating to website"
        finally:
            self.lock.release()

    def set_task_getting_metadata(self, task_id: str):
        """Mark task as getting metadata. Non-blocking."""
        if not self.lock.acquire(blocking=False):
            return
        try:
            if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.RUNNING:
                self.tasks[task_id].sub_status = SubStatus.METADATA
                self.tasks[task_id].current_operation = "Getting page metadata"
        finally:
            self.lock.release()

    def set_task_taking_screenshot(self, task_id: str, step: int = None):
        """Mark task as taking screenshot. Non-blocking."""
        if not self.lock.acquire(blocking=False):
            return
        try:
            if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.RUNNING:
                self.tasks[task_id].sub_status = SubStatus.SCREENSHOT
                step_info = f" (step {step + 1})" if step is not None else ""
                self.tasks[task_id].current_operation = f"Taking screenshot{step_info}"
        finally:
            self.lock.release()

    def set_task_getting_ac_tree(self, task_id: str):
        """Mark task as getting accessibility tree. Non-blocking."""
        if not self.lock.acquire(blocking=False):
            return
        try:
            if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.RUNNING:
                self.tasks[task_id].sub_status = SubStatus.AC_TREE
                self.tasks[task_id].current_operation = "Getting accessibility tree"
        finally:
            self.lock.release()

    def set_task_getting_action(self, task_id: str):
        """Mark task as getting action from vLLM. Non-blocking."""
        if not self.lock.acquire(blocking=False):
            return
        try:
            if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.RUNNING:
                self.tasks[task_id].sub_status = SubStatus.ACTION
                self.tasks[task_id].current_operation = "Getting action from vLLM"
        finally:
            self.lock.release()

    def set_task_executing_action(self, task_id: str, action: str = ""):
        """Mark task as executing browser action. Non-blocking."""
        if not self.lock.acquire(blocking=False):
            return
        try:
            if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.RUNNING:
                self.tasks[task_id].sub_status = SubStatus.EXECUTING
                action_desc = f": {action[:30]}" if action else ""
                self.tasks[task_id].current_operation = f"Executing action{action_desc}"
        finally:
            self.lock.release()

    def set_task_computing_reward(self, task_id: str):
        """Mark task as computing final reward. Non-blocking."""
        if not self.lock.acquire(blocking=False):
            return
        try:
            if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.RUNNING:
                self.tasks[task_id].sub_status = SubStatus.REWARD
                self.tasks[task_id].current_operation = "Computing final reward"
        finally:
            self.lock.release()

    def set_task_normal_phase(self, task_id: str):
        """Mark a task as in normal processing. Non-blocking."""
        if not self.lock.acquire(blocking=False):
            return
        try:
            if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.RUNNING:
                self.tasks[task_id].sub_status = SubStatus.NORMAL
                self.tasks[task_id].current_operation = "Processing"
        finally:
            self.lock.release()

    def finish_task(self, task_id: str, success: bool = True, error_message: str = ""):
        """Mark a task as finished. Uses short timeout for important state change."""
        if not self.lock.acquire(timeout=1.0):  # Short timeout, this is important
            return
        try:
            if task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.FINISHED if success else TaskStatus.FAILED
                self.tasks[task_id].sub_status = SubStatus.NORMAL
                self.tasks[task_id].end_time = time.time()

                if success:
                    self.tasks[task_id].current_operation = "Completed successfully"
                else:
                    self.tasks[task_id].current_operation = f"Failed: {error_message}"

                if not success and error_message:
                    self.task_error_details[task_id] = error_message
        finally:
            self.lock.release()

    def get_progress_summary(self) -> Dict:
        """Get current progress summary (CLI helper)."""
        # Use same snapshot logic to minimize lock time
        snapshot = self.get_status_snapshot()
        summary_data = snapshot["summary"]
        summary_data.update({
            'elapsed_seconds': snapshot['elapsed_seconds'],
            'elapsed_hms': self._format_duration(snapshot['elapsed_seconds'])
        })
        return summary_data

    # =========================
    # Web dashboard setup (Enhanced to show fine-grained status)
    # =========================
    def _setup_web_dashboard(self):
        """Initialize Flask web application for the dashboard with enhanced status display."""
        self.web_app = Flask(__name__)

        # Suppress Flask logging to avoid cluttering console
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        @self.web_app.route('/')
        def dashboard():
            return render_template_string(ENHANCED_WEB_DASHBOARD_TEMPLATE)

        @self.web_app.route('/api/status')
        def get_status():
            """Pull-based fallback endpoint (JSON)."""
            snapshot = self.get_status_snapshot()
            return jsonify(snapshot)

        @self.web_app.route('/stream')
        def stream():
            """Server-Sent Events stream for live updates."""
            try:
                interval_ms = int(request.args.get('interval', '1000'))
            except Exception:
                interval_ms = 1000
            interval_ms = max(100, min(interval_ms, 5000))  # clamp 100ms..5s

            def event_stream():
                # Initial burst delivers one snapshot immediately
                payload = self.get_status_snapshot()
                yield _sse_message(payload)

                # Periodic updates
                last_emit = 0.0
                while self.running:  # Check if monitoring is still running
                    now = time.time()
                    if (now - last_emit) * 1000.0 >= interval_ms:
                        payload = self.get_status_snapshot()
                        yield _sse_message(payload)
                        last_emit = now
                    # Tiny sleep to avoid busy loop
                    time.sleep(0.01)

            headers = {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
            return Response(event_stream(), headers=headers)

        @self.web_app.route('/shutdown', methods=['POST'])
        def shutdown():
            """Shutdown endpoint for graceful server termination"""
            print("🛑 Flask shutdown endpoint called")
            # Try werkzeug shutdown if available
            func = request.environ.get('werkzeug.server.shutdown')
            if func is not None:
                func()
                return jsonify({'status': 'shutting down'}), 200
            else:
                # Werkzeug shutdown not available in newer versions
                # Just return success - the daemon thread will exit when main exits
                print("⚠️ Werkzeug shutdown not available (newer Flask version)")
                return jsonify({'status': 'acknowledged'}), 200

    def _run_web_server(self):
        """Run Flask web server in separate thread."""
        try:
            print(f"🚀 Flask attempting to bind to 0.0.0.0:{self.web_port}")
            self.web_app.run(
                host='0.0.0.0',
                port=self.web_port,
                debug=False,
                use_reloader=False,
                threaded=True
            )
        except OSError as e:
            if "Address already in use" in str(e) or "Errno 48" in str(e) or "Errno 98" in str(e):
                print(f"❌ Port {self.web_port} is still in use! Error: {e}")
                print(f"   Try: lsof -ti:{self.web_port} | xargs kill -9")
            else:
                print(f"❌ OS error starting web dashboard: {e}")
        except Exception as e:
            print(f"❌ Web dashboard failed to start: {e}")
            import traceback
            traceback.print_exc()

    # =========================
    # Snapshot helpers (Enhanced for fine-grained status)
    # =========================
    def get_status_snapshot(self) -> Dict:
        """
        Build a full snapshot for the dashboard while minimizing lock hold time.
        Uses short timeout to avoid blocking - returns fallback if busy.
        """
        # Try to acquire lock with very short timeout (dashboard updates are non-critical)
        lock_acquired = self.lock.acquire(timeout=0.5)

        if not lock_acquired:
            # Lock busy - return fallback data (silent, no warning spam)
            return self._get_fallback_snapshot()

        try:
            # Copy minimal data under lock
            task_items: List[Tuple[str, TaskProgress]] = list(self.tasks.items())
            total_tasks = self.total_tasks
            elapsed_seconds = self._elapsed_seconds()
            # Summary counts under lock for speed/consistency
            summary_counts = self._get_summary_counts_locked()
        finally:
            self.lock.release()

        # Downsample & build tasks payload off-lock
        tasks_data = self._build_downsampled_payload(task_items, total_tasks)

        # Build summary dict off-lock
        summary = self._build_summary_from_items(task_items, summary_counts)

        return {
            'tasks': tasks_data,
            'summary': summary,
            'total_tasks': total_tasks,
            'elapsed_seconds': elapsed_seconds,
            'downsampled': len(tasks_data) < total_tasks,
            'sample_size': len(tasks_data),
        }

    def _get_summary_counts_locked(self) -> Dict[str, int]:
        """Compute status counters including fine-grained sub-status counts."""
        not_started = waiting = running = finished = failed = 0
        
        # Fine-grained operation counts
        navigating = metadata = screenshot = ac_tree = action = executing = reward = 0
        
        for t in self.tasks.values():
            if t.status == TaskStatus.NOT_STARTED:
                not_started += 1
            elif t.status == TaskStatus.WAITING_ALLOCATION:
                waiting += 1
            elif t.status == TaskStatus.RUNNING:
                running += 1
                # Count sub-operations
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
            'not_started': not_started,
            'waiting_allocation': waiting,
            'running': running,
            'finished': finished,
            'failed': failed,
            'navigating': navigating,
            'metadata': metadata,
            'screenshot': screenshot,
            'ac_tree': ac_tree,
            'action': action,
            'executing': executing,
            'reward': reward,
        }

    def _build_summary_from_items(self, task_items: List[Tuple[str, TaskProgress]], counts: Dict[str, int]) -> Dict:
        """Compute averages from copied task list."""
        running_tasks = [t for _, t in task_items if t.status == TaskStatus.RUNNING]
        avg_step = (sum(t.current_step for t in running_tasks) / len(running_tasks)) if running_tasks else 0.0

        finished_tasks = [t for _, t in task_items if t.status == TaskStatus.FINISHED and t.start_time and t.end_time]
        avg_duration = (sum(t.end_time - t.start_time for t in finished_tasks) / len(finished_tasks)) if finished_tasks else 0.0

        return {
            **counts,
            'avg_step': round(avg_step, 1),
            'avg_duration': round(avg_duration, 1),
            'total': self.total_tasks
        }

    def _build_downsampled_payload(self, task_items: List[Tuple[str, TaskProgress]], total_tasks: int, max_display_tasks: int = 512) -> Dict[str, Dict]:
        """Enhanced payload building with fine-grained status display."""
        # Check the actual number of tracked tasks, not total_tasks
        num_tracked = len(task_items)

        if num_tracked <= max_display_tasks:
            # Display all tracked tasks
            tasks_data: Dict[str, Dict] = {}
            for task_id, task in task_items:
                tasks_data[task_id] = {
                    'task_id': task.task_id,
                    'status': task.status.value,
                    'current_step': task.current_step,
                    'max_steps': task.max_steps,
                    'sub_status': task.sub_status.value,
                    'task_name': task.task_name[:25] if task.task_name else "",
                    'current_operation': task.current_operation[:50],
                    'start_time': task.start_time,
                    'end_time': task.end_time,
                }
            return tasks_data
        else:
            # Further downsample if we have more tracked tasks than max_display_tasks
            step = max(1, num_tracked // max_display_tasks)
            tasks_data: Dict[str, Dict] = {}
            task_items_sorted = sorted(task_items, key=lambda kv: kv[0])
            for i in range(0, len(task_items_sorted), step):
                task_id, task = task_items_sorted[i]
                tasks_data[f"sample_{i//step}"] = {
                    'task_id': f"{task.task_id}+{step-1}" if step > 1 else task.task_id,
                    'status': task.status.value,
                    'current_step': task.current_step,
                    'max_steps': task.max_steps,
                    'sub_status': task.sub_status.value,
                    'task_name': f"{task.task_name[:20]}..." if task.task_name else "",
                    'current_operation': f"{task.current_operation[:30]}..." if task.current_operation else "",
                    'start_time': task.start_time,
                    'end_time': task.end_time,
                    'represents_tasks': step
                }
            return tasks_data

    # =========================
    # Internal helpers
    # =========================
    def _format_duration(self, seconds: float) -> str:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _elapsed_seconds(self) -> float:
        if self.monitor_start_time is None:
            return 0.0
        return max(0.0, time.time() - self.monitor_start_time)

    def _get_fallback_snapshot(self) -> Dict:
        """
        Return a minimal fallback snapshot when lock cannot be acquired.
        This prevents the dashboard from hanging when there's lock contention.
        """
        return {
            'tasks': {},
            'summary': {
                'not_started': 0,
                'waiting_allocation': 0,
                'running': 0,
                'finished': 0,
                'failed': 0,
                'navigating': 0,
                'metadata': 0,
                'screenshot': 0,
                'ac_tree': 0,
                'action': 0,
                'executing': 0,
                'reward': 0,
                'avg_step': 0.0,
                'avg_duration': 0.0,
                'total': self.total_tasks
            },
            'total_tasks': self.total_tasks,
            'elapsed_seconds': self._elapsed_seconds(),
            'downsampled': True,
            'sample_size': 0,
            'lock_timeout': True  # Flag to indicate this is fallback data
        }


def _sse_message(payload: Dict) -> str:
    """Format a JSON payload as an SSE message."""
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


ENHANCED_WEB_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>WebGym Enhanced Task Monitor</title>
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
    .downsample-info{
      font-size:12px;color:#fbbf24;margin-bottom:10px;padding:5px 10px;
      background:rgba(251,191,36,.1);border-radius:4px;border-left:3px solid #fbbf24;
    }

    /* GRID */
    .task-grid{
      display:grid;gap:var(--gap);justify-content:center;margin-bottom:15px;width:100%;
      position:relative;grid-auto-flow:row;
    }

    /* CELLS — uniform sizing & true centering */
    .task-cell{
      width:var(--cell);height:var(--cell);aspect-ratio:1/1;
      border:var(--border) solid var(--stroke);border-radius:var(--radius);
      display:grid;place-items:center;        /* exact centering */
      line-height:1;                           /* tight line box */
      cursor:pointer;position:relative;overflow:visible;
      font-weight:bold;font-size:13px;
      font-variant-numeric:tabular-nums;
      transition:box-shadow .15s ease;
      user-select:none;
    }
    .task-cell:hover{ box-shadow:0 0 0 2px rgba(255,255,255,.18), 0 6px 18px rgba(0,0,0,.35); z-index:20 }

    /* ensure glyph itself doesn't introduce baseline quirks */
    .task-cell > span{
      display:block;
      line-height:1;
      pointer-events:none;
    }

    /* States (backgrounds only; border identical for all) */
    .task-cell.not-started{ background:#374151; color:#9ca3af }
    .task-cell.waiting{ background:linear-gradient(45deg,#fbbf24,#f59e0b); color:#1f2937 }
    .task-cell.running{ background:linear-gradient(45deg,#3b82f6,#1d4ed8); color:#fff }
    .task-cell.running.navigating{ background:linear-gradient(45deg,#8b5cf6,#7c3aed) } /* Purple for navigation */
    .task-cell.running.metadata{ background:linear-gradient(45deg,#06b6d4,#0891b2) }   /* Cyan for metadata */
    .task-cell.running.screenshot{ background:linear-gradient(45deg,#10b981,#059669) }  /* Green for screenshot */
    .task-cell.running.ac-tree{ background:linear-gradient(45deg,#f59e0b,#d97706) }     /* Orange for AC tree */
    .task-cell.running.action{ background:linear-gradient(45deg,#d946ef,#c026d3) }      /* Magenta for action */
    .task-cell.running.executing{ background:linear-gradient(45deg,#ec4899,#db2777) }   /* Pink for executing */
    .task-cell.running.reward{ background:linear-gradient(45deg,#eab308,#ca8a04) }      /* Yellow for reward */
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

    /* MOBILE */
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
        <h1>WebGym Enhanced Task Monitor</h1>
        <div class="info">
          <span id="elapsedTime">Elapsed: 00:00</span> |
          <span id="totalTasks">Total Tasks: 0</span> |
          <span id="lastUpdate">Last update: --</span>
        </div>
      </div>
      <div class="controls">
        <span class="mode-badge">Live mode: SSE</span>
        <button class="refresh-btn" onclick="manualRefresh()" id="refreshBtn" title="Fallback pull refresh">
          🔄 Pull Refresh
        </button>
      </div>
    </div>

    <div class="grid-container">
      <div class="grid-title">Task Progress Grid (Enhanced Status)</div>
      <div id="downsampleInfo" class="downsample-info" style="display:none;"></div>
      <div class="task-grid" id="taskGrid"></div>
    </div>

    <div class="summary">
      <h2>Summary Statistics</h2>

      <!-- Row 1: Task Status Metadata -->
      <div style="margin-bottom:8px;color:#9ca3af;font-size:12px;font-weight:bold;">Task Status</div>
      <div class="summary-grid">
        <div class="summary-item"><div class="label">Not Started</div><div class="value" id="notStarted">0</div></div>
        <div class="summary-item"><div class="label">Waiting</div><div class="value" id="waitingAllocation">0</div></div>
        <div class="summary-item"><div class="label">Running</div><div class="value" id="running">0</div></div>
        <div class="summary-item"><div class="label">Failed</div><div class="value" id="failed">0</div></div>
        <div class="summary-item"><div class="label">Finished</div><div class="value" id="finished">0</div></div>
      </div>

      <!-- Row 2: Once-per-task operations -->
      <div style="margin:15px 0 8px 0;color:#9ca3af;font-size:12px;font-weight:bold;">Once-per-task Operations</div>
      <div class="summary-grid">
        <div class="summary-item"><div class="label">Navigating</div><div class="value" id="navigating">0</div></div>
        <div class="summary-item"><div class="label">Computing Reward</div><div class="value" id="reward">0</div></div>
      </div>

      <!-- Row 3: Per-step operations -->
      <div style="margin:15px 0 8px 0;color:#9ca3af;font-size:12px;font-weight:bold;">Per-step Operations</div>
      <div class="summary-grid">
        <div class="summary-item"><div class="label">Getting Metadata</div><div class="value" id="metadata">0</div></div>
        <div class="summary-item"><div class="label">Taking Screenshots</div><div class="value" id="screenshot">0</div></div>
        <div class="summary-item"><div class="label">Getting AC Tree</div><div class="value" id="acTree">0</div></div>
        <div class="summary-item"><div class="label">Getting vLLM Action</div><div class="value" id="action">0</div></div>
        <div class="summary-item"><div class="label">Executing Action</div><div class="value" id="executing">0</div></div>
      </div>

      <div class="legend">
        <div class="legend-item"><div class="legend-color" style="background:#374151;"></div><span>N: Not Started</span></div>
        <div class="legend-item"><div class="legend-color" style="background:linear-gradient(45deg,#fbbf24,#f59e0b);"></div><span>W: Waiting</span></div>
        <div class="legend-item"><div class="legend-color" style="background:linear-gradient(45deg,#3b82f6,#1d4ed8);"></div><span>0-9: Running</span></div>
        <div class="legend-item"><div class="legend-color" style="background:linear-gradient(45deg,#8b5cf6,#7c3aed);"></div><span>Xn: Navigate</span></div>
        <div class="legend-item"><div class="legend-color" style="background:linear-gradient(45deg,#06b6d4,#0891b2);"></div><span>Xm: Metadata</span></div>
        <div class="legend-item"><div class="legend-color" style="background:linear-gradient(45deg,#10b981,#059669);"></div><span>Xs: Screenshot</span></div>
        <div class="legend-item"><div class="legend-color" style="background:linear-gradient(45deg,#f59e0b,#d97706);"></div><span>Xt: AC Tree</span></div>
        <div class="legend-item"><div class="legend-color" style="background:linear-gradient(45deg,#d946ef,#c026d3);"></div><span>Xa: vLLM Action</span></div>
        <div class="legend-item"><div class="legend-color" style="background:linear-gradient(45deg,#ef4444,#dc2626);"></div><span>Xe: Execute</span></div>
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
      if(h>0) return `${h}:${m.toString().padStart(2,'0')}:${s.toString().padStart(2,'0')}`;
      return `${m.toString().padStart(2,'0')}:${s.toString().padStart(2,'0')}`;
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
      if(task.status==='W') {
        // Show step 0 + sub_status for waiting allocation (0n for navigating, 0m for metadata, etc.)
        const subChar = task.sub_status;
        return subChar ? '0' + subChar : '0';
      }
      if(task.status==='F') return 'F';
      if(task.status==='X') return 'X';
      if(task.status==='R'){
        const baseChar = task.current_step.toString();
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
      let tooltip=`Task: ${task.task_id}`;
      if(task.task_name) tooltip+=`\\nName: ${task.task_name}`;
      if(task.represents_tasks && task.represents_tasks>1) tooltip+=`\\nRepresents: ${task.represents_tasks} tasks`;
      tooltip+=`\\nStatus: ${task.status}`;
      if(task.current_operation) tooltip+=`\\nOperation: ${task.current_operation}`;
      if(task.status==='R'){
        tooltip+=`\\nStep: ${task.current_step}`;
        const subStatusNames = {
          'n': 'Navigating to URL',
          'm': 'Getting metadata', 
          's': 'Taking screenshot',
          't': 'Getting AC tree',
          'a': 'Getting vLLM action',
          'e': 'Executing action',
          'r': 'Computing reward'
        };
        if(task.sub_status && subStatusNames[task.sub_status]) {
          tooltip += `\\n${subStatusNames[task.sub_status]}`;
        }
      }
      if(task.start_time){
        const duration=(task.end_time || Date.now()/1000)-task.start_time;
        tooltip+=`\\nDuration: ${duration.toFixed(1)}s`;
      }
      return tooltip;
    }

    function updateDisplay(data){
      document.getElementById('elapsedTime').textContent=`Elapsed: ${formatDuration(data.elapsed_seconds)}`;
      document.getElementById('totalTasks').textContent=`Total Tasks: ${data.total_tasks}`;
      document.getElementById('lastUpdate').textContent=`Last update: ${new Date().toLocaleTimeString()}`;

      const downsampleInfo=document.getElementById('downsampleInfo');
      if(data.downsampled){
        downsampleInfo.style.display='block';
        downsampleInfo.textContent=`⚡ Performance mode: Showing ${data.sample_size} representative tasks out of ${data.total_tasks} total`;
      }else{
        downsampleInfo.style.display='none';
      }

      // Update summary stats including fine-grained operations
      // Row 1: Task Status
      document.getElementById('notStarted').textContent=data.summary.not_started;
      document.getElementById('waitingAllocation').textContent=data.summary.waiting_allocation;
      document.getElementById('running').textContent=data.summary.running;
      document.getElementById('finished').textContent=data.summary.finished;
      document.getElementById('failed').textContent=data.summary.failed;

      // Row 2: Once-per-task operations
      document.getElementById('navigating').textContent=data.summary.navigating || 0;
      document.getElementById('reward').textContent=data.summary.reward || 0;

      // Row 3: Per-step operations
      document.getElementById('metadata').textContent=data.summary.metadata || 0;
      document.getElementById('screenshot').textContent=data.summary.screenshot || 0;
      document.getElementById('acTree').textContent=data.summary.ac_tree || 0;
      document.getElementById('action').textContent=data.summary.action || 0;
      document.getElementById('executing').textContent=data.summary.executing || 0;

      const taskGrid=document.getElementById('taskGrid');
      const tasks=Object.values(data.tasks);
      const { cols, rows, cell, gap } = calculateGridDimensions(tasks.length);

      taskGrid.style.gridTemplateColumns=`repeat(${cols}, ${cell}px)`;
      taskGrid.style.gridTemplateRows=`repeat(${rows}, ${cell}px)`;
      taskGrid.style.gap=`${gap}px`;
      taskGrid.innerHTML='';

      tasks.forEach(task=>{
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

    // SSE live updates with pull fallback
    let evtSource, pullTimer;
    function startSSE(){
      try{
        evtSource=new EventSource('/stream?interval=2000'); // Balanced update frequency to reduce lock contention
        evtSource.onmessage=(event)=>{
          const data=JSON.parse(event.data);
          updateDisplay(data);
        };
        evtSource.onerror=(err)=>{
          console.error('SSE error, switching to pull fallback...',err);
          try{evtSource.close();}catch{}
          startPullFallback();
        };
      }catch(e){
        console.error('Failed to start SSE, using pull fallback',e);
        startPullFallback();
      }
    }
    function startPullFallback(){
      clearInterval(pullTimer);
      pullTimer=setInterval(async()=>{
        try{
          const resp=await fetch('/api/status');
          const data=await resp.json();
          updateDisplay(data);
        }catch(e){ console.error('Pull fallback fetch failed:',e); }
      },2000);
    }
    async function manualRefresh(){
      const btn=document.getElementById('refreshBtn');
      btn.disabled=true; btn.textContent='🔄 Loading...';
      try{
        const resp=await fetch('/api/status');
        const data=await resp.json();
        updateDisplay(data);
      }catch(e){
        alert('Failed to refresh. See console for details.');
        console.error(e);
      }finally{
        btn.disabled=false; btn.textContent='🔄 Pull Refresh';
      }
    }

    document.addEventListener('DOMContentLoaded',()=>{ startSSE(); });

    // Resize reflow
    let resizeTimeout;
    window.addEventListener('resize',function(){
      clearTimeout(resizeTimeout);
      resizeTimeout=setTimeout(function(){
        if(window.lastDisplayData) updateDisplay(window.lastDisplayData);
      },250);
    });
  </script>
</body>
</html>
"""
