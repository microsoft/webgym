"""
OmniBoxes Process Manager

Manages the lifecycle of OmniBoxes servers including Redis, instance servers,
node server, and master server with automatic process recovery.

This module provides the OmniboxesLauncher class which handles:
- Starting/stopping Redis, instance servers, node server, master server
- Parallel instance startup for fast scaling
- Automatic process recovery on failure
- Graceful shutdown

Use deploy.py as the CLI entry point, which wraps this module.
"""

import atexit
import subprocess
import signal
import sys
import time
import threading
import os
import concurrent.futures
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ProcessInfo:
    """Information about a managed process"""
    process: subprocess.Popen
    cmd: List[str]
    name: str
    restart_count: int = 0
    last_restart: float = 0.0
    max_restarts: int = 5
    restart_delay: float = 2.0


class OmniboxesLauncher:
    """Launches and manages OmniBoxes servers"""

    def __init__(self, num_instances: int):
        self.num_instances = num_instances
        self.processes: Dict[str, ProcessInfo] = {}
        self.instance_start_port = 9000
        self.node_port = 8080
        self.master_port = 7000
        self.redis_port = 6379
        self.redis_host = 'localhost'
        self.node_workers = 32
        self.master_workers = 32
        self.redis_pid: Optional[int] = None
        self._we_started_redis = False
        self._cleaned_up = False
        self.shutdown_event = threading.Event()
        self.recovery_enabled = True
        self.max_parallel_starts = min(50, max(10, num_instances // 10))
        self.startup_batch_delay = 0.1

    def kill_existing_processes(self):
        """Kill any existing processes on the ports we need before launching."""
        print("🧹 Killing existing processes on required ports...")

        ports_to_free = set()

        # Instance server ports
        for i in range(self.num_instances):
            ports_to_free.add(self.instance_start_port + i)

        # Node and master ports
        ports_to_free.add(self.node_port)
        ports_to_free.add(self.master_port)

        killed = 0
        for port in sorted(ports_to_free):
            try:
                result = subprocess.run(
                    ["lsof", "-ti", f"tcp:{port}"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        pid = pid.strip()
                        if pid:
                            try:
                                os.kill(int(pid), signal.SIGKILL)
                                killed += 1
                            except (ProcessLookupError, ValueError):
                                pass
            except Exception:
                pass

        if killed:
            print(f"   Killed {killed} process(es) on {len(ports_to_free)} ports")
            time.sleep(1)
        else:
            print("   No existing processes found")

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}. Shutting down all processes...")
            self.shutdown_event.set()
            self.recovery_enabled = False
            # Don't call cleanup() here - let the finally block in run() handle it.
            # Calling cleanup() + sys.exit() here would cause double-cleanup since
            # sys.exit() raises SystemExit which still triggers the finally block.

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Safety net: ensure Redis is stopped even on unexpected exit paths
        atexit.register(self._atexit_cleanup)

    def start_redis(self) -> bool:
        """Start Redis server as a daemon"""
        print(f"🔴 Starting Redis server on port {self.redis_port}...")

        try:
            if self.is_redis_running():
                print(f"⚠️  Redis is already running on port {self.redis_port}")
                return True

            script_dir = os.path.dirname(os.path.abspath(__file__))
            redis_conf = os.path.join(script_dir, "redis.conf")
            redis_data_dir = os.path.join(script_dir, "redis-data")

            cmd = [
                "redis-server", redis_conf,
                "--port", str(self.redis_port),
                "--dir", redis_data_dir,
                "--daemonize", "yes",
            ]

            print(f"   Command: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                time.sleep(1)
                if self.is_redis_running():
                    self.redis_pid = self.get_redis_pid()
                    self._we_started_redis = True
                    print(f"✅ Redis server started successfully (PID: {self.redis_pid})")
                    return True
                else:
                    print("❌ Redis failed to start (not responding)")
                    return False
            else:
                print(f"❌ Failed to start Redis: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("❌ Redis startup timed out")
            return False
        except FileNotFoundError:
            print("❌ Redis not found. Please install Redis server")
            return False
        except Exception as e:
            print(f"❌ Error starting Redis: {e}")
            return False

    def is_redis_running(self) -> bool:
        """Check if Redis is running on the configured port"""
        try:
            result = subprocess.run(
                ["redis-cli", "-p", str(self.redis_port), "ping"],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0 and "PONG" in result.stdout
        except Exception:
            return False

    def get_redis_pid(self) -> Optional[int]:
        """Get the PID of the Redis process"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", f"redis-server.*{self.redis_port}"],
                capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip().split('\n')[0])
        except Exception:
            pass
        return None

    def stop_redis(self):
        """Stop the Redis server we started.

        Uses a three-stage fallback: redis-cli shutdown -> kill by PID -> pkill.
        Only stops Redis if we started it (avoids killing a pre-existing instance).
        """
        if not self._we_started_redis:
            return

        if not self.is_redis_running():
            return

        print("🔴 Stopping Redis server...")

        # Stage 1: Graceful shutdown via redis-cli
        try:
            subprocess.run(
                ["redis-cli", "-p", str(self.redis_port), "shutdown", "nosave"],
                capture_output=True, text=True, timeout=5
            )
            time.sleep(1)
            if not self.is_redis_running():
                print("✅ Redis stopped gracefully")
                return
        except Exception:
            pass

        # Stage 2: Kill by PID (refresh PID in case it changed)
        pid = self.redis_pid or self.get_redis_pid()
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)
                try:
                    os.kill(pid, 0)  # Check if still alive
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(0.5)
                    print("✅ Redis force-stopped (SIGKILL)")
                except ProcessLookupError:
                    print("✅ Redis stopped (SIGTERM)")
                return
            except ProcessLookupError:
                pass  # Already dead, but verify below
            except Exception as e:
                print(f"⚠️  Kill by PID failed: {e}")

        # Stage 3: pkill fallback (catches any Redis on our port)
        if self.is_redis_running():
            try:
                subprocess.run(
                    ["pkill", "-f", f"redis-server.*:{self.redis_port}"],
                    capture_output=True, text=True, timeout=5
                )
                time.sleep(1)
                if not self.is_redis_running():
                    print("✅ Redis stopped (pkill)")
                else:
                    print("⚠️  Redis may still be running on port %d" % self.redis_port)
            except Exception:
                print("⚠️  Could not stop Redis")

    def start_single_instance_server(self, instance_index: int) -> Tuple[bool, str]:
        """Start a single instance server (for parallel execution)"""
        port = self.instance_start_port + instance_index
        cmd = ["python", "-m", "omniboxes.node.instance_server", "--port", str(port), "--redis_host", self.redis_host, "--redis_port", str(self.redis_port)]
        name = f"Instance-{instance_index+1}:{port}"

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1
            )

            process_info = ProcessInfo(process=process, cmd=cmd, name=name)
            self.processes[name] = process_info

            output_thread = threading.Thread(
                target=self.handle_output, args=(process, name), daemon=True
            )
            output_thread.start()

            return True, f"✅ Instance server {instance_index+1} started on port {port}"

        except Exception as e:
            return False, f"❌ Failed to start instance server {instance_index+1}: {e}"

    def start_instance_servers_parallel(self) -> bool:
        """Start all instance servers in parallel batches"""
        print(f"📦 Starting {self.num_instances} instance servers in parallel...")
        print(f"   Using {self.max_parallel_starts} parallel workers")

        success_count = 0
        failed_instances = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_starts) as executor:
            future_to_index = {
                executor.submit(self.start_single_instance_server, i): i
                for i in range(self.num_instances)
            }

            for future in concurrent.futures.as_completed(future_to_index):
                if self.shutdown_event.is_set():
                    break

                instance_index = future_to_index[future]
                try:
                    success, message = future.result()
                    if success:
                        success_count += 1
                        if success_count % 50 == 0 or success_count <= 10:
                            print(f"   Progress: {success_count}/{self.num_instances} started")
                    else:
                        failed_instances.append(instance_index)
                        print(message)
                except Exception as e:
                    failed_instances.append(instance_index)
                    print(f"❌ Exception starting instance {instance_index+1}: {e}")

        if failed_instances:
            print(f"⚠️  Failed to start {len(failed_instances)} instance servers")
            return False

        print(f"✅ All {success_count} instance servers started successfully")
        return True

    def start_process(self, cmd: List[str], name: str) -> subprocess.Popen:
        """Start a subprocess and add it to the process list"""
        try:
            print(f"🚀 Starting {name}...")
            print(f"   Command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1
            )

            process_info = ProcessInfo(process=process, cmd=cmd, name=name)
            self.processes[name] = process_info

            output_thread = threading.Thread(
                target=self.handle_output, args=(process, name), daemon=True
            )
            output_thread.start()

            return process

        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")
            return None

    def handle_output(self, process: subprocess.Popen, name: str):
        """Handle output from a subprocess"""
        while True:
            try:
                line = process.stdout.readline()
                if not line:
                    break
                print(f"[{name}] {line.rstrip()}")
            except Exception:
                break

    def start_node_server(self):
        """Start the node server"""
        print(f"🔗 Starting node server on port {self.node_port} with {self.node_workers} workers...")

        cmd = [
            "python", "-m", "omniboxes.node.server",
            "--port", str(self.node_port),
            "--workers", str(self.node_workers),
            "--redis_host", self.redis_host,
            "--redis_port", str(self.redis_port)
        ]

        process = self.start_process(cmd, f"Node:{self.node_port}")

        if process:
            print(f"✅ Node server started on port {self.node_port} with {self.node_workers} workers")
            return True
        else:
            print("❌ Failed to start node server")
            return False

    def start_master_server(self):
        """Start the master server"""
        print(f"👑 Starting master server on port {self.master_port} with {self.master_workers} workers...")

        cmd = [
            "python", "-m", "omniboxes.master.server",
            "--port", str(self.master_port),
            "--nodes", f"http://localhost:{self.node_port}",
            "--workers", str(self.master_workers)
        ]

        process = self.start_process(cmd, f"Master:{self.master_port}")

        if process:
            print(f"✅ Master server started on port {self.master_port} with {self.master_workers} workers")
            return True
        else:
            print("❌ Failed to start master server")
            return False

    def restart_process(self, name: str, process_info: ProcessInfo) -> bool:
        """Restart a failed process with exponential backoff"""
        current_time = time.time()

        if process_info.restart_count >= process_info.max_restarts:
            print(f"🚫 Process {name} has exceeded maximum restart attempts ({process_info.max_restarts})")
            return False

        if current_time - process_info.last_restart < process_info.restart_delay:
            return False

        delay = process_info.restart_delay * (2 ** process_info.restart_count)
        if current_time - process_info.last_restart < delay:
            return False

        print(f"🔄 Restarting process {name} (attempt {process_info.restart_count + 1}/{process_info.max_restarts})")

        try:
            new_process = subprocess.Popen(
                process_info.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1
            )

            process_info.process = new_process
            process_info.restart_count += 1
            process_info.last_restart = current_time

            output_thread = threading.Thread(
                target=self.handle_output, args=(new_process, name), daemon=True
            )
            output_thread.start()

            print(f"✅ Process {name} restarted successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to restart process {name}: {e}")
            return False

    def check_and_restart_redis(self):
        """Check if Redis is still running and restart it if needed"""
        if not self._we_started_redis:
            return True

        if self.is_redis_running():
            return True

        print("⚠️  Redis is down! Restarting...")
        if self.start_redis():
            print("✅ Redis restarted successfully")
            return True
        else:
            print("❌ Failed to restart Redis")
            return False

    def check_and_restart_processes(self):
        """Check processes and restart failed ones"""
        if not self.recovery_enabled:
            return True

        # Check Redis daemon (not tracked in self.processes)
        self.check_and_restart_redis()

        dead_processes = []
        restarted_processes = []

        for name, process_info in list(self.processes.items()):
            if process_info.process and process_info.process.poll() is not None:
                dead_processes.append(name)

                if self.restart_process(name, process_info):
                    restarted_processes.append(name)
                else:
                    del self.processes[name]

        if dead_processes:
            if restarted_processes:
                print(f"🔄 Restarted {len(restarted_processes)} of {len(dead_processes)} failed processes")
            else:
                print(f"⚠️  Warning: {len(dead_processes)} process(es) terminated and could not be restarted")
            return len(restarted_processes) > 0

        return True

    def wait_for_startup(self):
        """Wait a moment for servers to initialize"""
        print("⏳ Waiting for servers to initialize...")
        time.sleep(2)

    def _atexit_cleanup(self):
        """Safety-net called by atexit to ensure Redis is stopped."""
        if not self._cleaned_up:
            self.cleanup()

    def flush_redis_pool(self):
        """Remove instance pool keys from Redis to prevent stale entries on next startup.

        Without this, a non-graceful shutdown (kill -9, crash) leaves orphaned
        port entries in the 'available' and 'in_use' sets, inflating the
        capacity reported on the next run.
        """
        if not self.is_redis_running():
            return

        print("🧹 Flushing instance pool from Redis...")
        try:
            result = subprocess.run(
                ["redis-cli", "-p", str(self.redis_port), "DEL", "available", "in_use"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                print("✅ Redis instance pool flushed")
            else:
                print(f"⚠️  Failed to flush Redis pool: {result.stderr}")
        except Exception as e:
            print(f"⚠️  Error flushing Redis pool: {e}")

    def cleanup(self):
        """Terminate all running processes including Redis"""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        print("🧹 Cleaning up processes...")

        if self.processes:
            for name, process_info in self.processes.items():
                if process_info.process and process_info.process.poll() is None:
                    try:
                        print(f"   Terminating process {name}...")
                        process_info.process.terminate()
                    except Exception as e:
                        print(f"   Error terminating process {name}: {e}")

            time.sleep(3)

            for name, process_info in self.processes.items():
                if process_info.process and process_info.process.poll() is None:
                    try:
                        print(f"   Force killing process {name}...")
                        process_info.process.kill()
                        process_info.process.wait()
                    except Exception as e:
                        print(f"   Error force killing process {name}: {e}")

            self.processes.clear()

        self.flush_redis_pool()
        self.stop_redis()
        print("✅ All processes cleaned up")

    def run(self):
        """Main execution method"""
        print("=" * 60)
        print("🎯 OMNIBOXES MULTI-SERVER LAUNCHER")
        print("=" * 60)
        print(f"📊 Configuration:")
        print(f"   • Redis server port: {self.redis_port}")
        print(f"   • Instance servers: {self.num_instances}")
        print(f"   • Instance ports: {self.instance_start_port}-{self.instance_start_port + self.num_instances - 1}")
        print(f"   • Node server port: {self.node_port}")
        print(f"   • Master server port: {self.master_port}")
        print(f"   • Parallel startup workers: {self.max_parallel_starts}")
        print(f"   • Process recovery: {'Enabled' if self.recovery_enabled else 'Disabled'}")
        print("=" * 60)

        instance_port_range = range(self.instance_start_port, self.instance_start_port + self.num_instances)
        reserved_ports = [self.redis_port, self.node_port, self.master_port]

        for port in reserved_ports:
            if port in instance_port_range:
                print(f"❌ ERROR: Port conflict detected!")
                print(f"   Instance servers will use ports {self.instance_start_port}-{self.instance_start_port + self.num_instances - 1}")
                print(f"   This conflicts with reserved port {port}")
                print(f"\n💡 Solution: Use a different instance start port or reduce number of instances")
                print(f"   Example: --instance-start-port {max(reserved_ports) + 1}")
                sys.exit(1)

        self.kill_existing_processes()
        self.setup_signal_handlers()

        try:
            if not self.start_redis():
                raise Exception("Failed to start Redis server")

            if not self.start_instance_servers_parallel():
                raise Exception("Failed to start instance servers")

            self.wait_for_startup()

            if not self.start_node_server():
                raise Exception("Failed to start node server")

            time.sleep(1)

            if not self.start_master_server():
                raise Exception("Failed to start master server")

            print("=" * 60)
            print("🎉 All servers started successfully!")
            print("📋 Server Status:")
            print(f"   • Redis server running on port {self.redis_port}")
            print(f"   • {self.num_instances} instance server(s) running")
            print(f"   • Node server running on http://localhost:{self.node_port}")
            print(f"   • Master server running on http://localhost:{self.master_port}")
            print("=" * 60)
            print("💡 Press Ctrl+C to stop all servers")
            print("🔄 Automatic process recovery is enabled")
            print("=" * 60)

            while not self.shutdown_event.is_set():
                time.sleep(5)
                self.check_and_restart_processes()

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cleanup()
