import multiprocessing
import signal
import time
import os
import pickle
import traceback
from typing import Any, Callable, Optional
import threading
import queue
import psutil
import concurrent.futures
from concurrent.futures import Future




class ProcessIsolator:
    """
    Execute functions in separate processes with unified timeout and retry configuration.
    Uses operation_timeout and max_retries from rollout.yaml for consistency.
    """

    def __init__(self, max_workers: int = 256, wait_timeout: float = 3600.0, operation_timeout: float = 120, max_retries: int = 2):
        self.max_workers = max_workers
        self.wait_timeout = wait_timeout
        self.operation_timeout = operation_timeout
        self.max_retries = max_retries
        self.process_pool = None
        self._shutdown = False
        self._process_lock = threading.Lock()
        self._pool_generation = 0
        self._pool_start_attempts = 0
        self._max_pool_start_attempts = 5

        print(f"🔧 ProcessIsolator initialized with max_workers={max_workers}, wait_timeout={wait_timeout}s, operation_timeout={operation_timeout}s, max_retries={max_retries}")
        
    def start(self):
        """Start the process pool with retry logic"""
        with self._process_lock:
            if self.process_pool is None and not self._shutdown:
                self._create_new_pool_with_retry()
                print(f"🚀 Process isolator started with {self.max_workers} processes")
    
    def _create_new_pool_with_retry(self):
        """Create a new process pool with retry logic"""
        for attempt in range(self._max_pool_start_attempts):
            try:
                # Use standard multiprocessing pool
                self.process_pool = multiprocessing.Pool(
                    processes=self.max_workers,  # Use exact config value
                    initializer=self._worker_init
                )
                self._pool_generation += 1
                self._pool_start_attempts = 0  # Reset on success
                print(f"✅ Created new process pool (generation {self._pool_generation}) with {self.max_workers} workers")
                return
                
            except Exception as e:
                self._pool_start_attempts += 1
                print(f"❌ Failed to create pool (attempt {attempt + 1}/{self._max_pool_start_attempts}): {e}")
                
                if attempt < self._max_pool_start_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise Exception(f"Failed to create process pool after {self._max_pool_start_attempts} attempts")
    
    def stop(self):
        """Stop the process pool aggressively"""
        print("🛑 Stopping process isolator...")
        self._shutdown = True

        # Use a timeout for the entire stop operation
        import threading

        def _do_stop():
            with self._process_lock:
                if self.process_pool:
                    try:
                        # Force kill all workers first (don't wait for terminate)
                        killed_count = 0
                        if hasattr(self.process_pool, '_pool'):
                            for p in self.process_pool._pool:
                                try:
                                    if p.is_alive():
                                        print(f"🔪 Force killing worker {p.pid}")
                                        self._kill_process_tree(p.pid)
                                        killed_count += 1
                                except Exception as e:
                                    print(f"⚠️ Error checking/killing worker: {e}")

                        print(f"   Killed {killed_count} workers")

                    except Exception as e:
                        print(f"⚠️ Error during process pool shutdown: {e}")
                    finally:
                        self.process_pool = None

        # Run stop with 10 second timeout
        stop_thread = threading.Thread(target=_do_stop, daemon=True)
        stop_thread.start()
        stop_thread.join(timeout=10)

        if stop_thread.is_alive():
            print("⚠️ Stop operation timed out after 10s - continuing anyway")
            self.process_pool = None  # Force cleanup

        print("✅ Process isolator stopped")
    
    def _kill_process_tree(self, pid):
        """Kill a process and all its children - fast and non-blocking"""
        try:
            # Just send SIGKILL directly - don't wait
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        except Exception as e:
            # Fallback to psutil if os.kill fails
            try:
                parent = psutil.Process(pid)
                parent.kill()
            except:
                pass
    
    @staticmethod
    def _worker_init():
        """Initialize worker process"""
        def signal_handler(signum, frame):
            os._exit(1)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        os.setpgrp()
    
    def _is_pool_healthy(self):
        """Check if the process pool is healthy"""
        if not self.process_pool:
            return False

        try:
            # Try to get pool state
            if hasattr(self.process_pool, '_state'):
                return self.process_pool._state == 'RUN'
            return True
        except:
            return False
    
    def execute_with_timeout(self, func: Callable, args: tuple = (), kwargs: dict = None,
                           timeout: float = None, task_id: str = "", func_name: str = "",
                           check_timeout: Callable = None) -> Any:
        """
        Execute function in isolated process with granular timeout handling.
        Uses unified timeout and retry configuration from rollout.yaml.
        """
        if self._shutdown:
            raise Exception("Process isolator is shut down")

        kwargs = kwargs or {}
        args = args or ()
        start_time = time.time()

        # Use unified configuration values
        timeout = timeout or self.operation_timeout
        max_execution_retries = self.max_retries
        execution_retry_count = 0

        # Ensure pool is started and healthy
        if not self._is_pool_healthy():
            print(f"⚠️ Pool not healthy for {func_name}, restarting...")
            with self._process_lock:
                if not self._shutdown:
                    if self.process_pool:
                        try:
                            self.process_pool.terminate()
                            self.process_pool.join()
                        except:
                            pass
                        self.process_pool = None
                    self._create_new_pool_with_retry()

        while execution_retry_count < max_execution_retries:
            try:
                # Double-check pool is still healthy before submission
                if not self._is_pool_healthy():
                    raise Exception("Pool not running")

                # Submit the task to process pool
                async_result = self.process_pool.apply_async(
                    self._execute_wrapper,
                    (func, args, kwargs, func_name, task_id)
                )

                try:
                    # Total timeout = wait_timeout (for queue) + operation_timeout (for execution)
                    # This allows tasks to wait longer in queue without burning through operation timeout
                    total_timeout = self.wait_timeout + timeout

                    # Poll with short intervals to allow shutdown checks
                    poll_interval = 1.0  # Check every 1 second
                    elapsed = 0
                    while elapsed < total_timeout:
                        # Check for shutdown/timeout via callback
                        if check_timeout:
                            try:
                                check_timeout()
                            except Exception as e:
                                # Shutdown or timeout requested - abort immediately
                                raise Exception(f"Operation aborted: {e}")

                        # Check if pool is shutting down
                        if self._shutdown:
                            raise Exception("Process isolator is shutting down")

                        try:
                            result = async_result.get(timeout=poll_interval)
                            return result
                        except multiprocessing.TimeoutError:
                            elapsed += poll_interval
                            continue

                    # If we get here, we've exceeded total_timeout
                    raise multiprocessing.TimeoutError()

                except multiprocessing.TimeoutError:
                    # Process timeout - just abandon the hung process
                    execution_time = time.time() - start_time
                    print(f"❌ Process timeout for {func_name} after {execution_time:.1f}s (max: {total_timeout:.0f}s = {self.wait_timeout:.0f}s wait + {timeout:.0f}s operation)")
                    print(f"🔄 Abandoning hung process, pool will replace worker automatically")

                    # The multiprocessing.Pool will detect the hung worker and replace it
                    # No need to recreate the entire pool
                    raise Exception(f"Process timeout after {total_timeout:.0f}s (wait_timeout={self.wait_timeout:.0f}s + operation_timeout={timeout:.0f}s)")

            except Exception as e:
                execution_time = time.time() - start_time
                error_str = str(e).lower()

                if 'timeout' in error_str:
                    # Timeout error - don't retry, just propagate (message already printed above)
                    raise
                elif 'pool not running' in error_str or 'pool' in error_str:
                    # Pool-related error - try to restart pool
                    print(f"⚠️ Pool error for {func_name}: {e}")
                    execution_retry_count += 1

                    if execution_retry_count < max_execution_retries:
                        print(f"🔄 Retrying {func_name} (attempt {execution_retry_count + 1}/{max_execution_retries})")
                        with self._process_lock:
                            if not self._shutdown:
                                if self.process_pool:
                                    try:
                                        self.process_pool.terminate()
                                        self.process_pool.join()
                                    except:
                                        pass
                                    self.process_pool = None
                                self._create_new_pool_with_retry()
                        time.sleep(1)  # Brief pause before retry
                        continue
                    else:
                        print(f"❌ Max execution retries reached for {func_name}")
                        raise Exception(f"Pool execution failed after {max_execution_retries} retries: {e}")
                else:
                    # Non-pool error - propagate immediately
                    print(f"❌ Process execution failed for {func_name} after {execution_time:.1f}s: {e}")
                    raise

        # Should never reach here due to the logic above, but just in case
        raise Exception(f"Unexpected error in retry loop for {func_name}")
    
    @staticmethod
    def _execute_wrapper(func, args, kwargs, func_name, task_id):
        """Wrapper to execute function in worker process"""
        worker_pid = os.getpid()
        try:
            def timeout_handler(signum, frame):
                os._exit(1)

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(900)  # 15-minute hard limit

            result = func(*args, **kwargs)

            signal.alarm(0)
            return result

        except Exception as e:
            print(f"❌ Worker {worker_pid} exception in {func_name}: {e}")
            traceback.print_exc()
            raise
        finally:
            try:
                signal.alarm(0)
            except:
                pass


class ProcessBasedHttpStack:
    """
    HTTP stack using process isolation with dedicated pool per operation type.
    Uses unified timeout and retry configuration from rollout.yaml.
    """

    def __init__(self, pool_config: dict, wait_timeout: float, timeout: float, max_retries: int = 2):
        """
        Initialize HTTP stack with separate pools for each operation type.

        Args:
            pool_config: Dictionary mapping operation names to pool sizes, e.g.:
                {
                    'navigate': 64,
                    'screenshot': 512,
                    'ac_tree': 256,
                    'metadata': 64,
                    'page_metadata': 256,
                    'execute': 256,
                    'allocate': 4,
                    'release': 4
                }
            wait_timeout: How long to wait in queue (seconds)
            timeout: How long for actual operation execution (seconds)
            max_retries: Maximum retry attempts
        """
        self.pool_config = pool_config
        self.wait_timeout = wait_timeout
        self.timeout = timeout
        self.max_retries = max_retries

        # Create a process isolator for each operation type (skip pools with size <= 0)
        self.isolators = {}
        skipped_pools = []
        for operation_name, pool_size in pool_config.items():
            if pool_size > 0:
                self.isolators[operation_name] = ProcessIsolator(
                    pool_size, wait_timeout, timeout, max_retries
                )
            else:
                skipped_pools.append(f"{operation_name}={pool_size}")

        self.stats = {
            'total': 0,
            'completed': 0,
            'failed': 0,
            'timed_out': 0,
            'pool_recreations': 0,
            'retried': 0
        }
        self._running = False

        print(f"🚀 ProcessBasedHttpStack initialized with dedicated pools:")
        for operation_name, pool_size in sorted(pool_config.items()):
            if pool_size > 0:
                print(f"   - {operation_name}: {pool_size} workers")
        if skipped_pools:
            print(f"⏭️  Skipped pools (size <= 0): {', '.join(skipped_pools)}")
        print(f"   - wait_timeout: {wait_timeout}s (queue wait)")
        print(f"   - operation_timeout: {timeout}s (execution)")
        print(f"   - max_retries: {max_retries}")
    
    def _get_pool_name(self, func_name: str) -> str:
        """
        Map function name to pool name for routing.

        Args:
            func_name: Name of the function being executed

        Returns:
            Pool name to use (key in pool_config)
        """
        func_lower = func_name.lower()

        # Map function names to pool names
        if 'allocate' in func_lower:
            return 'allocate'
        elif 'reset' in func_lower or 'release' in func_lower or 'cleanup' in func_lower:
            return 'release'
        elif 'navigate' in func_lower:
            return 'navigate'
        elif 'screenshot' in func_lower:
            return 'screenshot'
        elif 'ac_tree' in func_lower:
            return 'ac_tree'
        elif 'get_metadata' in func_lower and 'page' not in func_lower:
            return 'metadata'
        elif 'page_metadata' in func_lower:
            return 'page_metadata'
        else:
            # Default to execute pool for action execution and unknown operations
            return 'execute'

    def start(self):
        """Start all process pools in parallel for faster initialization"""
        if not self._running:
            import concurrent.futures
            import time

            total_workers = sum(self.pool_config.values())
            print(f"🚀 Starting {len(self.isolators)} process pools ({total_workers} total workers) in parallel...")

            # Show breakdown
            for op_name in sorted(self.pool_config.keys()):
                print(f"   - {op_name}: {self.pool_config[op_name]} workers")

            start_time = time.time()
            completed = 0

            # Start all pools concurrently using threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.isolators)) as executor:
                futures = {
                    executor.submit(isolator.start): operation_name
                    for operation_name, isolator in self.isolators.items()
                }

                # Wait for all to complete with progress updates
                for future in concurrent.futures.as_completed(futures):
                    operation_name = futures[future]
                    try:
                        future.result()
                        completed += 1
                        pool_size = self.pool_config[operation_name]
                        elapsed = time.time() - start_time
                        print(f"✓ Pool '{operation_name}' started ({pool_size} workers) - {completed}/{len(self.isolators)} pools done ({elapsed:.1f}s elapsed)")
                    except Exception as e:
                        print(f"❌ Failed to start pool '{operation_name}': {e}")
                        raise

            self._running = True
            total_time = time.time() - start_time
            print(f"✅ All {len(self.isolators)} pools started in {total_time:.1f}s ({total_workers} total workers)")

    def stop(self):
        """Stop all process pools"""
        if self._running:
            print("🛑 Stopping process-based HTTP stack...")
            for operation_name, isolator in self.isolators.items():
                print(f"   Stopping isolator: {operation_name}...")
                isolator.stop()
                print(f"   ✓ Isolator {operation_name} stopped")
            self._running = False
            print("✅ Process-based HTTP stack stopped")
    
    def execute(self, func: Callable, args: tuple = (), kwargs: dict = None,
               task_id: str = "", func_name: str = "",
               check_timeout: Callable = None, timeout: float = None) -> Any:
        """
        Execute HTTP operation in isolated process with retry logic.
        Routes to appropriate pool based on operation type.
        """
        if not self._running:
            raise RuntimeError("Process-based HTTP stack is not running")

        actual_timeout = timeout or self.timeout
        self.stats['total'] += 1

        # Use unified retry configuration
        max_retries = self.max_retries
        retry_count = 0

        # Route to appropriate pool based on function name
        pool_name = self._get_pool_name(func_name)
        isolator = self.isolators.get(pool_name, self.isolators.get('execute'))

        # Check if isolator exists (pool might be disabled with size=0)
        if isolator is None:
            raise ValueError(f"No process pool available for operation '{func_name}' (pool '{pool_name}' is disabled or doesn't exist)")

        # Special retry behavior for allocation operations
        is_allocation = pool_name == 'allocate'

        while True:  # Changed from retry_count <= max_retries
            try:
                # Check if we should abort before starting
                if check_timeout:
                    check_timeout()

                # Execute in isolated process using the appropriate pool
                result = isolator.execute_with_timeout(
                    func=func,
                    args=args or (),
                    kwargs=kwargs or {},
                    timeout=actual_timeout,
                    task_id=task_id,
                    func_name=func_name,
                    check_timeout=check_timeout
                )

                self.stats['completed'] += 1
                if retry_count > 0:
                    self.stats['retried'] += 1
                    if is_allocation:
                        print(f"✅ {func_name} succeeded after {retry_count} retries (allocation kept trying)")
                    else:
                        print(f"✅ {func_name} succeeded after {retry_count} retries")

                return result

            except Exception as e:
                error_str = str(e).lower()
                retry_count += 1

                # For allocation, always retry on retryable errors
                # For other operations, respect max_retries limit
                if is_allocation:
                    should_retry = (
                        'pool not running' in error_str or
                        'pool' in error_str or
                        'timeout' in error_str or
                        'failed to create' in error_str or
                        '503' in error_str or
                        'no available nodes' in error_str or
                        'capacity' in error_str
                    )
                else:
                    should_retry = (
                        retry_count <= max_retries and
                        ('pool not running' in error_str or
                         'pool' in error_str or
                         'timeout' in error_str or
                         'failed to create' in error_str or
                         '503' in error_str or
                         'no available nodes' in error_str or
                         'capacity' in error_str)
                    )

                if should_retry:
                    # Special handling for instance allocation - longer backoff
                    if is_allocation:
                        backoff_time = min(2 ** min(retry_count - 1, 6), 60)  # Cap exponential at 2^6, max 60s for allocation
                        if retry_count <= max_retries:
                            print(f"🔄 Retrying {func_name} in {backoff_time}s (attempt {retry_count}, will keep trying)")
                        else:
                            print(f"🔄 Retrying {func_name} in {backoff_time}s (attempt {retry_count}, allocation never gives up)")
                    else:
                        backoff_time = min(2 ** (retry_count - 1), 8)   # Exponential backoff, max 8s for other operations
                        print(f"🔄 Retrying {func_name} in {backoff_time}s (attempt {retry_count}/{max_retries + 1})")

                    time.sleep(backoff_time)
                    continue
                else:
                    # Final failure - update stats and raise (only for non-allocation operations)
                    if 'timeout' in error_str:
                        self.stats['timed_out'] += 1
                        if 'pool recreations' in error_str:
                            self.stats['pool_recreations'] += 1
                    else:
                        self.stats['failed'] += 1

                    if retry_count > 1:
                        print(f"❌ {func_name} failed after {retry_count - 1} retries: {e}")

                    raise
        
        # Should never reach here
        raise Exception(f"Unexpected error in retry loop for {func_name}")
    
    def get_stats(self) -> dict:
        """Get execution statistics"""
        total = max(self.stats['total'], 1)
        total_workers = sum(self.pool_config.values())
        return {
            **self.stats,
            'success_rate': (self.stats['completed'] / total) * 100,
            'timeout_rate': (self.stats['timed_out'] / total) * 100,
            'failure_rate': (self.stats['failed'] / total) * 100,
            'retry_rate': (self.stats['retried'] / total) * 100,
            'pool_recreation_rate': (self.stats['pool_recreations'] / total) * 100,
            'total_workers': total_workers,
            'pool_config': self.pool_config
        }


class WatchdogMonitor:
    """
    Watchdog process that monitors the main execution and kills it if it hangs.
    """

    def __init__(self, max_execution_time: int = 10800):
        self.max_execution_time = max_execution_time
        self.heartbeat_file = "/tmp/webgym_heartbeat"
        self.watchdog_process = None
        self._shutdown = False
    
    def start(self):
        """Start the watchdog process"""
        if self.watchdog_process is None:
            ctx = multiprocessing.get_context('spawn')
            self.watchdog_process = ctx.Process(
                target=self._watchdog_loop,
                args=(self.heartbeat_file, self.max_execution_time),
                daemon=True
            )
            self.watchdog_process.start()
            self.update_heartbeat()
            print(f"🐕 Watchdog started (PID: {self.watchdog_process.pid})")
    
    def stop(self):
        """Stop the watchdog process"""
        self._shutdown = True
        if self.watchdog_process:
            try:
                self.watchdog_process.terminate()
                self.watchdog_process.join(timeout=5)
                if self.watchdog_process.is_alive():
                    self.watchdog_process.kill()
            except:
                pass
            finally:
                self.watchdog_process = None
        
        try:
            os.unlink(self.heartbeat_file)
        except:
            pass
        
        print("🐕 Watchdog stopped")
    
    def update_heartbeat(self):
        """Update heartbeat timestamp"""
        if not self._shutdown:
            try:
                with open(self.heartbeat_file, 'w') as f:
                    f.write(str(time.time()))
            except:
                pass
    
    @staticmethod
    def _watchdog_loop(heartbeat_file: str, max_time: int):
        """Watchdog loop that monitors heartbeat"""
        print(f"🐕 Watchdog monitoring started (max time: {max_time}s)")
        
        while True:
            try:
                time.sleep(30)
                
                if not os.path.exists(heartbeat_file):
                    continue
                
                try:
                    with open(heartbeat_file, 'r') as f:
                        last_heartbeat = float(f.read().strip())
                except:
                    continue
                
                time_since_heartbeat = time.time() - last_heartbeat
                if time_since_heartbeat > max_time:
                    print(f"💀 WATCHDOG: No heartbeat for {time_since_heartbeat:.1f}s, killing main process")
                    parent_pid = os.getppid()
                    os.kill(parent_pid, signal.SIGKILL)
                    break
                    
            except Exception as e:
                print(f"🐕 Watchdog error: {e}")
                time.sleep(5)