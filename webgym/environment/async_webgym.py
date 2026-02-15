import os
import time
import hashlib
import json
import traceback
import fcntl
from typing import List, Dict, Optional, Tuple, Callable
from datetime import datetime
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import signal
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from webgym.data.components import Task, Observation, Action, Response, Reward
from webgym.data.response_decomposer import decompose_raw_response, get_action_string
from webgym.environment import client
from webgym.environment.task_monitor import TaskMonitor
from webgym.navigation_error_logger import log_navigation_error, set_navigation_error_logger

# Import fixed checkpoint utilities
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts'))
from checkpoint_utils import handle_grace_period_expiry_fixed, GracePeriodExpiredException


def _truncate_error_msg(error_msg: str, max_length: int = 150) -> str:
    """Truncate error messages to keep logs clean and readable"""
    return error_msg[:max_length] + "..." if len(error_msg) > max_length else error_msg

class AsyncWebGym:
    """
    Enhanced WebGym with HTTP stack and fine-grained task monitoring.
    Silent mode - only shows errors and critical messages.
    """
    
    def __init__(self,
                master_port: int,
                host_ip: str,
                cpu_cluster_token: str,
                sampled_tasks: List[Dict],
                save_path: str,
                num_workers: int,
                verbose: bool,
                retry_policy: Dict,
                task_timeout_minutes: int = 20,
                completion_threshold: float = 0.98,
                completion_grace_period: int = 120,
                blocklist_manager = None,
                skip_instance_cleanup: bool = False,
                multinode_rank_suffix: str = '',
                split: str = 'train',
                env_config = None,
                interaction_mode: str = 'coordinates'):
        """
        Initialize AsyncWebGym with process-based HTTP handling
        """
        self.master_host = host_ip
        self.master_port = master_port
        self.api_key = cpu_cluster_token
        self.save_path = save_path

        # Configure navigation error logger with correct path
        # Use environment variable so child processes can access it
        nav_error_log_path = os.path.join(save_path, "navigation_error_websites.txt")
        os.environ['WEBGYM_NAV_ERROR_LOG_PATH'] = nav_error_log_path
        set_navigation_error_logger(nav_error_log_path)

        self.server_size = num_workers
        self.verbose = verbose
        self.task_timeout_minutes = task_timeout_minutes
        self.completion_threshold = completion_threshold
        self.completion_grace_period = completion_grace_period
        self.blocklist_manager = blocklist_manager
        self.skip_instance_cleanup = skip_instance_cleanup
        self.multinode_rank_suffix = multinode_rank_suffix
        self.split = split  # Store split to use for trajectory file naming
        self.env_config = env_config  # Store env_config for per-task max_steps
        self.interaction_mode = interaction_mode  # Store interaction mode for AC tree optimization

        # Shutdown flag for all components
        self._shutdown_requested = threading.Event()

        # Import and use process-based components
        from .process_isolator import ProcessBasedHttpStack, WatchdogMonitor

        # Extract config properly - use actual config values
        wait_timeout = retry_policy.get('wait_timeout', 3600.0)
        timeout = retry_policy.get('operation_timeout', 240.0)
        self.max_retries = retry_policy.get('max_retries', 2)
        pool_config = retry_policy.get('http_pools', {
            # Default pool configuration if not specified
            'navigate': 64,
            'screenshot': 256,
            'ac_tree': 128,
            'metadata': 64,
            'page_metadata': 128,
            'execute': 128,
            'allocate': 4,
            'release': 4
        })
        self.max_vllm_sessions = retry_policy.get('max_vllm_sessions', 32)

        self.http_stack = ProcessBasedHttpStack(
            pool_config=pool_config,  # Dictionary of pool sizes per operation type
            wait_timeout=wait_timeout,  # Timeout for waiting in queue (3600s = 1 hour)
            timeout=timeout,  # Timeout for operation execution
            max_retries=self.max_retries  # Use unified retry config from rollout.yaml
        )

        total_workers = sum(pool_config.values())
        print(f"🔧 WebGym initialized with {total_workers} total HTTP workers across {len(pool_config)} pools, {self.max_vllm_sessions} vLLM sessions, {self.server_size} task workers, {timeout}s timeout, {self.max_retries} max_retries")
        print(f"🎯 Completion policy: {self.completion_threshold*100:.0f}% threshold with {self.completion_grace_period}s grace period")
        print(f"🎨 Interaction mode: {self.interaction_mode} (AC tree fetch: {'disabled' if self.interaction_mode == 'coordinates' else 'enabled'})")

        # Add watchdog monitor (no hard time limit - rely on completion threshold instead)
        self.watchdog = WatchdogMonitor(max_execution_time=86400)  # 24 hours safety limit
        
        # Simplified monitoring
        self.system_monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.system_monitor_thread.start()

        # Use regular ThreadPoolExecutor for task coordination (not HTTP)
        self.task_executor = None

        # Async evaluation infrastructure
        self.evaluation_executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="eval")
        self.pending_evaluations = {}  # task_id -> Future mapping
        self.evaluation_lock = threading.Lock()

        # Async screenshot comparison infrastructure
        self.screenshot_comparison_executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="screenshot_cmp")
        self.pending_screenshot_comparisons = {}  # task_id -> Future mapping
        self.screenshot_comparison_lock = threading.Lock()

        # Master client
        self.master_client = client.MasterClient(
            host=self.master_host,
            port=self.master_port,
            api_key=self.api_key
        )
        
        # Prepare tasks
        self.tasks = self._prepare_tasks(sampled_tasks)
        self.total_tasks = len(self.tasks)

        # Results storage
        self.trajectories = [None] * self.total_tasks

        # Get maximum possible steps from difficulty-based config for TaskMonitor
        if self.env_config is not None:
            config_key = f"{self.split}_difficulty_max_steps"
            if hasattr(self.env_config, config_key):
                max_steps_config = getattr(self.env_config, config_key)
                # Check if it's dict-like (works for both regular dict and OmegaConf DictConfig)
                if hasattr(max_steps_config, 'keys') and hasattr(max_steps_config, 'values'):
                    # For dict config, use the maximum value (hard difficulty)
                    monitor_max_steps = max(max_steps_config.values())
                elif hasattr(max_steps_config, '__len__') and hasattr(max_steps_config, '__getitem__'):
                    # For list config, use the maximum value (works for list, tuple, and OmegaConf ListConfig)
                    monitor_max_steps = max(max_steps_config) if len(max_steps_config) > 0 else 100
                else:
                    monitor_max_steps = 100  # Fallback default
            else:
                monitor_max_steps = 100  # Fallback default
        else:
            monitor_max_steps = 100  # Fallback default

        # Enhanced Task Monitor with fine-grained status
        self.monitor = TaskMonitor(
            total_tasks=self.total_tasks,
            max_steps=monitor_max_steps,
            enable_web_dashboard=True,
            web_port=5000
        )
        
        # System stats
        self.system_stats = {
            'start_time': time.time(),
            'tasks_started': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'instances_allocated': 0,
            'instances_released': 0,
            'http_requests_total': 0,
            'avg_task_duration': 0.0,
            'operations': {
                'navigations': 0,
                'screenshots': 0,
                'ac_trees': 0,
                'vllm_calls': 0,
                'actions_executed': 0,
                'rewards_computed': 0
            }
        }
        self.stats_lock = threading.Lock()

    def _monitor_system(self):
        """System monitoring - only shows warnings, respects shutdown"""
        while not self._shutdown_requested.is_set():
            # Use wait with timeout instead of sleep for responsive shutdown
            if self._shutdown_requested.wait(timeout=15):
                break  # Shutdown requested
                
            try:
                # Get HTTP stack stats
                http_stats = self.http_stack.get_stats()
                
                # Get task progress
                progress = self.monitor.get_progress_summary() if hasattr(self.monitor, 'get_progress_summary') else {}
                
                # Alert conditions only
                if http_stats['stack_size'] > 100:
                    print(f"⚠️  Large HTTP stack: {http_stats['stack_size']} requests waiting")
                
                if http_stats['success_rate'] < 80 and http_stats['completed'] > 20:
                    print(f"⚠️  Low HTTP success rate: {http_stats['success_rate']:.1f}%")
                
                # Performance warnings
                if progress.get('action', 0) > 20:
                    print(f"⚠️  Many tasks waiting for vLLM: {progress['action']} (possible bottleneck)")
                    
            except Exception as e:
                if not self._shutdown_requested.is_set() and self.verbose:
                    print(f"❌ Monitor error: {e}")

    def _prepare_tasks(self, sampled_tasks) -> List[Dict]:
        """
        Convert sampled tasks to internal format.

        NOTE: URL filtering/correction already happened at sampling time for train split.
        We just normalize URLs here, no skipping.
        """
        tasks = []

        for i in range(len(sampled_tasks)):
            task_data = sampled_tasks.iloc[i]

            # Create unique output directory
            timestamp = int(time.time())
            task_hash = hashlib.sha256(f"{timestamp}_{i}_{task_data['task_name']}".encode()).hexdigest()[:12]
            output_dir = os.path.join(self.save_path, "images", f"task_{len(tasks):04d}_{task_hash}")

            # Just normalize website URL (no filtering/correction at runtime)
            website = task_data['website']
            if not website.startswith('http'):
                website = f"https://{website}"

            # Ensure evaluator_reference is a list of dicts with 'description' and 'difficulty'
            evaluator_reference = task_data['evaluator_reference']
            if isinstance(evaluator_reference, str):
                # Legacy format: single string -> convert to list of dicts
                evaluator_reference = [{"description": evaluator_reference, "difficulty": 1}]
            elif isinstance(evaluator_reference, list):
                # Check if it's already in the new format (list of dicts)
                if evaluator_reference and isinstance(evaluator_reference[0], dict):
                    # Check if it's the fact-group format (with 'facts' field)
                    if 'facts' in evaluator_reference[0]:
                        # Flatten fact groups into individual rubric items with group context
                        flattened = []
                        for group in evaluator_reference:
                            group_desc = group.get('description', '')
                            group_id = group.get('id', '')
                            for fact in group.get('facts', []):
                                # Include group context for the evaluator
                                fact_with_context = f"[Group {group_id}: {group_desc}] {fact}"
                                flattened.append({"description": fact_with_context, "difficulty": 1})
                        evaluator_reference = flattened
                    # Otherwise already in correct format, keep as is
                else:
                    # Legacy format: list of strings -> convert to list of dicts
                    evaluator_reference = [{"description": item, "difficulty": 1} for item in evaluator_reference]
            else:
                # Other types -> convert to list of dicts
                evaluator_reference = [{"description": str(evaluator_reference), "difficulty": 1}]

            # Compute max_steps based on difficulty level
            max_steps = self._get_max_steps_for_difficulty(task_data['difficulty'])

            task = Task(
                task_name=task_data['task_name'],
                domain=task_data['domain'],
                subdomain=task_data['subdomain'],
                website=website,  # Use corrected website
                difficulty=task_data['difficulty'],
                evaluator_reference=evaluator_reference,
                reference_answer=task_data['definite_answer'],
                attempt_level=task_data.get('attempt_level', 0),  # Get from DataFrame, default 0
                task_id=task_data.get('task_id'),  # Sequential task ID from task file
                max_steps=max_steps,  # Per-task max steps based on difficulty
                trajectory_index=task_data.get('_trajectory_index', len(tasks))  # Use preserved index if available (resume mode), else position
            )

            tasks.append({
                'index': len(tasks),  # Use filtered index
                'task': task,
                'task_id': f"task_{len(tasks):04d}",  # Use filtered index
                'output_dir': output_dir
            })

        return tasks

    def _get_max_steps_for_difficulty(self, difficulty: int) -> int:
        """
        Get max_steps for a task based on its difficulty level using easy/medium/hard categorization.

        Args:
            difficulty: Task difficulty level (any positive integer)

        Returns:
            Max steps for this difficulty level
        """
        if self.env_config is None:
            raise ValueError("env_config is required for difficulty-based max_steps")

        # Get the appropriate config: train_difficulty_max_steps or test_difficulty_max_steps
        config_key = f"{self.split}_difficulty_max_steps"

        if not hasattr(self.env_config, config_key):
            raise ValueError(f"Config key '{config_key}' not found in env_config")

        max_steps_config = getattr(self.env_config, config_key)

        # Convert to Python int (in case it's numpy.int64 from pandas)
        difficulty_int = int(difficulty)

        # Map difficulty to easy/medium/hard category
        # easy: 1-3, medium: 4-6, hard: 7+
        if difficulty_int <= 3:
            category = 'easy'
        elif difficulty_int <= 6:
            category = 'medium'
        else:
            category = 'hard'

        # Support both dict-based and list-based configs for backward compatibility
        # Check if it's dict-like (works for both regular dict and OmegaConf DictConfig)
        if hasattr(max_steps_config, 'keys') and hasattr(max_steps_config, '__getitem__'):
            if category not in max_steps_config:
                raise ValueError(f"Category '{category}' not found in max_steps config")
            return max_steps_config[category]
        elif isinstance(max_steps_config, (list, tuple)) or (hasattr(max_steps_config, '__len__') and hasattr(max_steps_config, '__getitem__') and not hasattr(max_steps_config, 'keys')):
            # Legacy support: map to old list-based system (works for list, tuple, and OmegaConf ListConfig)
            if difficulty_int < 1 or difficulty_int > len(max_steps_config):
                # Fallback for out-of-range difficulties
                return max_steps_config[-1] if len(max_steps_config) > 0 else 30
            return max_steps_config[difficulty_int - 1]
        else:
            raise ValueError(f"Invalid max_steps config type: {type(max_steps_config)}")

    def _release_all_instances_sync(self):
        """Release ALL instances with controlled parallelism and proper error handling"""
        try:
            # Get server info
            print(f"Getting server info...")
            info = self.master_client.get_info()

            # Collect all instances to release with defensive handling
            instances = []
            num_instances_to_release = 0

            # Handle different possible server response formats
            if 'nodes' in info:
                for node in info.get('nodes', []):
                    node_url = node.get('url', '') if isinstance(node, dict) else ''

                    # Get instances from node
                    node_instances = node.get('instances', []) if isinstance(node, dict) else (node if isinstance(node, list) else [])

                    for inst in node_instances:
                        # Create proper instance dict for reset_instance
                        if isinstance(inst, str):
                            instance_dict = {"instance_id": inst}
                            if node_url:
                                instance_dict["node"] = node_url
                        elif isinstance(inst, dict):
                            instance_dict = inst.copy()
                            if 'node' not in instance_dict and node_url:
                                instance_dict['node'] = node_url
                        else:
                            # Fallback for unexpected format
                            instance_dict = {"instance_id": str(inst)}
                            if node_url:
                                instance_dict["node"] = node_url

                        instances.append(instance_dict)
                        num_instances_to_release += 1

            # Fallback for simple server format
            elif 'in_use' in info:
                for inst in info.get('in_use', []):
                    instance_id = inst if isinstance(inst, str) else str(inst)
                    instance_dict = {"instance_id": instance_id, "node": ""}
                    instances.append(instance_dict)
                    num_instances_to_release += 1

            print(f"Number of instances to release: {num_instances_to_release}")

            if len(instances) == 0:
                print("No instances to release")
                # MULTINODE: Still create instances_released flag even when no instances to release
                if self.multinode_rank_suffix and self.multinode_rank_suffix.startswith('_rank_0'):
                    import os
                    rank_num = self.multinode_rank_suffix.replace('_rank_', '')
                    flag_file = os.path.join(self.save_path, f'multinode_flags/instances_released_rank_{rank_num}')
                    os.makedirs(os.path.dirname(flag_file), exist_ok=True)
                    with open(flag_file, 'w') as f:
                        f.write('')
                    print(f"🚩 Master node: Created instances_released flag (no instances to release): {flag_file}")
                return

            # **CONTROLLED PARALLELISM** - prevent server overload
            max_workers = min(20, len(instances))  # Max 20 concurrent, not unlimited!
            print(f"Using {max_workers} workers for controlled parallel cleanup")

            successful = 0
            already_released = 0  # Track UUID errors (actually successes)
            failed = 0

            def reset_single_instance_safe(inst):
                """Safe instance reset with proper error classification"""
                try:
                    result = self.master_client.reset_instance(inst)
                    return "success", inst.get('instance_id', 'unknown'), None
                except Exception as e:
                    error_str = str(e).lower()
                    instance_id = inst.get('instance_id', 'unknown')

                    # Classify UUID/not-found errors as successes (already released)
                    if any(pattern in error_str for pattern in [
                        'invalid instance uuid',
                        'invalid uuid',
                        'instance not found',
                        'does not exist',
                        'not found'
                    ]):
                        return "already_released", instance_id, str(e)
                    else:
                        return "failed", instance_id, str(e)

            # Progress tracking
            start_time = time.time()
            print(f"Starting controlled parallel cleanup of {len(instances)} instances...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_instance = {
                    executor.submit(reset_single_instance_safe, inst): inst
                    for inst in instances
                }

                # Process results as they complete
                completed = 0
                for future in as_completed(future_to_instance):
                    try:
                        result_type, instance_id, error = future.result()

                        if result_type == "success":
                            successful += 1
                        elif result_type == "already_released":
                            already_released += 1
                        else:  # failed
                            failed += 1
                            if failed <= 5:  # Only show first 5 real failures
                                print(f"  ⚠️  Real failure: {instance_id[:8] if instance_id else 'unknown'}... - {error[:80] if error else 'unknown error'}")

                        completed += 1

                        # Progress update every 100 instances or at completion
                        if completed % 100 == 0 or completed == len(instances):
                            elapsed = time.time() - start_time
                            rate = completed / elapsed if elapsed > 0 else 0
                            print(f"  Progress: {completed}/{len(instances)} ({100*completed/len(instances):.1f}%) "
                                  f"[{elapsed:.0f}s, {rate:.1f}/s] ✓={successful}, ~={already_released}, ✗={failed}")

                    except Exception as e:
                        failed += 1
                        print(f"  ⚠️  Future result error: {e}")

            # Final summary
            total_effective_success = successful + already_released
            elapsed = time.time() - start_time

            print(f"✅ Controlled parallel cleanup complete ({elapsed:.1f}s total):")
            print(f"  - Successfully released: {successful}")
            print(f"  - Already released (UUID errors): {already_released}")
            print(f"  - Real failures: {failed}")
            print(f"  - Total effective success: {total_effective_success}/{len(instances)} ({100*total_effective_success/len(instances):.1f}%)")
            print(f"  - Average rate: {len(instances)/elapsed:.1f} instances/second")

            if failed > 0:
                print(f"⚠️  {failed} instances had real failures (may need manual cleanup)")
            else:
                print("🎉 All instances processed successfully (including already-released ones)")

        except Exception as e:
            print(f"⚠️  Critical error during controlled cleanup: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()

        # MULTINODE: Create instances_released flag for master node after instance release completes
        if self.multinode_rank_suffix and self.multinode_rank_suffix.startswith('_rank_0'):
            import os
            rank_num = self.multinode_rank_suffix.replace('_rank_', '')
            flag_file = os.path.join(self.save_path, f'multinode_flags/instances_released_rank_{rank_num}')
            os.makedirs(os.path.dirname(flag_file), exist_ok=True)
            with open(flag_file, 'w') as f:
                f.write('')
            print(f"🚩 Master node: Created instances_released flag: {flag_file}")

    def _release_single_instance_direct(self, instance: Dict, task_id: str):
        """
        Release a single instance directly without process isolation to avoid cascade failures.
        Uses the same error classification as the bulk cleanup.
        """
        try:
            result = self.master_client.reset_instance(instance)
            return True
        except Exception as e:
            error_str = str(e).lower()

            # Classify UUID/not-found errors as successes (already released)
            if any(pattern in error_str for pattern in [
                'invalid instance uuid',
                'invalid uuid',
                'instance not found',
                'does not exist',
                'not found'
            ]):
                # This is actually a success - instance was already cleaned up
                return True
            else:
                # This is a real failure
                raise e

    @staticmethod
    def _add_screenshot_comparison_to_trajectory(trajectory: List[Dict]) -> List[Dict]:
        """
        Add 'same_as_next_screenshot' field to each step in a single trajectory.
        This field is True if the step's screenshot is pixel-identical to the next step's screenshot.
        Last step always has False (no next step to compare).

        Args:
            trajectory: Single trajectory (list of step dicts)

        Returns:
            Same trajectory with same_as_next_screenshot field added to each step
        """
        from PIL import Image
        import numpy as np

        if not trajectory:
            return trajectory

        for j, step in enumerate(trajectory):
            # Last step always has False (no next step)
            if j == len(trajectory) - 1:
                step['same_as_next_screenshot'] = False
                continue

            # Compare current and next step screenshots
            current_obs = step.get('observation')
            next_obs = trajectory[j + 1].get('observation')

            current_image_path = getattr(current_obs, 'image_path', None) if current_obs else None
            next_image_path = getattr(next_obs, 'image_path', None) if next_obs else None

            same_screenshot = False
            if current_image_path and next_image_path:
                # Fast path: same file path
                if current_image_path == next_image_path:
                    same_screenshot = True
                else:
                    # Load and compare pixel values
                    try:
                        current_img = np.array(Image.open(current_image_path))
                        next_img = np.array(Image.open(next_image_path))
                        same_screenshot = np.array_equal(current_img, next_img)
                    except Exception:
                        # If comparison fails, assume different
                        same_screenshot = False

            step['same_as_next_screenshot'] = same_screenshot

        return trajectory

    def run_automation_with_fairness(self, agent, progress_callback=None, checkpoint_callback=None, checkpoint_interval=0.25, checkpoint_state=None, traj_metadata=None) -> List[List[Dict]]:
        """
        Run automation with process-based isolation and watchdog protection

        Args:
            agent: Agent to use for task execution
            progress_callback: Optional callback for progress updates
            checkpoint_callback: Optional callback(completed_count, total_tasks, trajectories)
                               called at checkpoint percentages
            checkpoint_interval: Interval for checkpoints (e.g., 0.25 = 25%)
            checkpoint_state: Optional dict to track checkpoint state (for duplicate prevention)
            traj_metadata: Optional dict with config metadata (solution_model, judge_model, etc.)
                          to save alongside trajectories
        """
        print("🚀 Starting process-based automation with watchdog protection")

        # Start watchdog first
        self.watchdog.start()

        try:
            # Start process-based HTTP stack
            self.http_stack.start()

            # Cleanup instances (only on master node in multinode setup)
            if not self.skip_instance_cleanup:
                self._release_all_instances_sync()
            else:
                print("⏭️  Skipping instance cleanup (worker node in multinode setup)")

            # Start monitor
            self.monitor.start_monitoring()

            # Use simpler ThreadPoolExecutor for task coordination (not HTTP requests)
            task_executor = ThreadPoolExecutor(max_workers=self.server_size)  # Use actual config (256, not 50!)
            try:
                # Submit all tasks
                future_to_task = {}

                for task_info in self.tasks:
                    future = task_executor.submit(
                        self._process_single_task_with_process_isolation,
                        task_info, agent
                    )
                    future_to_task[future] = task_info

                print(f"📤 Submitted {len(future_to_task)} tasks to process-based executor")

                # Collect results with completion threshold monitoring
                completed_count = 0
                start_time = time.time()
                last_progress_time = start_time
                completion_threshold_reached_at = None  # Track when we hit 98%
                grace_period_expired = False  # Track if grace period expired

                # Track checkpoint milestones (will be populated if checkpoint_callback is provided)
                checkpoint_thresholds = []
                reached_checkpoints = set()

                # Initialize checkpoint thresholds if callback is provided
                # CRITICAL: Exclude 1.0 (100%) to avoid duplicate with final save
                if checkpoint_callback:
                    num_checkpoints = int(1.0 / checkpoint_interval)
                    checkpoint_thresholds = [checkpoint_interval * (i + 1) for i in range(num_checkpoints - 1)]  # Exclude final
                    print(f"📊 Checkpoints at: {[f'{int(t*100)}%' for t in checkpoint_thresholds]} (100% will be final save)")

                # Use a timeout on as_completed so we can check grace period regularly
                remaining_futures = set(future_to_task.keys())
                while remaining_futures:
                    # Check grace period timeout first (even if no tasks complete)
                    if completion_threshold_reached_at is not None:
                        grace_elapsed = time.time() - completion_threshold_reached_at
                        if grace_elapsed >= self.completion_grace_period:
                            print(f"⏰ Grace period expired ({grace_elapsed:.0f}s)")
                            grace_period_expired = True
                            break

                    # Wait for next task with 5 second timeout to allow grace period checks
                    try:
                        future = next(as_completed(remaining_futures, timeout=5))
                        remaining_futures.remove(future)
                    except StopIteration:
                        # No more futures
                        break
                    except:
                        # Timeout - no task completed in 5s, loop back to check grace period
                        continue

                    # Update watchdog heartbeat
                    self.watchdog.update_heartbeat()

                    # Check for shutdown
                    if self._shutdown_requested.is_set():
                        print("🛑 Shutdown requested, stopping task collection")
                        break

                    # Get result
                    task_info = future_to_task[future]
                    try:
                        trajectory = future.result(timeout=60)  # 1 minute to get result
                        # If trajectory is None or empty, create error trajectory
                        if not trajectory or len(trajectory) == 0:
                            task_id = task_info['task_id']
                            print(f"⚠️ {task_id}: Task returned empty trajectory, creating error trajectory")
                            trajectory = self._create_dummy_trajectory_with_error(
                                task_info['task'], task_info['output_dir'], "EMPTY_RESULT",
                                "Task processing completed but returned no trajectory"
                            )
                        self.trajectories[task_info['index']] = trajectory
                    except Exception as e:
                        error_msg = str(e)
                        task_id = task_info['task_id']
                        print(f"❌ Failed to get result for {task_id}: {_truncate_error_msg(str(e))}")

                        # Enhanced error categorization for thread result failures
                        if any(keyword in error_msg.lower() for keyword in ['process timeout', 'killed', 'process was killed']):
                            error_type = "PROCESS_KILLED"
                            detailed_error = f"[{error_type}] Task thread result retrieval was killed: {error_msg}"
                        elif "timeout" in error_msg.lower():
                            error_type = "TIMEOUT"
                            detailed_error = f"[{error_type}] Task thread result retrieval timed out: {error_msg}"
                        elif "shutdown requested" in error_msg.lower():
                            error_type = "SHUTDOWN"
                            detailed_error = f"[{error_type}] Task thread aborted due to shutdown: {error_msg}"
                        else:
                            error_type = "ERROR"
                            detailed_error = f"[{error_type}] Task thread failed: {error_msg}"

                        # Create dummy trajectory with error logging
                        self.trajectories[task_info['index']] = self._create_dummy_trajectory_with_error(
                            task_info['task'], task_info['output_dir'], error_type, detailed_error
                        )

                    completed_count += 1
                    last_progress_time = time.time()

                    # Check for checkpoint saves (if callback provided)
                    if checkpoint_callback and checkpoint_thresholds:
                        completion_rate = completed_count / self.total_tasks
                        for threshold in checkpoint_thresholds:
                            if completion_rate >= threshold and threshold not in reached_checkpoints:
                                reached_checkpoints.add(threshold)
                                print(f"✅ Checkpoint reached: {int(threshold*100)}% ({completed_count}/{self.total_tasks})")
                                try:
                                    # Note: reward=-1 placeholders are filtered out by checkpoint_save in checkpoint_utils.py
                                    # They will be saved at the next checkpoint once evaluation completes
                                    checkpoint_callback(completed_count, self.total_tasks, self.trajectories)
                                except Exception as e:
                                    print(f"⚠️ Checkpoint save failed at {int(threshold*100)}%: {e}")

                    # Check for 98% completion threshold
                    completion_rate = completed_count / self.total_tasks
                    if completion_rate >= self.completion_threshold and completion_threshold_reached_at is None:
                        completion_threshold_reached_at = time.time()
                        print(f"✅ Completion threshold reached: {completed_count}/{self.total_tasks} ({completion_rate*100:.1f}%)")
                        print(f"⏳ Grace period active: waiting {self.completion_grace_period}s before cancelling remaining tasks")

                    # Grace period check is now at the top of the loop (lines 503-509)
                    # This ensures it's checked every 5 seconds even if no tasks complete

                    # Progress update
                    if completed_count % 50 == 0:
                        print(f"📊 Progress: {completed_count}/{self.total_tasks} tasks completed")
                    # Call progress callback for wandb logging every 100 tasks
                    if progress_callback and completed_count % 100 == 0:
                        progress_callback(completed_count, self.total_tasks)

                    # Update watchdog
                    self.watchdog.update_heartbeat()

                # Handle grace period expiry using fixed checkpoint logic
                # Pass split and rank_suffix so grace period handler can use incremental format
                import torch
                # Use the split passed during initialization (train or test)
                rank_suffix = self.multinode_rank_suffix if hasattr(self, 'multinode_rank_suffix') else ''

                # Wait for all pending evaluations and screenshot comparisons to complete before saving
                if grace_period_expired:
                    print("⏳ Grace period expired - waiting for pending evaluations and screenshot comparisons to complete...")
                    self._wait_for_pending_evaluations()
                    self._wait_for_pending_screenshot_comparisons()
                    print("✅ All evaluations and screenshot comparisons complete - proceeding with save")

                handle_grace_period_expiry_fixed(
                    grace_period_expired=grace_period_expired,
                    completed_count=completed_count,
                    total_tasks=self.total_tasks,
                    future_to_task=future_to_task,
                    trajectories=self.trajectories,
                    tasks=self.tasks,
                    checkpoint_state=checkpoint_state,
                    trajectory_file=None,  # Not used - using incremental format instead
                    _create_dummy_trajectory_with_error=self._create_dummy_trajectory_with_error,
                    save_path=self.save_path,  # Pass for incremental saving
                    split=self.split,  # Pass split (train/test)
                    rank_suffix=rank_suffix,  # Pass rank suffix for multinode
                    metadata=traj_metadata  # Pass config metadata for trajectory files
                )

                # Handle remaining tasks - use task indices instead of dict objects
                all_task_indices = {task_info['index'] for task_info in future_to_task.values()}
                completed_task_indices = {
                    future_to_task[f]['index'] for f in future_to_task
                    if f.done()
                }
                remaining_task_indices = all_task_indices - completed_task_indices

                if remaining_task_indices:
                    print(f"⚠️ {len(remaining_task_indices)} tasks did not complete, creating dummy trajectories")
                    # Create dummy trajectories for remaining tasks with error logging
                    for task_index in remaining_task_indices:
                        if self.trajectories[task_index] is None:
                            # Find the corresponding task_info for this index
                            task_info = self.tasks[task_index]
                            task_id = task_info['task_id']

                            error_type = "INCOMPLETE"
                            detailed_error = f"[{error_type}] Task did not complete execution, likely due to timeout or system shutdown"

                            self.trajectories[task_index] = self._create_dummy_trajectory_with_error(
                                task_info['task'], task_info['output_dir'], error_type, detailed_error
                            )

                # Wait for all pending evaluations and screenshot comparisons to complete
                self._wait_for_pending_evaluations()
                self._wait_for_pending_screenshot_comparisons()

                # Final stats
                valid_trajectories = len([t for t in self.trajectories if t])
                print(f"✅ Automation complete: {valid_trajectories}/{self.total_tasks} valid trajectories")
                print(f"   (Screenshot comparison metadata was added asynchronously during task processing)")

                return self.trajectories

            finally:
                # Shutdown executor WITHOUT waiting for tasks to complete
                print("🔪 Shutting down task executor without waiting...")
                task_executor.shutdown(wait=False)
                print("✅ Task executor shutdown complete")

        except GracePeriodExpiredException:
            # Grace period expired - trajectories already saved, re-raise for proper cleanup
            print("✅ Grace period handling complete - propagating exception for cleanup")
            raise

        except Exception as e:
            print(f"💥 Critical error in process-based automation: {e}")
            import traceback
            traceback.print_exc()

            # Wait for pending evaluations and screenshot comparisons even on error
            self._wait_for_pending_evaluations()
            self._wait_for_pending_screenshot_comparisons()

            # Ensure we return something
            # Note: Screenshot metadata is added asynchronously
            if self.trajectories:
                return self.trajectories
            else:
                return [[] for _ in range(self.total_tasks)]

        finally:
            print("🧹 Starting process-based cleanup...")

            # Signal shutdown to all monitoring threads
            print("🚦 Setting shutdown flag...")
            self._shutdown_requested.set()
            print("✓ Shutdown flag set")

            # Wait for system monitor thread to finish
            if self.system_monitor_thread and self.system_monitor_thread.is_alive():
                print("⏳ Waiting for system monitor thread to finish...")
                self.system_monitor_thread.join(timeout=20)  # Wait up to 20 seconds (longer than the 15s wait in monitor)
                if self.system_monitor_thread.is_alive():
                    print("⚠️ System monitor thread did not stop cleanly")
                else:
                    print("✓ System monitor thread stopped")

            # Stop watchdog
            try:
                self.watchdog.stop()
            except Exception as e:
                print(f"⚠️ Error stopping watchdog: {e}")

            # Shutdown evaluation executor
            try:
                print("🔪 Shutting down evaluation executor...")
                self.evaluation_executor.shutdown(wait=True, cancel_futures=False)
                print("✓ Evaluation executor stopped")
            except Exception as e:
                print(f"⚠️ Error stopping evaluation executor: {e}")

            # Shutdown screenshot comparison executor
            try:
                print("🔪 Shutting down screenshot comparison executor...")
                self.screenshot_comparison_executor.shutdown(wait=True, cancel_futures=False)
                print("✓ Screenshot comparison executor stopped")
            except Exception as e:
                print(f"⚠️ Error stopping screenshot comparison executor: {e}")

            # Stop HTTP stack
            try:
                self.http_stack.stop()
            except Exception as e:
                print(f"⚠️ Error stopping HTTP stack: {e}")

            # Stop monitor
            try:
                print("🛑 Stopping task monitor...")
                self.monitor.stop_monitoring()
                print("✅ Task monitor stopped")
            except Exception as e:
                print(f"⚠️ Error stopping monitor: {e}")

            # Skip final cleanup - instances are released per-task in the finally block (line 980-1010)
            # Initial cleanup at next iteration start (line 558-559) will handle any leftover instances
            # Removing this prevents race condition where master releases instances still in use by workers
            print("⏭️  Skipping final bulk instance cleanup (per-task cleanup already done, startup cleanup handles leftovers)")

            print("✅ Process-based cleanup complete")

    def _process_single_task_with_process_isolation(self, task_info: Dict, agent) -> List[Dict]:
        """
        Process a single task using process isolation for HTTP requests
        """
        if self._shutdown_requested.is_set():
            return []
        task_id = task_info['task_id']
        task = task_info['task']
        trajectory = []
        instance = None
        task_start_time = time.time()
        TASK_TIMEOUT_SECONDS = self.task_timeout_minutes * 60

        def check_task_timeout():
            if self._shutdown_requested.is_set():
                raise Exception("Shutdown requested")
            elapsed = time.time() - task_start_time
            if elapsed > (self.task_timeout_minutes * 60):
                raise Exception(f"Task timeout: exceeded {self.task_timeout_minutes} minutes")

        with self.stats_lock:
            self.system_stats['tasks_started'] += 1
        
        try:
            # Allocation phase
            self.monitor.start_allocation_wait(task_id, task.task_name[:30])

            # Import the standalone function
            from .pickleable_http_functions import allocate_instance

            # Allocate instance using process isolation
            check_task_timeout()

            try:
                instance = self.http_stack.execute(
                    func=allocate_instance,
                    args=(self.master_host, self.master_port, self.api_key),
                    task_id=task_id,
                    func_name="allocate_instance",
                    check_timeout=check_task_timeout,
                    timeout=120  # 2 minutes for allocation
                )
            except Exception as e:
                # Enhanced error handling for instance allocation
                error_str = str(e).lower()
                if 'process timeout' in error_str or 'killed' in error_str:
                    raise Exception(f"Instance allocation process was killed due to timeout: {str(e)}")
                elif 'pool not running' in error_str:
                    raise Exception(f"Process pool not running during instance allocation: {str(e)}")
                else:
                    raise Exception(f"Instance allocation failed: {str(e)}")

            if not instance:
                raise Exception("Failed to allocate browser instance")

            with self.stats_lock:
                self.system_stats['instances_allocated'] += 1

            # Process task steps with process isolation and enhanced error handling
            try:
                trajectory = self._process_task_steps_with_process_isolation(
                    instance, task_info, agent, check_task_timeout
                )
            except Exception as e:
                # Check if this was a process timeout/kill during task steps
                error_str = str(e).lower()
                if 'process timeout' in error_str or 'killed' in error_str:
                    step_info = ""
                    if len(trajectory) > 0:
                        step_info = f" at step {len(trajectory)}"
                    raise Exception(f"Task processing{step_info} was killed due to process timeout: {str(e)}")
                else:
                    raise e
            
            # Success
            task_duration = time.time() - task_start_time
            with self.stats_lock:
                self.system_stats['tasks_completed'] += 1
                total_done = self.system_stats['tasks_completed']
                current_avg = self.system_stats['avg_task_duration']
                self.system_stats['avg_task_duration'] = (
                    (current_avg * (total_done - 1) + task_duration) / total_done
                )
            
            self.monitor.finish_task(task_id, True)
            
        except Exception as e:
            error_msg = str(e)
            task_duration = time.time() - task_start_time

            # Enhanced error categorization for process kills
            if any(keyword in error_msg.lower() for keyword in ['process timeout', 'killed', 'process was killed']):
                # This was definitely a process kill
                kill_type = "PROCESS_KILLED"
                detailed_error = f"[{kill_type}] Process was forcibly killed due to timeout after {task_duration:.1f}s: {error_msg}"
                print(f"🔪 {task_id}: Process killed - {_truncate_error_msg(error_msg)}")
            elif "timeout" in error_msg.lower():
                # Some other kind of timeout
                kill_type = "TIMEOUT"
                detailed_error = f"[{kill_type}] Operation timed out after {task_duration:.1f}s: {error_msg}"
                print(f"⏰ {task_id}: Timeout - {_truncate_error_msg(error_msg)}")
            elif "shutdown requested" in error_msg.lower():
                # Shutdown case
                kill_type = "SHUTDOWN"
                detailed_error = f"[{kill_type}] Task aborted due to shutdown: {error_msg}"
                print(f"🛑 {task_id}: Shutdown - {error_msg}")
            elif "pool not running" in error_msg.lower():
                # Pool failure case
                kill_type = "POOL_FAILURE"
                detailed_error = f"[{kill_type}] Process pool failure after {task_duration:.1f}s: {error_msg}"
                print(f"🏊 {task_id}: Pool failure - {error_msg}")
            else:
                # Other error
                kill_type = "ERROR"
                detailed_error = f"[{kill_type}] Task failed after {task_duration:.1f}s: {error_msg}"
                print(f"❌ {task_id}: Failed - {_truncate_error_msg(error_msg)}")

            # Mark task as failed in monitor with detailed error message
            self.monitor.finish_task(task_id, False, detailed_error)

            with self.stats_lock:
                self.system_stats['tasks_failed'] += 1

            # Always append error step to current trajectory (even if empty)
            # This preserves any successful steps that were completed before the error
            # Instead of replacing entire trajectory with dummy, just append error step and return
            trajectory = self._append_error_step_to_trajectory(
                trajectory if trajectory else [], task, detailed_error, task_info['output_dir']
            )
                
        finally:
            # Release instance using DIRECT sequential approach (not process isolation)
            # IMPORTANT: All nodes (master + worker) must release their allocated instances
            # to return them to the shared pool and prevent capacity exhaustion
            if instance:
                try:
                    # Use direct client call instead of process isolation to avoid cascade failures
                    self._release_single_instance_direct(instance, task_id)

                    with self.stats_lock:
                        self.system_stats['instances_released'] += 1

                except Exception as e:
                    if not self._shutdown_requested.is_set():
                        error_msg = str(e).lower()

                        # Handle UUID errors gracefully - don't treat as critical failures
                        if any(pattern in error_msg for pattern in [
                            'invalid instance uuid', 'invalid uuid', 'instance not found', 'does not exist'
                        ]):
                            print(f"🔄 {task_id}: Instance already released - {e}")
                            # Don't record this as an error, it's expected behavior
                            with self.stats_lock:
                                self.system_stats['instances_released'] += 1  # Count as success
                        elif any(keyword in error_msg for keyword in ['process timeout', 'killed', 'process was killed']):
                            print(f"🔪 {task_id}: Process killed during instance reset - {e}")
                        elif "pool not running" in error_msg:
                            print(f"🏊 {task_id}: Pool failure during instance reset - {e}")
                        else:
                            print(f"⚠️ {task_id}: Failed to release instance: {str(e)[:100]}")

        # Submit screenshot comparison asynchronously (non-blocking)
        if trajectory and len(trajectory) > 0:
            self._submit_screenshot_comparison(trajectory, task_id, task_info['index'])

        return trajectory or []

    def _process_task_steps_with_process_isolation(self, instance: Dict, task_info: Dict,
                                                 agent, check_timeout: Callable) -> List[Dict]:
        """
        Process task steps using process isolation for all HTTP operations
        """
        from .pickleable_http_functions import (
            get_metadata, navigate_with_retries, navigate_with_retries_and_correction, capture_screenshot,
            get_ac_tree, get_page_metadata, execute_command, wait_for_content
        )

        task_id = task_info['task_id']
        task = task_info['task']
        output_dir = task_info['output_dir']
        trajectory = []

        try:
            # Start task
            self.monitor.start_task(task_id, task.task_name[:30])

            # Get screen dimensions with process isolation
            self.monitor.set_task_getting_metadata(task_id)
            try:
                screen_metadata = self.http_stack.execute(
                    func=get_metadata,
                    args=(self.master_host, self.master_port, self.api_key, instance),
                    task_id=task_id,
                    func_name="get_screen_dimensions",
                    check_timeout=check_timeout,
                    timeout=60
                )
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['process timeout', 'killed', 'process was killed']):
                    print(f"🔪 {task_id}: Process killed during metadata retrieval - {e}")
                    raise Exception(f"[PROCESS_KILLED] Metadata retrieval process was forcibly killed: {e}")
                else:
                    raise Exception(f"[ERROR] Metadata retrieval failed: {e}")
            screen_dimensions = (
                screen_metadata.get('width', 1280),
                screen_metadata.get('height', 768)
            )

            with self.stats_lock:
                self.system_stats['operations']['navigations'] += 1

            # Navigate with process isolation
            self.monitor.set_task_navigating(task_id, task.website)
            try:
                self.http_stack.execute(
                    func=navigate_with_retries_and_correction,
                    args=(self.master_host, self.master_port, self.api_key, instance, task_id, task.website, self.max_retries, None),
                    task_id=task_id,
                    func_name="navigate_to_website",
                    check_timeout=check_timeout,
                    timeout=120  # 2 minutes for navigation
                )
            except Exception as e:
                # This should rarely be triggered now that navigate_with_retries returns success
                # Only catches process timeout/killed errors
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['process timeout', 'killed', 'process was killed']):
                    print(f"🔪 {task_id}: Process killed during initial navigation - {e}")
                    raise Exception(f"[PROCESS_KILLED] Navigation process was forcibly killed: {e}")
                else:
                    # Non-critical navigation errors - log but continue
                    log_navigation_error(task.website, task_id)
                    print(f"⚠️ {task_id}: Initial navigation encountered error (continuing trajectory): {_truncate_error_msg(str(e))}")
                    # Don't raise - let trajectory continue

            # Content check to ensure page has loaded (no artificial wait - server-side already waits for load state)
            try:
                self.http_stack.execute(
                    func=wait_for_content,
                    args=(self.master_host, self.master_port, self.api_key, instance),
                    task_id=task_id,
                    func_name="wait_for_content",
                    check_timeout=check_timeout,
                    timeout=15
                )
            except Exception:
                # Continue even if content wait fails (page might still be usable)
                pass

            # Process steps - use task-specific max_steps based on difficulty
            if task.max_steps is None:
                raise ValueError(f"Task {task_id} has no max_steps defined")
            max_steps_for_task = task.max_steps
            print(f"🎯 {task_id}: Using difficulty-based max_steps={max_steps_for_task} (difficulty={task.difficulty})")

            step = 0
            goback_from_homepage_count = 0  # Track repeated goback attempts from homepage
            goback_warning_logged = False  # Track if we've already logged the warning for test mode
            while step < max_steps_for_task:
                check_timeout()
                self.monitor.update_task_step(task_id, step)

                # Screenshot with process isolation
                self.monitor.set_task_taking_screenshot(task_id, step)

                # Get previous step context for better error messages
                prev_screenshot_path = None
                prev_action = None
                if len(trajectory) > 0 and step > 0:
                    prev_step = trajectory[-1]
                    if prev_step.get('observation'):
                        prev_screenshot_path = prev_step['observation'].image_path
                    if prev_step.get('action') and prev_step['action'].action:
                        prev_action = prev_step['action'].action

                # Track if screenshot is a fallback (due to blank page after agent navigation)
                is_fallback_screenshot = False
                try:
                    screenshot_result = self.http_stack.execute(
                        func=capture_screenshot,
                        args=(
                            self.master_host, self.master_port, self.api_key, instance,
                            output_dir, step, agent.context_manager.get_interaction_mode(), self.max_retries,
                            prev_screenshot_path, prev_action  # Pass context for error messages
                        ),
                        task_id=task_id,
                        func_name=f"screenshot_step_{step}",
                        check_timeout=check_timeout,
                        timeout=180  # 3 minutes for screenshot
                    )
                    # capture_screenshot now returns (path, is_fallback) tuple
                    if isinstance(screenshot_result, tuple):
                        screenshot_path, is_fallback_screenshot = screenshot_result
                    else:
                        # Backward compatibility if somehow a plain path is returned
                        screenshot_path = screenshot_result
                except Exception as e:
                    error_msg = str(e).lower()
                    if any(keyword in error_msg for keyword in ['process timeout', 'timeout', 'killed', 'process was killed']):
                        # Print last screenshot for debugging
                        last_screenshot = "None"
                        if len(trajectory) > 0 and trajectory[-1].get('observation'):
                            last_screenshot = trajectory[-1]['observation'].image_path
                        print(f"🔪 {task_id}: Process killed during screenshot at step {step}")
                        print(f"📸 {task_id}: Last screenshot before timeout: {last_screenshot}")
                        raise Exception(f"[PROCESS_KILLED] Screenshot process was forcibly killed at step {step}: {e}")
                    else:
                        raise Exception(f"[ERROR] Screenshot failed at step {step}: {e}")

                with self.stats_lock:
                    self.system_stats['operations']['screenshots'] += 1

                # AC Tree with process isolation (skip for coordinates mode - not used in prompts)
                if self.interaction_mode == 'set_of_marks':
                    self.monitor.set_task_getting_ac_tree(task_id)
                    try:
                        ac_tree = self.http_stack.execute(
                            func=get_ac_tree,
                            args=(self.master_host, self.master_port, self.api_key, instance),
                            task_id=task_id,
                            func_name=f"get_ac_tree_step_{step}",
                            check_timeout=check_timeout,
                            timeout=60
                        )
                    except Exception as e:
                        error_msg = str(e).lower()
                        if any(keyword in error_msg for keyword in ['process timeout', 'timeout', 'killed', 'process was killed']):
                            print(f"🔪 {task_id}: Process killed during AC tree retrieval at step {step}")
                            print(f"📸 {task_id}: Current step screenshot: {screenshot_path}")
                            raise Exception(f"[PROCESS_KILLED] AC tree retrieval process was forcibly killed at step {step}: {e}")
                        else:
                            raise Exception(f"[ERROR] AC tree retrieval failed at step {step}: {e}")

                    with self.stats_lock:
                        self.system_stats['operations']['ac_trees'] += 1
                else:
                    # Coordinates mode: skip AC tree fetch (not used in prompts)
                    ac_tree = ""

                # Page metadata with process isolation
                self.monitor.set_task_getting_metadata(task_id)
                try:
                    page_metadata = self.http_stack.execute(
                        func=get_page_metadata,
                        args=(self.master_host, self.master_port, self.api_key, instance),
                        task_id=task_id,
                        func_name=f"get_page_metadata_step_{step}",
                        check_timeout=check_timeout,
                        timeout=60
                    )
                except Exception as e:
                    error_msg = str(e).lower()
                    if any(keyword in error_msg for keyword in ['process timeout', 'timeout', 'killed', 'process was killed']):
                        print(f"🔪 {task_id}: Process killed during page metadata retrieval at step {step}")
                        print(f"📸 {task_id}: Screenshot at timeout: {screenshot_path}")
                        raise Exception(f"[PROCESS_KILLED] Page metadata retrieval process was forcibly killed at step {step}: {e}")
                    else:
                        raise Exception(f"[ERROR] Page metadata retrieval failed at step {step}: {e}")

                # Create observation
                observation = Observation(
                    task=task,
                    image_path=screenshot_path,
                    ac_tree=ac_tree,
                    page_metadata=page_metadata
                )

                # Add step to trajectory with placeholder reward
                trajectory_step = {
                    'observation': observation,
                    'action': None,
                    'response': None,
                    'reward': Reward(reward=0, evaluation="Task in progress", is_blocked=False)  # Placeholder reward
                }
                trajectory.append(trajectory_step)

                # Generate observation text for PREVIOUS step (if exists)
                if len(trajectory) > 1:
                    prev_step = trajectory[-2]
                    prev_screenshot = prev_step['observation'].image_path if prev_step.get('observation') else None
                    prev_action = prev_step.get('action')

                    # Compare screenshots and get metadata
                    observation_text = self._generate_observation_text(
                        prev_screenshot=prev_screenshot,
                        current_screenshot=screenshot_path,
                        page_metadata=page_metadata,
                        prev_action=prev_action,
                        is_fallback_screenshot=is_fallback_screenshot
                    )

                    # Store observation in previous step's response
                    if prev_step.get('response'):
                        prev_step['response'].answering_tokens['observation'] = observation_text

                # Get action from vLLM (no process isolation needed - different server)
                check_timeout()
                self.monitor.set_task_getting_action(task_id)

                step_data = {
                    'screenshot_path': screenshot_path,
                    'ac_tree': ac_tree,
                    'page_metadata': page_metadata,
                    'step': step
                }

                try:
                    action, response_obj = agent.get_action_and_observation_sync(
                        trajectory, screenshot_path, page_metadata, step_data
                    )

                    with self.stats_lock:
                        self.system_stats['operations']['vllm_calls'] += 1

                    self.monitor.set_task_normal_phase(task_id)
                except Exception as vllm_error:
                    # vLLM request failed - mark task as error
                    error_msg = f"vLLM request failed at step {step}: {str(vllm_error)}"
                    print(f"❌ {task_id}: {_truncate_error_msg(error_msg)}")
                    # Re-raise to mark task as failed
                    raise Exception(f"[ERROR] {error_msg}")

                # Update trajectory step
                if response_obj:
                    trajectory_step['action'] = action
                    trajectory_step['response'] = response_obj

                # Check for answer
                if action and action.action and action.action.get('key') == 'answer':
                    # Submit for async evaluation (non-blocking)
                    self._submit_evaluation(trajectory, agent, task_id, task_info['index'])
                    break

                # Execute action with process isolation
                if action and action.action:
                    self.monitor.set_task_executing_action(task_id, action.action_string[:30])

                    browser_command = agent.context_manager.parse_action_to_browser_command(
                        action, screen_dimensions, homepage_url=task.website
                    )

                    # Check current page location and reset counter if not at homepage
                    current_url = page_metadata.get('url', '') if page_metadata else ''
                    task_url = task.website if hasattr(task, 'website') else ''

                    # Normalize URLs for comparison (remove trailing slash, http/https, www)
                    def normalize_url(url):
                        url = url.lower().strip()
                        url = url.replace('https://', '').replace('http://', '')
                        url = url.rstrip('/')
                        if url.startswith('www.'):
                            url = url[4:]
                        return url

                    current_normalized = normalize_url(current_url)
                    task_normalized = normalize_url(task_url)
                    is_at_homepage = (current_normalized == task_normalized or current_url.startswith(task_url))

                    # Check if agent is trying to go back from homepage
                    action_key = action.action.get('key', '')
                    is_back_action = (action_key == 'goback' or 'back' in browser_command)
                    skip_action_execution = False

                    # Reset counter if agent is not at homepage (navigated away successfully)
                    # OR if agent does any action other than goback from homepage
                    if not is_at_homepage and goback_from_homepage_count > 0:
                        goback_from_homepage_count = 0
                        goback_warning_logged = False
                        # Optional: print confirmation
                        # print(f"✓ {task_id}: Agent navigated away from homepage, reset goback counter")
                    elif is_at_homepage and not is_back_action and goback_from_homepage_count > 0:
                        # Reset counter when agent does a different action (not goback) while at homepage
                        # This ensures we only count CONSECUTIVE goback attempts
                        goback_from_homepage_count = 0
                        goback_warning_logged = False
                        # print(f"✓ {task_id}: Agent did non-goback action at homepage, reset goback counter")

                    if is_back_action and is_at_homepage:
                        # Agent is trying to go back while already at homepage
                        goback_from_homepage_count += 1

                        if goback_from_homepage_count == 1:
                            # First time: Execute goback to homepage
                            print(f"🏠 {task_id}: Agent tried to go back from homepage (attempt {goback_from_homepage_count}/3), executing goback to homepage")
                            browser_command = {"visit_page": {"url": task.website}}
                        else:
                            # Second and third times: Skip execution, just take screenshot
                            print(f"🏠 {task_id}: Agent tried to go back from homepage (attempt {goback_from_homepage_count}/3), skipping execution")
                            skip_action_execution = True

                    # Track if we navigated to homepage (need longer wait for page load)
                    navigated_to_homepage = is_back_action and is_at_homepage and goback_from_homepage_count == 1

                    # Print navigation debug message if this is a navigate action
                    if action_key == 'navigate' and 'visit_page' in browser_command:
                        destination_url = browser_command['visit_page'].get('url', 'unknown')
                        print(f"🧭 {task_id}: Agent navigating from '{current_url}' to '{destination_url}'")

                    try:
                        if not skip_action_execution:
                            result = self.http_stack.execute(
                                func=execute_command,
                                args=(self.master_host, self.master_port, self.api_key, instance, browser_command, self.max_retries),
                                task_id=task_id,
                                func_name=f"execute_action_step_{step}",
                                check_timeout=check_timeout,
                                timeout=120  # 2 minutes for action execution
                            )

                            with self.stats_lock:
                                self.system_stats['operations']['actions_executed'] += 1

                            # DISABLED: Allow agent to navigate to other websites (e.g., to bypass reCAPTCHA)
                            # The navigate action is intended to let agents go to alternative websites
                            # if self._is_navigation_away_from_task(browser_command, task.website):
                            #     print(f"🔄 {task_id}: Agent tried to navigate away from task URL, redirecting back to {task.website}")
                            #     try:
                            #         # Force navigate back to the task URL
                            #         self.http_stack.execute(
                            #             func=navigate_with_retries,
                            #             args=(self.master_host, self.master_port, self.api_key, instance, task_id, task.website, self.max_retries),
                            #             task_id=task_id,
                            #             func_name=f"force_navigate_back_step_{step}",
                            #             check_timeout=check_timeout,
                            #             timeout=60  # 1 minute for corrective navigation
                            #         )
                            #         print(f"✅ {task_id}: Successfully redirected back to task URL")
                            #     except Exception as nav_error:
                            #         print(f"⚠️ {task_id}: Failed to redirect back to task URL: {_truncate_error_msg(str(nav_error))}")

                        # Wait longer after navigating to homepage to allow page to fully load
                        # This prevents blank screenshots at the next step
                        if navigated_to_homepage:
                            print(f"🕐 {task_id}: Waiting 10s for homepage to load after navigation...")
                            time.sleep(10.0)
                        else:
                            time.sleep(1.0)

                        # Check if agent has tried to go back from homepage 3 times
                        # For train: terminate trajectory with reward 0
                        # For test: just log once but continue (allow agent to explore)
                        if goback_from_homepage_count >= 3 and self.split == 'train':
                            print(f"🔴 {task_id}: Agent attempted goback from homepage 3 times - terminating trajectory with reward 0")
                            # Add reward to last trajectory step
                            if len(trajectory) > 0:
                                trajectory[-1]['reward'] = Reward(
                                    reward=0,
                                    evaluation="Agent attempted to go back from homepage 3 times. Trajectory terminated.",
                                    is_blocked=False
                                )
                            break
                        elif goback_from_homepage_count >= 3 and self.split == 'test' and not goback_warning_logged:
                            # Test mode: just log once but let trajectory continue
                            print(f"⚠️ {task_id}: Agent attempted goback from homepage 3 times - continuing (test mode)")
                            goback_warning_logged = True

                    except Exception as action_error:
                        # Enhanced action execution error handling with process kill detection
                        error_msg = str(action_error).lower()
                        if any(keyword in error_msg for keyword in ['process timeout', 'timeout', 'killed', 'process was killed']):
                            # Print screenshot for debugging - use current step's screenshot from trajectory
                            current_screenshot = trajectory[-1]['observation'].image_path if len(trajectory) > 0 and trajectory[-1].get('observation') else "None"
                            print(f"🔪 {task_id}: Process killed during action execution at step {step}")
                            print(f"📸 {task_id}: Screenshot at timeout: {current_screenshot}")
                            raise Exception(f"[PROCESS_KILLED] Action execution process was forcibly killed at step {step}: {action_error}")
                        else:
                            # For other errors (not process kills), silently continue execution
                            # The execute_command function now handles retries internally
                            # Don't raise exception - let task continue
                            pass

                step += 1

            # Generate observation for the LAST step (final state)
            if len(trajectory) > 0:
                last_step = trajectory[-1]
                if last_step.get('response'):
                    # For the last step, we don't have a "next" screenshot to compare
                    # So just use the current page metadata
                    last_observation = last_step['observation']
                    last_metadata = {'title': 'Final state', 'url': 'Task completed or terminated'}

                    # Try to get actual metadata if available
                    try:
                        last_metadata = {
                            'title': last_observation.task.task_name if hasattr(last_observation, 'task') else 'Unknown',
                            'url': 'Final step'
                        }
                    except:
                        pass

                    observation_text = f"Final step. {last_metadata.get('title', 'Unknown')}"
                    last_step['response'].answering_tokens['observation'] = observation_text

            # Compute reward (no process isolation needed - different service)
            self.monitor.set_task_computing_reward(task_id)
            self._add_reward(trajectory, agent, task_id)

            with self.stats_lock:
                self.system_stats['operations']['rewards_computed'] += 1

            self.monitor.set_task_normal_phase(task_id)
            return trajectory

        except Exception as e:
            print(f"❌ Task steps failed for {task_id}: {_truncate_error_msg(str(e))}")

            # Re-raise the exception so the task gets marked as failed, not finished
            raise Exception(f"Task steps failed: {str(e)}")

    def _generate_observation_text(self, prev_screenshot: str, current_screenshot: str, page_metadata: dict, prev_action=None, is_fallback_screenshot: bool = False) -> str:
        """
        Generate observation text by comparing screenshots and including page metadata.

        Args:
            prev_screenshot: Path to previous screenshot
            current_screenshot: Path to current screenshot
            page_metadata: Dictionary with page metadata (title, url)
            prev_action: Previous action object (to check if it was a failed navigation)
            is_fallback_screenshot: True if current screenshot is a fallback due to blank page after navigation

        Returns:
            Observation text describing what happened after the action
        """
        # Extract page metadata
        title = page_metadata.get('title', 'Unknown') if page_metadata else 'Unknown'
        url = page_metadata.get('url', 'Unknown') if page_metadata else 'Unknown'

        # Shorten URL if too long (keep domain and first part of path)
        if url != 'Unknown' and len(url) > 60:
            try:
                import urllib.parse
                parsed = urllib.parse.urlparse(url)
                url = f"{parsed.netloc}{parsed.path[:30]}..." if len(parsed.path) > 30 else f"{parsed.netloc}{parsed.path}"
            except:
                url = url[:60] + "..."

        # Check if the previous action was a navigate action
        is_navigate_action = False
        navigate_url = None
        if prev_action and hasattr(prev_action, 'action'):
            action_key = prev_action.action.get('key', '')
            if action_key == 'navigate':
                is_navigate_action = True
                navigate_url = prev_action.action.get('arguments', {}).get('url', '')

        # If this is a fallback screenshot due to blank page after navigation, provide explicit feedback
        if is_fallback_screenshot and is_navigate_action and navigate_url:
            observation = f"Navigation failed: The website '{navigate_url}' returned a blank page and is not accessible. The screenshot shows the previous page before the failed navigation. Please try navigating to a different website or use a different approach to complete the task. Current URL: {url}"
            return observation

        # Compare screenshots to check if they're identical
        images_identical = False
        if prev_screenshot and current_screenshot:
            try:
                import filecmp
                images_identical = filecmp.cmp(prev_screenshot, current_screenshot, shallow=False)
            except:
                images_identical = False

        # Generate observation text
        if images_identical:
            # If it was a navigate action that failed, provide explicit message
            if is_navigate_action and navigate_url:
                observation = f"Navigation failed: The website '{navigate_url}' is not accessible or does not exist. The page did not change. Please try navigating to a different website or use a different approach to complete the task. Current URL: {url}"
            else:
                observation = f"After the action above is executed by the environment, the webpage did not change (this means the last action is not effective). The URL of the webpage after executing the action: {url}"
        else:
            observation = f"After the action above is executed by the environment, the webpage changed (this means the last action was effective). The URL of the webpage after executing the action: {url}"

        return observation

    def _is_navigation_away_from_task(self, browser_command: dict, task_url: str) -> bool:
        """
        Check if the browser command is a navigation that takes us away from the task URL.
        Returns True if we should redirect back to the task URL.
        """
        if not browser_command or not isinstance(browser_command, dict):
            return False

        # Check if this is a visit_page command
        if 'visit_page' in browser_command:
            command_url = browser_command['visit_page'].get('url', '')
            if command_url:
                # Extract domain from both URLs for comparison
                import urllib.parse
                try:
                    task_domain = urllib.parse.urlparse(task_url).netloc.lower()
                    command_domain = urllib.parse.urlparse(command_url).netloc.lower()

                    # If navigating to a different domain, redirect back
                    if command_domain and command_domain != task_domain:
                        return True

                    # Check for common "back" navigation patterns that might break the task
                    if any(pattern in command_url.lower() for pattern in ['javascript:history.back', 'history.go(-1)', 'about:blank']):
                        return True

                except Exception:
                    # If URL parsing fails, be safe and redirect back
                    return True

        return False

    def _append_error_step_to_trajectory(self, trajectory: List[Dict], task: Task, error_msg: str, output_dir: str) -> List[Dict]:
        """
        Append an error step to an existing trajectory when an error occurs after successful steps.
        
        Args:
            trajectory: Existing trajectory with successful steps
            task: The task object
            error_msg: The error message from the server or exception
            output_dir: Directory to save the dummy error image
        
        Returns:
            Updated trajectory with error step appended
        """
        # Create dummy image for error step with same dimensions as screenshots
        error_step = len(trajectory)
        dummy_image_path = self._create_dummy_image(
            output_dir=output_dir,
            step=error_step,
            width=1280,  # Default screen width
            height=768,  # Default screen height
            error_text="Error Occurred"
        )
        
        # Create observation for error step
        observation = Observation(
            task=task,
            image_path=dummy_image_path,
            ac_tree=""
        )
        
        # Create action with "invalid_step" key and error details
        action = Action(
            action={"key": "invalid_step", "arguments": {"error": error_msg}},
            action_string=f"Error at step {len(trajectory)}: {error_msg}"
        )
        
        # Create response with error information
        # Create Response directly (no model parsing needed for error steps)
        error_response_text = f"Task failed at step {len(trajectory)} due to server error: {error_msg}"
        response = Response(
            raw_response=error_response_text,
            answering_tokens={"action": f"Error at step {len(trajectory)}"},
            raw_prompt=""
        )

        # Add reward indicating failure
        reward = Reward(reward=0, evaluation=f"Task failed: {error_msg}", is_blocked=False)
        
        # Append the error step
        trajectory.append({
            'observation': observation,
            'action': action,
            'response': response,
            'reward': reward
        })
        
        print(f"   📝 Appended error step to trajectory (total steps: {len(trajectory)})")
        
        return trajectory

    # Helper methods remain the same...
    def _execute_command(self, instance, command):
        """Execute command on instance"""
        result = self.master_client.execute(instance, command)
        if result.status_code != 200:
            raise Exception(f"Command failed with status {result.status_code}: {result.text}")
        return result

    def _add_reward(self, trajectory, agent, task_id):
        """Add reward to trajectory"""
        if not trajectory or 'reward' in trajectory[-1]:
            return

        try:
            if trajectory[-1]['action'] and trajectory[-1]['action'].action['key'] == 'answer':
                reward_value, evaluation, is_blocked = agent.evaluator.get_verifiable_reward(trajectory)
                reward = Reward(reward=reward_value, evaluation=evaluation, is_blocked=is_blocked)

                # Check if website blocked the agent (detected by verifier)
                if is_blocked:
                    print(f"🚫 {task_id}: Verifier detected website blocking (fallback) - recording in blocklist")
                    if self.blocklist_manager and trajectory[-1].get('observation'):
                        obs = trajectory[-1]['observation']
                        url = getattr(obs.task, 'url', None) if hasattr(obs, 'task') else None
                        screenshot_path = obs.image_path if hasattr(obs, 'image_path') else "unknown"
                        if url:
                            self.blocklist_manager.record_blocked_website(
                                url=url,
                                task_id=task_id,
                                screenshot_path=screenshot_path
                            )
            else:
                # Agent did not answer - check if website blocked it
                print(f"🔍 {task_id}: No answer provided - checking if website blocked the agent")
                is_blocked = agent.evaluator.check_if_blocked(trajectory)

                if is_blocked:
                    print(f"🚫 {task_id}: Verifier detected website blocking (no answer case) - recording in blocklist")
                    if self.blocklist_manager and trajectory[-1].get('observation'):
                        obs = trajectory[-1]['observation']
                        url = getattr(obs.task, 'url', None) if hasattr(obs, 'task') else None
                        screenshot_path = obs.image_path if hasattr(obs, 'image_path') else "unknown"
                        if url:
                            self.blocklist_manager.record_blocked_website(
                                url=url,
                                task_id=task_id,
                                screenshot_path=screenshot_path
                            )
                    reward = Reward(reward=0, evaluation=f"Website blocked the agent - detected by verifier", is_blocked=True)
                else:
                    reward = Reward(reward=0, evaluation="Task incomplete - no answer provided", is_blocked=False)
        except Exception as e:
            print(f"⚠️  {task_id}: Reward calculation failed: {str(e)[:60]}")
            reward = Reward(reward=0, evaluation=f"Reward calculation failed: {str(e)}", is_blocked=False)

        trajectory[-1]['reward'] = reward

    def _evaluate_async(self, trajectory, agent, task_id, trajectory_idx):
        """
        Evaluate trajectory asynchronously in background thread.
        This is called by the evaluation executor and updates the trajectory in place.
        """
        try:
            # Perform the expensive evaluation
            # Note: get_verifiable_reward internally calls judge_submission_images which sets submit flags
            reward_value, evaluation, is_blocked = agent.evaluator.get_verifiable_reward(trajectory)

            # Preserve submit and submission_judgment from the last step if they were set by judge_submission_images
            submit_flag = False
            submission_judgment = None
            if trajectory[-1].get('reward'):
                old_reward = trajectory[-1]['reward']
                submit_flag = getattr(old_reward, 'submit', False)
                submission_judgment = getattr(old_reward, 'submission_judgment', None)

            reward = Reward(
                reward=reward_value,
                evaluation=evaluation,
                is_blocked=is_blocked,
                submit=submit_flag,
                submission_judgment=submission_judgment
            )

            # Check if website blocked the agent (detected by verifier)
            if is_blocked:
                print(f"🚫 {task_id}: Verifier detected website blocking (async) - recording in blocklist")
                if self.blocklist_manager and trajectory[-1].get('observation'):
                    obs = trajectory[-1]['observation']
                    url = getattr(obs.task, 'url', None) if hasattr(obs, 'task') else None
                    screenshot_path = obs.image_path if hasattr(obs, 'image_path') else "unknown"
                    if url:
                        self.blocklist_manager.record_blocked_website(
                            url=url,
                            task_id=task_id,
                            screenshot_path=screenshot_path
                        )

            # Update trajectory with real reward
            trajectory[-1]['reward'] = reward

            # Update trajectories list
            self.trajectories[trajectory_idx] = trajectory

            print(f"✅ {task_id}: Async evaluation complete (reward={reward_value})")

        except Exception as e:
            print(f"⚠️  {task_id}: Async evaluation failed: {str(e)[:100]}")
            # Preserve submit flag even on error
            submit_flag = False
            submission_judgment = None
            if trajectory[-1].get('reward'):
                old_reward = trajectory[-1]['reward']
                submit_flag = getattr(old_reward, 'submit', False)
                submission_judgment = getattr(old_reward, 'submission_judgment', None)

            trajectory[-1]['reward'] = Reward(
                reward=0,
                evaluation=f"Async evaluation failed: {str(e)}",
                is_blocked=False,
                submit=submit_flag,
                submission_judgment=submission_judgment
            )
            self.trajectories[trajectory_idx] = trajectory

        finally:
            # Remove from pending evaluations
            with self.evaluation_lock:
                self.pending_evaluations.pop(task_id, None)

    def _submit_evaluation(self, trajectory, agent, task_id, trajectory_idx):
        """Submit trajectory for async evaluation and return placeholder reward"""
        # Create placeholder reward immediately
        # Note: submit flag will be set during async evaluation by judge_submission_images
        placeholder = Reward(reward=-1, evaluation="Evaluation in progress...", is_blocked=False, submit=False)
        trajectory[-1]['reward'] = placeholder

        # Submit to evaluation executor
        future = self.evaluation_executor.submit(
            self._evaluate_async,
            trajectory,
            agent,
            task_id,
            trajectory_idx
        )

        # Track pending evaluation
        with self.evaluation_lock:
            self.pending_evaluations[task_id] = future

        print(f"⏳ {task_id}: Evaluation queued (async)")

    def _wait_for_pending_evaluations(self):
        """Wait for all pending evaluations to complete before shutdown"""
        with self.evaluation_lock:
            pending_count = len(self.pending_evaluations)
            if pending_count > 0:
                print(f"⏳ Waiting for {pending_count} pending evaluations to complete...")
                pending_futures = list(self.pending_evaluations.values())

        # Wait outside the lock to avoid blocking
        if pending_count > 0:
            for future in as_completed(pending_futures):
                try:
                    future.result()  # Wait for completion
                except Exception as e:
                    print(f"⚠️  Evaluation error during shutdown: {str(e)[:100]}")

            print(f"✅ All pending evaluations complete")

    def _compare_screenshots_async(self, trajectory, task_id, trajectory_idx):
        """
        Compare screenshots asynchronously in background thread.
        This adds 'same_as_next_screenshot' field to each step in the trajectory.
        """
        try:
            # Perform the screenshot comparison
            trajectory = self._add_screenshot_comparison_to_trajectory(trajectory)

            # Update trajectories list with the modified trajectory
            self.trajectories[trajectory_idx] = trajectory

            print(f"✅ {task_id}: Screenshot comparison complete")

        except Exception as e:
            print(f"⚠️  {task_id}: Screenshot comparison failed: {str(e)[:100]}")
            # Don't fail the trajectory if screenshot comparison fails
            # Just leave it without the comparison metadata

        finally:
            # Remove from pending comparisons
            with self.screenshot_comparison_lock:
                self.pending_screenshot_comparisons.pop(task_id, None)

    def _submit_screenshot_comparison(self, trajectory, task_id, trajectory_idx):
        """Submit trajectory for async screenshot comparison"""
        # Submit to screenshot comparison executor
        future = self.screenshot_comparison_executor.submit(
            self._compare_screenshots_async,
            trajectory,
            task_id,
            trajectory_idx
        )

        # Track pending comparison
        with self.screenshot_comparison_lock:
            self.pending_screenshot_comparisons[task_id] = future

        print(f"⏳ {task_id}: Screenshot comparison queued (async)")

    def _wait_for_pending_screenshot_comparisons(self):
        """Wait for all pending screenshot comparisons to complete before shutdown"""
        with self.screenshot_comparison_lock:
            pending_count = len(self.pending_screenshot_comparisons)
            if pending_count > 0:
                print(f"⏳ Waiting for {pending_count} pending screenshot comparisons to complete...")
                pending_futures = list(self.pending_screenshot_comparisons.values())

        # Wait outside the lock to avoid blocking
        if pending_count > 0:
            for future in as_completed(pending_futures):
                try:
                    future.result()  # Wait for completion
                except Exception as e:
                    print(f"⚠️  Screenshot comparison error during shutdown: {str(e)[:100]}")

            print(f"✅ All pending screenshot comparisons complete")

    def _create_dummy_image(self, output_dir: str, step: int, width: int = 1280, height: int = 768, error_text: str = "Error") -> str:
        """
        Don't create dummy error images to avoid overwriting trajectory screenshots.

        Args:
            output_dir: Directory to save the image (unused)
            step: Step number for filename (unused)
            width: Image width (unused)
            height: Image height (unused)
            error_text: Text to display on the image (unused)

        Returns:
            Empty string (no image created)
        """
        # Don't create error screenshots to avoid overwriting trajectory screenshots
        return ""

    def _create_dummy_trajectory_with_error(self, task, output_dir: str, error_type: str, error_message: str) -> List[Dict]:
        """
        Create dummy trajectory with specific error information for enhanced error tracking

        Args:
            task: The task object
            output_dir: Directory to save dummy screenshot
            error_type: Type of error (PROCESS_KILLED, TIMEOUT, ERROR, etc.)
            error_message: Detailed error message

        Returns:
            List containing single trajectory step with error information
        """
        # Create error-specific dummy image
        error_display_text = {
            "PROCESS_KILLED": "Process Killed",
            "TIMEOUT": "Task Timeout",
            "SHUTDOWN": "System Shutdown",
            "GRACE_PERIOD_INCOMPLETE": "Grace Period Incomplete",
            "INCOMPLETE": "Task Incomplete",
            "ERROR": "Task Error"
        }.get(error_type, "Task Failed")

        dummy_image_path = self._create_dummy_image(
            output_dir=output_dir,
            step=0,
            width=1280,
            height=768,
            error_text=error_display_text
        )

        # Create observation with error context
        observation = Observation(
            task=task,
            image_path=dummy_image_path,
            ac_tree=f"Error: {error_type}"
        )

        # Create action with detailed error information
        action = Action(
            action={
                "key": "error",
                "arguments": {
                    "error_type": error_type,
                    "error_message": error_message[:200]  # Truncate long messages
                }
            },
            action_string=f"{error_type}: {error_message[:100]}..."
        )

        # Create response with error context
        # Create Response directly (no model parsing needed for error steps)
        response = Response(
            raw_response=f"Task failed due to {error_type}: {error_message}",
            answering_tokens={"action": f"{error_type}: {error_message[:100]}"},
            raw_prompt=""
        )

        # Zero reward for all failures
        reward = Reward(reward=0, evaluation=f"Failed - {error_type}: {error_message}", is_blocked=False)

        return [{
            'observation': observation,
            'action': action,
            'response': response,
            'reward': reward,
            'error_type': error_type,  # Additional metadata for analysis
            'error_message': error_message
        }]

