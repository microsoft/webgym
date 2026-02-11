"""
This script imports the UnifiedAgent to collect data from the web.
It interacts with the distributed collection server to collect data.
Updated to use context management and unified timeout system.
"""

import os
import torch
import hydra
import random
import pandas as pd
import transformers
import numpy as np
import json
import sys
import io
from pathlib import Path
from collections import Counter
from huggingface_hub import login
from webgym.models import WebAgent
from webgym.misc import colorful_print
from webgym.utils import BlocklistManager, RolloutSampler
from urllib.parse import urlparse
from omegaconf import DictConfig, OmegaConf
from webgym.data.components import domain_subdomain_map
from webgym.environment import AsyncWebGym
from checkpoint_utils import create_fixed_checkpoint_callback, handle_grace_period_expiry_fixed, GracePeriodExpiredException
import time

# set seed
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
transformers.logging.set_verbosity_error()

def set_seed(seed: int = 42):
    # 1. Python random
    random.seed(seed)
    
    # 2. Numpy
    np.random.seed(seed)
    
    # 3. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # 5. Environment variable
    os.environ["PYTHONHASHSEED"] = str(seed)


def filter_blocked_tasks(tasks, blocked_domains):
    """Filter out tasks from blocked domains"""
    if not blocked_domains:
        return tasks

    def extract_domain(url):
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            # Remove 'www.' prefix if present
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception:
            return url.lower()

    def is_task_blocked(task):
        """Check if a task is from a blocked domain"""
        # Check if task has URL in different possible fields
        urls_to_check = []

        # Try different URL field names that might exist in the task
        url_fields = ['url', 'page_url', 'starting_url', 'website_url']
        for field in url_fields:
            if field in task and task[field]:
                urls_to_check.append(task[field])

        # Check if any URL is from a blocked domain
        for url in urls_to_check:
            if url:
                domain = extract_domain(url)
                if domain in blocked_domains:
                    return True

        return False

    # Filter out blocked tasks
    original_count = len(tasks)
    if hasattr(tasks, 'apply'):  # pandas DataFrame
        filtered_tasks = tasks[~tasks.apply(is_task_blocked, axis=1)]
    else:  # list of dicts
        filtered_tasks = [task for task in tasks if not is_task_blocked(task)]

    filtered_count = len(filtered_tasks)
    blocked_count = original_count - filtered_count

    if blocked_count > 0:
        colorful_print(f"🔒 Filtered out {blocked_count} tasks from blocked domains ({original_count} -> {filtered_count})", fg='yellow')

    return filtered_tasks

# ───── Metrics Logging Functions (from log_iteration_metrics.py) ─────

class DummyAction:
    def __init__(self):
        self.action = {'key': 'none'}  # Default key that won't match 'invalid_url' or 'goback'
        self.action_string = ""  # Empty string for token counting

class DummyReward:
    def __init__(self):
        self.reward = 0  # Default reward of 0 (failure)

class DummyResponse:
    def __init__(self):
        self.raw_response = ""  # Empty string for token counting

def clean_trajectories(trajs):
    """Replace None values with dummy objects and filter out None trajectories"""
    cleaned_trajs = []

    for traj in trajs:
        # Skip None trajectories entirely
        if traj is None:
            continue

        cleaned_traj = []
        for step in traj:
            cleaned_step = step.copy() if step else {}

            # Replace None action with dummy
            if cleaned_step.get('action') is None:
                cleaned_step['action'] = DummyAction()

            # Replace None reward with dummy
            if cleaned_step.get('reward') is None:
                cleaned_step['reward'] = DummyReward()

            # Replace None response with dummy
            if cleaned_step.get('response') is None:
                cleaned_step['response'] = DummyResponse()

            cleaned_traj.append(cleaned_step)
        cleaned_trajs.append(cleaned_traj)

    return cleaned_trajs


def build_trajectory_metadata(config):
    """
    Build metadata dict to save alongside trajectories.

    Includes:
    - solution_model: The model used to generate actions
    - judge_model: The model used for evaluation (if evaluated)
    - dataset_file: Path to the task dataset file
    - split: 'train' or 'test'
    - server_size: Number of parallel rollout workers
    - max_steps: Maximum steps per trajectory

    Args:
        config: Hydra config object

    Returns:
        Dict with metadata fields
    """
    split = config.env_config.split
    dataset_file = config.env_config.train_tasks if split == 'train' else config.env_config.test_tasks

    metadata = {
        'solution_model': getattr(config.policy_config, 'base_model', None),
        'judge_model': getattr(config.openai_config, 'model', None),
        'dataset_file': dataset_file,
        'split': split,
        'server_size': getattr(config.env_config, 'server_size', None),
        'max_steps': getattr(config.env_config, 'max_steps', None),
    }

    # Add per-task judge overrides if present
    judge_overrides = {}
    for task_type in ['image_judgment', 'blocking_detection', 'criterion_a', 'criterion_b', 'reference_answer']:
        task_config = getattr(config.openai_config, task_type, None)
        if task_config and hasattr(task_config, 'model'):
            judge_overrides[task_type] = task_config.model
    if judge_overrides:
        metadata['judge_model_overrides'] = judge_overrides

    return metadata


def save_run_hyperparameters(config: DictConfig, save_path: str):
    """
    Save hyperparameters/config used for this run to the logs directory.

    Creates a hyperparameters.yaml file in the save_path directory containing
    the full resolved Hydra configuration.

    Args:
        config: Hydra DictConfig object with all configuration
        save_path: Base directory for logs (e.g., /data/logs)
    """
    import datetime
    import yaml

    # Create hyperparameters directory if it doesn't exist
    hyperparams_file = os.path.join(save_path, 'hyperparameters.yaml')

    # Convert OmegaConf to regular dict for YAML serialization
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Add run metadata
    run_info = {
        'run_timestamp': datetime.datetime.now().isoformat(),
        'config': config_dict
    }

    # Check if file exists to track multiple runs
    if os.path.exists(hyperparams_file):
        # Append to existing file with a separator
        with open(hyperparams_file, 'a') as f:
            f.write('\n---\n')  # YAML document separator
            yaml.dump(run_info, f, default_flow_style=False, sort_keys=False)
    else:
        # Create new file
        with open(hyperparams_file, 'w') as f:
            yaml.dump(run_info, f, default_flow_style=False, sort_keys=False)

    colorful_print(f"💾 Saved run hyperparameters to {hyperparams_file}", fg='green')


def is_crashed(t, is_train=True):
    """
    Detect crashed trajectories based on multiple criteria:
    1. Invalid URL (original crash detection)
    2. Abnormal structure (None values replaced with dummies)
    3. For both train and test: <10 steps AND last action != "answer"
    """
    # Check for empty trajectory (crashed before any steps)
    if not t or len(t) == 0:
        return True

    # Check for invalid URL (original condition)
    if t[0]['action'].action['key'] == 'invalid_url':
        return True

    # Check for abnormal structure (None values we replaced)
    for step in t:
        if step['action'].action['key'] == 'none':  # Our dummy action
            return True

    # Check trajectory length and last action
    min_steps = 10
    if len(t) < min_steps:
        # Check if last step is NOT an answer (successful or blocked)
        last_action_key = t[-1]['action'].action['key']
        is_blocked = t[-1]['reward'].is_blocked if hasattr(t[-1]['reward'], 'is_blocked') else False
        # Trajectory is crashed if it didn't answer and wasn't blocked
        if last_action_key != 'answer' and not is_blocked:
            return True

    return False

def calculate_rollout_metrics(save_path, split, chunk_size, is_train=True, existing_trajs=None):
    """Calculate metrics for the latest chunk of trajectories

    Args:
        existing_trajs: Optional pre-loaded trajectories to avoid reloading
    """
    if existing_trajs is None:
        # Load if not provided (for backward compatibility)
        from webgym.utils import load_all_trajectories

        traj_dir = Path(save_path) / f"{split}_trajectories"

        if not traj_dir.exists():
            print(f"Warning: {traj_dir} does not exist")
            return None

        trajs = load_all_trajectories(base_dir=save_path, split=split)
        print(f"Loaded {len(trajs)} trajectories from {traj_dir}")
    else:
        # Use pre-loaded trajectories
        trajs = existing_trajs
        print(f"Using pre-loaded {len(trajs)} trajectories for metrics")

    # Clean trajectories to replace None values
    trajs = clean_trajectories(trajs)

    # Handle case with no trajectories
    if len(trajs) == 0:
        print("No trajectories found - returning zero metrics")
        return {
            "success_rate": 0,
            "avg_chars_per_response": 0,
            "avg_steps": 0,
            "goback_rate": 0,
            "crashed_rate": 0,
            "blocked_rate": 0,
            "total_trajectories": 0
        }

    # Use only the last chunk_size trajectories (sliding window)
    latest_chunk = trajs[-chunk_size:] if len(trajs) > chunk_size else trajs
    total_trajectories = len(trajs)  # Save total count before windowing
    print(f"Using last {len(latest_chunk)} of {total_trajectories} total {split} trajectories for metrics (window size: {chunk_size})")

    tot = len(latest_chunk)

    # Calculate crashed rate using all trajectories
    crashed_count = sum(is_crashed(t, is_train) for t in latest_chunk)
    crashed_rate = crashed_count / tot

    # Filter out crashed trajectories for other metrics
    valid_trajs = [t for t in latest_chunk if not is_crashed(t, is_train)]

    if not valid_trajs:
        print("All trajectories crashed - using zeros")
        return {
            "success_rate": 0,
            "avg_chars_per_response": 0,
            "avg_steps": 0,
            "goback_rate": 0,
            "crashed_rate": crashed_rate,
            "blocked_rate": 0,
            "total_trajectories": total_trajectories
        }

    tot_non_crashed = len(valid_trajs)

    # Calculate blocked rate (% of non-crashed trajectories that were blocked)
    blocked_count = sum(
        t[-1]['reward'].is_blocked
        for t in valid_trajs
    )
    blocked_rate = blocked_count / tot_non_crashed

    # Filter out blocked trajectories for success rate and other metrics
    non_blocked_trajs = [
        t for t in valid_trajs
        if not t[-1]['reward'].is_blocked
    ]

    if not non_blocked_trajs:
        print("All non-crashed trajectories are blocked - using zeros for success metrics")
        return {
            "success_rate": 0,
            "avg_chars_per_response": 0,
            "avg_steps": 0,
            "goback_rate": 0,
            "crashed_rate": crashed_rate,
            "blocked_rate": blocked_rate,
            "total_trajectories": total_trajectories
        }

    tot_non_blocked = len(non_blocked_trajs)
    tot_steps = sum(len(t) for t in non_blocked_trajs)

    # Calculate success rate (from non-crashed, non-blocked trajectories)
    successes = sum(t[-1]['reward'].reward == 1 for t in non_blocked_trajs)
    success_rate = successes / tot_non_blocked

    # Calculate average characters per response using raw_response
    total_chars = sum(len(s['response'].raw_response) for t in non_blocked_trajs for s in t)
    avg_chars_per_response = total_chars / tot_steps if tot_steps > 0 else 0

    # Calculate average steps
    avg_steps = tot_steps / tot_non_blocked

    # Calculate goback rate
    goback_count = sum(s['action'].action['key'] == 'goback' for t in non_blocked_trajs for s in t)
    goback_rate = goback_count / tot_steps if tot_steps > 0 else 0

    return {
        "success_rate": success_rate,
        "avg_chars_per_response": avg_chars_per_response,
        "avg_steps": avg_steps,
        "goback_rate": goback_rate,
        "crashed_rate": crashed_rate,
        "blocked_rate": blocked_rate,
        "total_trajectories": total_trajectories
    }


def main_logic(config: DictConfig, current_rollout_step: int = 0, existing_trajectories_last=None):
    # Use completely random seed to ensure different sampling each iteration
    dynamic_seed = int(time.time() * 1000000) % (2**32)  # Use microsecond timestamp as seed
    colorful_print(f"🎲 Setting random seed to {dynamic_seed} (generated from timestamp)", fg='cyan')
    set_seed(dynamic_seed)

    # HuggingFace login is optional - only needed if downloading models from HF Hub
    # Skip gracefully if rate limited or token not available
    try:
        hf_token = os.environ.get(config.policy_config.huggingface_token_env_var)
        if hf_token:
            login(hf_token)
    except Exception as e:
        colorful_print(f"⚠️ HuggingFace login skipped: {e}", fg='yellow')

    # Save hyperparameters to logs directory at the start of each run
    save_run_hyperparameters(config, config.save_path)

    # Get vLLM server URL from config
    vllm_server_url = getattr(config.env_config, 'vllm_server_url', 'http://localhost:8000')

    # Get model_config from config
    model_config = dict(config.model_config) if hasattr(config, 'model_config') else {'model_type': 'qwen3-instruct'}

    # Add interaction_mode to model_config for convenience
    model_config['interaction_mode'] = config.context_config.get('interaction_mode', 'coordinates')

    # DEBUG: Print model configuration
    print("\n" + "="*100)
    print("🔍 DEBUG: Model Configuration in rollout.py")
    print("="*100)
    print(f"hasattr(config, 'model_config'): {hasattr(config, 'model_config')}")
    if hasattr(config, 'model_config'):
        print(f"config.model_config (raw): {config.model_config}")
    print(f"model_config (final): {model_config}")
    print("="*100 + "\n")

    agent = WebAgent(
        policy_config=config.policy_config,
        context_config=config.context_config,
        model_config=model_config,
        save_path=config.save_path,
        vllm_server_url=vllm_server_url,
        openai_config=config.openai_config,
        operation_timeout=getattr(config.env_config, 'operation_timeout', 120),
        vllm_timeout=getattr(config.env_config, 'vllm_timeout', 120),
        max_retries=getattr(config.env_config, 'max_retries', 1),
        max_vllm_sessions=int(getattr(config.env_config, 'max_vllm_sessions', 32))
    )

    # Use regular context manager for agent
    with agent:
        # Initialize blocklist manager
        blocklist_manager = BlocklistManager(config.save_path)
        blocked_domains = blocklist_manager.get_blocked_domains()

        if blocked_domains:
            colorful_print(f"🚫 Found {len(blocked_domains)} blocked domains: {list(blocked_domains)[:5]}{' ...' if len(blocked_domains) > 5 else ''}", fg='yellow')
        else:
            colorful_print("✅ No blocked domains found", fg='green')

        # Initialize rollout sampler
        rollout_sampler = RolloutSampler(config, config.save_path)

        # Load tasks based on split
        if config.env_config.split == "train":
            task_file_path = os.path.join(config.data_path, config.env_config.train_tasks)
            tasks = pd.read_json(task_file_path, lines=True)

            # Filter out blocked websites (ONLY for training)
            tasks = filter_blocked_tasks(tasks, blocked_domains)
            colorful_print(f"Loaded and filtered {len(tasks)} tasks from {config.env_config.train_tasks}", fg='green')
        else:
            task_file_path = os.path.join(config.data_path, config.env_config.test_tasks)
            tasks = pd.read_json(task_file_path, lines=True)

            # DO NOT filter blocked websites for test set - use all original tasks
            colorful_print(f"Loaded {len(tasks)} tasks from {config.env_config.test_tasks} (no blocking filter applied)", fg='green')

        # ========================
        # Resume Mode: Load checkpoint trajectories
        # ========================
        resume_checkpoint_path = getattr(config.env_config, 'resume_checkpoint_path', None)
        checkpoint_trajectories = []
        completed_task_keys = set()  # Set of (task_id, trajectory_index) tuples

        if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
            colorful_print(f"🔄 Resume mode: Loading checkpoint from {resume_checkpoint_path}", fg='cyan')
            checkpoint_data = torch.load(resume_checkpoint_path, map_location='cpu', weights_only=False)

            # Handle both old format (list) and new format (dict with trajectories/metadata)
            if isinstance(checkpoint_data, dict) and 'trajectories' in checkpoint_data:
                checkpoint_trajectories = checkpoint_data['trajectories']
            else:
                checkpoint_trajectories = checkpoint_data

            colorful_print(f"   Found {len(checkpoint_trajectories)} completed trajectories in checkpoint", fg='cyan')

            # Extract completed (task_id, trajectory_index) pairs
            for traj in checkpoint_trajectories:
                if traj and len(traj) > 0 and 'observation' in traj[0]:
                    task = traj[0]['observation'].task
                    task_id = str(task.task_id)
                    traj_idx = int(task.trajectory_index)
                    completed_task_keys.add((task_id, traj_idx))

            colorful_print(f"   Identified {len(completed_task_keys)} completed (task_id, trajectory_index) pairs", fg='cyan')

        # Load existing trajectories and compute cumulative distribution (for training only)
        # Support multinode mode with rank-specific trajectory files
        rank_suffix = getattr(config.env_config, 'multinode_rank_suffix', '')

        # Determine if this is a worker node (skip instance cleanup to avoid conflicts)
        # Only master node (rank 0 or single-node with no suffix) should cleanup instances
        is_worker_node = rank_suffix != '' and rank_suffix != '_rank_0'
        cumulative_distribution = None
        total_trajectories = 0

        if config.env_config.split == "train":
            # Train mode: uniform and ratio samplers don't need attempt level computation
            train_tasks_sampler = getattr(config.env_config, 'train_tasks_sampler', 'uniform')
            colorful_print(f"ℹ️  Using '{train_tasks_sampler}' sampler", fg='cyan')

        # Extract multinode rank and total workers from config (only for train mode)
        multinode_rank = None
        multinode_total_workers = None

        if config.env_config.split == 'train':
            # Train mode: use multinode_rank/multinode_total_workers for master coordination
            if hasattr(config.env_config, 'multinode_rank') and hasattr(config.env_config, 'multinode_total_workers'):
                multinode_rank = config.env_config.multinode_rank
                multinode_total_workers = config.env_config.multinode_total_workers
            elif rank_suffix:
                # Parse from rank_suffix (e.g., "_rank_0" -> rank 0)
                try:
                    multinode_rank = int(rank_suffix.replace('_rank_', ''))
                    # Try to get total workers from config
                    if hasattr(config.env_config, 'total_workers'):
                        multinode_total_workers = config.env_config.total_workers
                except (ValueError, AttributeError):
                    pass
        # Test mode: Each worker independently calculates its own slice using test_rank/test_total_workers
        # No master coordination needed for test (handled in _sample_test_tasks)

        # Sample tasks using the rollout sampler (handles both train sampling and test sequential)
        sampled_tasks = rollout_sampler.sample_tasks(
            tasks,
            config.env_config.split,
            cumulative_distribution=cumulative_distribution,
            total_trajectories=total_trajectories,
            multinode_rank=multinode_rank,
            multinode_total_workers=multinode_total_workers
        )

        print(f"This is {config.env_config.split} split. You sampled {len(sampled_tasks)} tasks.")

        # Sort tasks by difficulty (descending) to prioritize harder tasks in the queue
        # This ensures higher-difficulty tasks are submitted first, potentially improving training efficiency
        # IMPORTANT: Sort BEFORE assigning trajectory_index so indices match the checkpoint
        if 'difficulty' in sampled_tasks.columns:
            sampled_tasks = sampled_tasks.sort_values('difficulty', ascending=False).reset_index(drop=True)
            colorful_print(f"📊 Sorted {len(sampled_tasks)} tasks by difficulty (high to low) for queue prioritization", fg='cyan')

            # Show difficulty distribution
            if len(sampled_tasks) > 0:
                difficulty_counts = sampled_tasks['difficulty'].value_counts().sort_index(ascending=False)
                colorful_print(f"   Difficulty distribution (submission order):", fg='cyan')
                for diff, count in difficulty_counts.items():
                    colorful_print(f"   Difficulty {diff}: {count} tasks", fg='cyan')
        else:
            colorful_print(f"⚠️  'difficulty' column not found - skipping priority sorting", fg='yellow')

        # ========================
        # Resume Mode: Filter out already-completed tasks
        # ========================
        # NOTE: This must happen AFTER the difficulty sort because trajectory_index
        # is assigned based on position in the sorted list
        if completed_task_keys:
            # Add trajectory_index column (same as row position after sorting, which is how async_webgym assigns it)
            sampled_tasks = sampled_tasks.reset_index(drop=True)
            sampled_tasks['_trajectory_index'] = sampled_tasks.index

            # Create key column for filtering
            sampled_tasks['_task_key'] = list(zip(
                sampled_tasks['task_id'].astype(str),
                sampled_tasks['_trajectory_index'].astype(int)
            ))

            # Filter out completed tasks
            original_count = len(sampled_tasks)
            sampled_tasks = sampled_tasks[~sampled_tasks['_task_key'].isin(completed_task_keys)]
            sampled_tasks = sampled_tasks.drop(columns=['_task_key'])  # Remove temp column, keep _trajectory_index

            remaining_count = len(sampled_tasks)
            skipped_count = original_count - remaining_count

            colorful_print(f"🔄 Resume mode: Skipping {skipped_count} already-completed tasks", fg='cyan')
            colorful_print(f"   Remaining tasks to collect: {remaining_count}", fg='cyan')

            if remaining_count == 0:
                colorful_print(f"✅ All tasks already completed in checkpoint! Nothing to do.", fg='green')
                # Rename checkpoint to final iteration file
                checkpoint_path = getattr(config.env_config, 'resume_checkpoint_path', None)
                if checkpoint_path and os.path.exists(checkpoint_path) and checkpoint_path.endswith('.checkpoint'):
                    final_file = checkpoint_path[:-len('.checkpoint')]  # Remove .checkpoint suffix
                    os.rename(checkpoint_path, final_file)
                    colorful_print(f"📦 Renamed checkpoint to final: {final_file}", fg='green')
                # Return the checkpoint trajectories as the result
                return checkpoint_trajectories

        # Free memory from existing_trajectories after sampling is complete
        if 'existing_trajectories' in locals():
            import gc
            trajectory_count = len(existing_trajectories)
            del existing_trajectories
            gc.collect()
            colorful_print(f"🧹 Freed memory from {trajectory_count} existing trajectories after sampling", fg='green')

        # Create unified retry policy
        # Get timeout settings from config or use defaults
        operation_timeout = getattr(config.env_config, 'operation_timeout', 30.0)
        max_retries = getattr(config.env_config, 'max_retries', 3)

        colorful_print(f"🔧 Unified Timeout Configuration:", fg='cyan')
        colorful_print(f"  - Operation timeout: {operation_timeout}s", fg='green')
        colorful_print(f"  - Max retries: {max_retries}", fg='green')

        # Create unified retry policy
        retry_policy = {
            'wait_timeout': getattr(config.env_config, 'wait_timeout', 3600.0),
            'operation_timeout': getattr(config.env_config, 'operation_timeout', 30.0),
            'max_retries': getattr(config.env_config, 'max_retries', 3),
            'http_pools': getattr(config.env_config, 'http_pools', {
                # Default pool configuration if not specified
                'navigate': 64,
                'screenshot': 256,
                'ac_tree': 128,
                'metadata': 64,
                'page_metadata': 128,
                'execute': 128,
                'allocate': 4,
                'release': 4
            }),
            'max_vllm_sessions': getattr(config.env_config, 'max_vllm_sessions', 32)
        }

        # Check if rank-based proportional load is configured
        rank_load_weight = getattr(config.env_config, 'rank_load_weight', None)
        total_load_weight = getattr(config.env_config, 'total_load_weight', None)

        # CRITICAL FIX: In multinode mode, divide server_size by total workers to prevent
        # capacity exhaustion (each node should use 1/N of the instance pool)
        # OR use proportional load weights if configured
        actual_server_size = config.env_config.server_size

        # Determine total workers (different for train vs test mode)
        total_workers_for_division = None
        if multinode_total_workers and multinode_total_workers > 1:
            # Train mode
            total_workers_for_division = multinode_total_workers
        elif hasattr(config.env_config, 'test_total_workers') and config.env_config.test_total_workers and config.env_config.test_total_workers > 1:
            # Test mode
            total_workers_for_division = config.env_config.test_total_workers

        if total_workers_for_division and total_workers_for_division > 1:
            mode = "train" if multinode_total_workers else "test"

            if rank_load_weight is not None and total_load_weight is not None:
                # Proportional load distribution based on GPU counts
                # Calculate this rank's proportional share
                actual_server_size = max(1, int(config.env_config.server_size * rank_load_weight / total_load_weight))

                # Calculate proportional pool sizes
                original_pools = retry_policy['http_pools'].copy()
                proportional_pools = {}
                for pool_name, base_pool_size in original_pools.items():
                    if pool_name in ['allocate', 'release']:
                        # Small pools - use base of 8
                        base_size = 8
                    else:
                        base_size = base_pool_size
                    proportional_pools[pool_name] = max(0, int(base_size * rank_load_weight / total_load_weight))

                # Disable ac_tree for coordinates mode
                proportional_pools['ac_tree'] = 0

                retry_policy['http_pools'] = proportional_pools

                # Also divide max_vllm_sessions proportionally (each node processes fewer tasks)
                original_vllm_sessions = retry_policy['max_vllm_sessions']
                retry_policy['max_vllm_sessions'] = max(1, int(original_vllm_sessions * rank_load_weight / total_load_weight))

                colorful_print(f"⚙️  Multinode {mode} mode: Using proportional load distribution", fg='cyan')
                colorful_print(f"   Rank load weight: {rank_load_weight}/{total_load_weight}", fg='cyan')
                colorful_print(f"   Base server_size: {config.env_config.server_size}, This rank's server_size: {actual_server_size}", fg='cyan')
                colorful_print(f"   HTTP pools: {proportional_pools}", fg='cyan')
                colorful_print(f"   vLLM sessions: {original_vllm_sessions} → {retry_policy['max_vllm_sessions']}", fg='cyan')
            else:
                # Legacy equal division mode
                actual_server_size = max(1, config.env_config.server_size // total_workers_for_division)
                colorful_print(f"⚙️  Multinode {mode} mode: Dividing server_size by {total_workers_for_division} nodes", fg='cyan')
                colorful_print(f"   Original server_size: {config.env_config.server_size}, Per-node server_size: {actual_server_size}", fg='cyan')
                colorful_print(f"   Total concurrent instances across all nodes: {actual_server_size * total_workers_for_division}", fg='cyan')

                # Split HTTP pool sizes and max_vllm_sessions (each node processes fewer tasks)
                original_pools = retry_policy['http_pools'].copy()
                divided_pools = {
                    pool_name: max(1, pool_size // total_workers_for_division)
                    for pool_name, pool_size in original_pools.items()
                }
                retry_policy['http_pools'] = divided_pools

                # Also divide max_vllm_sessions (each node handles fewer tasks, needs fewer concurrent vLLM requests)
                original_vllm_sessions = retry_policy['max_vllm_sessions']
                retry_policy['max_vllm_sessions'] = max(1, original_vllm_sessions // total_workers_for_division)

                colorful_print(f"⚙️  Multinode {mode} mode: Dividing resources by {total_workers_for_division} nodes", fg='cyan')
                colorful_print(f"   Original server_size: {config.env_config.server_size}, Per-node server_size: {actual_server_size}", fg='cyan')
                colorful_print(f"   Total concurrent instances across all nodes: {actual_server_size * total_workers_for_division}", fg='cyan')
                colorful_print(f"   Original pools: {original_pools}", fg='cyan')
                colorful_print(f"   Per-node pools: {divided_pools}", fg='cyan')
                colorful_print(f"   vLLM sessions: {original_vllm_sessions} → {retry_policy['max_vllm_sessions']}", fg='cyan')

        # Create AsyncWebGym (now uses ProcessPools internally)
        # Worker nodes skip instance cleanup to avoid conflicts with master
        if is_worker_node:
            colorful_print(f"🔧 Worker node detected (rank_suffix={rank_suffix}), skipping instance cleanup", fg='yellow')

        env = AsyncWebGym(
            master_port=config.env_config.master_port,
            host_ip=config.env_config.host_ip,
            cpu_cluster_token=os.environ[config.env_config.cpu_cluster_token_env_var],
            sampled_tasks=sampled_tasks,
            save_path=config.save_path,
            num_workers=actual_server_size,  # Use divided server_size in multinode mode
            verbose=config.env_config.verbose,
            retry_policy=retry_policy,
            task_timeout_minutes=getattr(config.env_config, 'task_timeout_minutes', 20),
            completion_threshold=getattr(config.env_config, 'completion_threshold', 0.98),
            completion_grace_period=getattr(config.env_config, 'completion_grace_period', 120),
            blocklist_manager=blocklist_manager,
            skip_instance_cleanup=is_worker_node,  # Only master node cleans up instances
            multinode_rank_suffix=rank_suffix,  # Pass rank suffix for flag creation
            split=config.env_config.split,  # Pass split (train/test) for correct trajectory file naming
            env_config=config.env_config,  # Pass env_config for per-task max_steps
            interaction_mode=config.context_config.get('interaction_mode', 'coordinates')  # Pass interaction mode for AC tree optimization
        )

        # Track checkpoint state using task IDs to avoid duplicates WITHIN THIS ITERATION
        # With incremental format, each iteration saves to a NEW file (e.g., iteration9)
        # For resume mode, pre-populate with IDs from checkpoint trajectories
        initial_task_ids = set()

        if resume_checkpoint_path and checkpoint_trajectories:
            # Pre-populate with trajectory IDs from checkpoint
            for traj in checkpoint_trajectories:
                if traj and len(traj) > 0 and 'observation' in traj[0]:
                    try:
                        task = traj[0]['observation'].task
                        traj_idx = getattr(task, 'trajectory_index', None)
                        if traj_idx is not None:
                            initial_task_ids.add(f"traj_{traj_idx}")
                        else:
                            # Fallback to task_id
                            initial_task_ids.add(task.task_id if hasattr(task, 'task_id') else None)
                    except:
                        pass
            colorful_print(f"📊 Checkpoint state initialized with {len(initial_task_ids)} IDs from resume checkpoint", fg='cyan')
        else:
            colorful_print(f"📊 Checkpoint state initialized (empty) - will track saves within this iteration", fg='cyan')

        checkpoint_state = {'saved_task_ids': initial_task_ids}

        checkpoint_callback = None
        checkpoint_percentage = None
        # Build metadata for trajectory files (used by checkpoint and grace period saving)
        traj_metadata = build_trajectory_metadata(config)

        if getattr(config.env_config, 'save_traj_progress', False):
            save_every_percent = getattr(config.env_config, 'save_every_percent', 25)
            checkpoint_percentage = save_every_percent / 100.0
            colorful_print(f"💾 Checkpoint saving enabled: every {save_every_percent}%", fg='cyan')

            checkpoint_callback = create_fixed_checkpoint_callback(
                trajectory_file=None,  # Not used - using incremental format instead
                checkpoint_state=checkpoint_state,
                save_path=config.save_path,
                split=config.env_config.split,
                rank_suffix=rank_suffix,
                metadata=traj_metadata,
                resume_checkpoint_path=resume_checkpoint_path  # Pass for resume mode merging
            )
        else:
            colorful_print(f"💾 Checkpoint saving disabled", fg='yellow')

        # Run automation with checkpoint callback - Pass checkpoint_state
        grace_period_triggered = False
        try:
            new_trajectories = env.run_automation_with_fairness(
                agent,
                progress_callback=None,
                checkpoint_callback=checkpoint_callback,
                checkpoint_interval=checkpoint_percentage if checkpoint_percentage else 0.25,
                checkpoint_state=checkpoint_state,  # Pass the state to avoid duplicates
                traj_metadata=traj_metadata  # Pass config metadata for trajectory files
            )
        except GracePeriodExpiredException as e:
            print(f"⚠️ Grace period expired: {e}")
            print("📝 Trajectories already saved - proceeding to cleanup")
            grace_period_triggered = True
            new_trajectories = []  # Trajectories already saved by grace period handler

        # NEW CODE - Only save NEW trajectories since last checkpoint
        # Skip saving if grace period already handled it
        if not grace_period_triggered:
            print("=" * 80)
            print("💾 SAVING FINAL TRAJECTORIES")
            print(f"Got {len(new_trajectories) if new_trajectories else 0} trajectories")

            # Debug: Count valid trajectories
            valid_count = len([t for t in new_trajectories if t and len(t) > 0])
            empty_count = len([t for t in new_trajectories if t is not None and len(t) == 0])
            none_count = len([t for t in new_trajectories if t is None])
            print(f"   Valid (len>0): {valid_count}, Empty (len==0): {empty_count}, None: {none_count}")
            print("=" * 80)

            if checkpoint_callback is not None:
                # Checkpoint saving was enabled - only save NEW trajectories using task ID tracking
                valid_new = [t for t in new_trajectories if t is not None]
                saved_task_ids = checkpoint_state.get('saved_task_ids', set())

                # Find NEW trajectories (not already saved) using trajectory_index tracking
                new_trajectories_to_save = []
                for traj in valid_new:
                    if not traj or len(traj) == 0:
                        continue

                    # Get trajectory_index from trajectory (unique per task slot, distinguishes repeats)
                    try:
                        task = traj[0]['observation'].task
                        # Use trajectory_index if available (unique per slot), fall back to task_id for backward compatibility
                        traj_id = getattr(task, 'trajectory_index', None)
                        if traj_id is not None:
                            traj_id = f"traj_{traj_id}"  # Prefix to distinguish from legacy task_id
                        else:
                            # Fallback for old trajectories without trajectory_index
                            traj_id = task.task_id if hasattr(task, 'task_id') else None
                            if not traj_id:
                                task_name = task.task_name
                                traj_id = f"task_{task_name}" if task_name else None

                        # Only add if not already saved
                        if traj_id and traj_id not in saved_task_ids:
                            new_trajectories_to_save.append(traj)
                        elif not traj_id:
                            # Can't verify if saved, include it to be safe
                            new_trajectories_to_save.append(traj)
                    except Exception as e:
                        print(f"⚠️ Could not extract trajectory_index, saving anyway: {e}")
                        new_trajectories_to_save.append(traj)

                # Get checkpoint file path
                checkpoint_file = checkpoint_state.get('checkpoint_file')

                if len(new_trajectories_to_save) > 0:
                    colorful_print(f"📂 Merging {len(new_trajectories_to_save)} NEW trajectories with checkpoint...", fg='cyan')

                    # Load existing checkpoint and merge with new trajectories
                    from webgym.utils.trajectory_storage import _strip_page_metadata, _extract_keypoint_step_ids
                    import datetime

                    all_trajectories = []
                    if checkpoint_file and os.path.exists(checkpoint_file):
                        checkpoint_data = torch.load(checkpoint_file, weights_only=False)
                        if isinstance(checkpoint_data, dict) and 'trajectories' in checkpoint_data:
                            all_trajectories = checkpoint_data['trajectories']
                        else:
                            all_trajectories = checkpoint_data
                        colorful_print(f"   Loaded {len(all_trajectories)} trajectories from checkpoint", fg='cyan')

                    # Strip and process new trajectories
                    new_trajectories_to_save = _strip_page_metadata(new_trajectories_to_save)
                    new_trajectories_to_save = _extract_keypoint_step_ids(new_trajectories_to_save)

                    # Merge
                    all_trajectories.extend(new_trajectories_to_save)
                    colorful_print(f"   Total trajectories after merge: {len(all_trajectories)}", fg='cyan')

                    # Save merged trajectories to final iteration file
                    if checkpoint_file and checkpoint_file.endswith('.checkpoint'):
                        final_file = checkpoint_file[:-len('.checkpoint')]
                    else:
                        # Create new iteration file path
                        from webgym.utils.trajectory_storage import get_next_iteration_number
                        traj_dir = os.path.join(config.save_path, f'{config.env_config.split}_trajectories')
                        os.makedirs(traj_dir, exist_ok=True)
                        iteration_num = get_next_iteration_number(config.save_path, config.env_config.split)
                        final_file = os.path.join(traj_dir, f'{config.env_config.split}_trajectories.pt.iteration{iteration_num}{rank_suffix}')

                    # Build save data with metadata
                    if traj_metadata:
                        save_metadata = traj_metadata.copy()
                        if 'timestamp' not in save_metadata:
                            save_metadata['timestamp'] = datetime.datetime.now().isoformat()
                        save_data = {'trajectories': all_trajectories, 'metadata': save_metadata}
                    else:
                        save_data = all_trajectories

                    torch.save(save_data, final_file)
                    colorful_print(f"💾 Saved {len(all_trajectories)} trajectories to: {final_file}", fg='green')

                    # Delete checkpoint file if it exists and is different from final file
                    if checkpoint_file and os.path.exists(checkpoint_file) and checkpoint_file != final_file:
                        os.remove(checkpoint_file)
                        colorful_print(f"🗑️  Removed checkpoint file: {checkpoint_file}", fg='cyan')

                    # Free memory after saving
                    import gc
                    del new_trajectories_to_save
                    del all_trajectories
                    gc.collect()
                else:
                    colorful_print(f"✓ All trajectories already saved by checkpoints (no final save needed)", fg='green')

                    # Rename checkpoint file to final iteration file
                    if checkpoint_file and os.path.exists(checkpoint_file) and checkpoint_file.endswith('.checkpoint'):
                        final_file = checkpoint_file[:-len('.checkpoint')]  # Remove .checkpoint suffix
                        os.rename(checkpoint_file, final_file)
                        colorful_print(f"📦 Renamed checkpoint to final: {final_file}", fg='green')
            else:
                # No checkpoint saving - use incremental saving to avoid loading large trajectory files
                from webgym.utils import save_trajectories_incremental

                saved_path = save_trajectories_incremental(
                    trajectories=new_trajectories,
                    base_dir=config.save_path,
                    split=config.env_config.split,
                    rank_suffix=rank_suffix,
                    metadata=traj_metadata
                )
                colorful_print(f"💾 Saved {len(new_trajectories)} trajectories to: {saved_path}", fg='green')

                # Free memory after saving
                import gc
                del new_trajectories
                gc.collect()
        else:
            print("=" * 80)
            print("✅ Trajectories already saved by grace period handler - skipping final save")
            print("=" * 80)

        # Force exit INSIDE the with block to prevent agent.__exit__ from closing vllm_client
        # while zombie threads are still running. All results have been saved.
        print("🚪 Forcing program exit after successful save...")
        print("⚠️ Exiting before agent cleanup to avoid hanging on vllm_client.close()")
        os._exit(0)

@hydra.main(version_base=None, config_path="config/main", config_name="rollout")
def main(config: DictConfig):
    # Print configuration summary
    colorful_print("=== WEBGYM CONFIGURATION ===", fg='cyan')
    colorful_print(f"Model: {config.policy_config.base_model}", fg='green')
    colorful_print(f"Interaction Mode: {config.context_config.interaction_mode}", fg='green')
    colorful_print(f"Split: {config.env_config.split}", fg='green')

    # Show unified timeout configuration if available
    operation_timeout = getattr(config.env_config, 'operation_timeout', 30.0)
    colorful_print(f"Unified Timeout: {operation_timeout}s", fg='green')
    colorful_print("========================", fg='cyan')

    # Uniform and ratio samplers don't need trajectory history
    existing_trajectories_last = None
    if config.env_config.split == "train":
        train_tasks_sampler = getattr(config.env_config, 'train_tasks_sampler', 'uniform')
        print(f"ℹ️  Using '{train_tasks_sampler}' sampler - no trajectory history needed")

    # Run main logic directly (no asyncio)
    main_logic(config, 0, existing_trajectories_last)

if __name__ == "__main__":
    main()