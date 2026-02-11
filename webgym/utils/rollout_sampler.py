"""
Rollout task sampling.
Handles task selection strategies for both training and testing (sequential).
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, Optional
from webgym.misc import colorful_print


class RolloutSampler:
    """
    Handles task sampling for rollout.
    """

    def __init__(self, config, save_path: str):
        """
        Initialize the rollout sampler with website correction support.

        Args:
            config: Hydra config object with env_config settings
            save_path: Path to save/load trajectory files
        """
        self.config = config
        self.save_path = save_path
        self.env_config = config.env_config

        # Get deduplicate_across_iterations flag (default to False for backward compatibility)
        self.deduplicate_across_iterations = getattr(self.env_config, 'deduplicate_across_iterations', False)

        # Initialize task history manager
        from webgym.utils import TaskHistoryManager
        self.task_history_manager = TaskHistoryManager(save_path)

    def _get_previously_sampled_tasks(self, split: str) -> set:
        """
        Extract task names from all previous iteration trajectories.

        Args:
            split: Either "train" or "test"

        Returns:
            Set of task names that have been sampled in previous iterations
        """
        if not self.deduplicate_across_iterations:
            return set()

        traj_dir = os.path.join(self.save_path, f"{split}_trajectories")

        if not os.path.exists(traj_dir):
            colorful_print(f"📂 No existing trajectories found at {traj_dir}, starting fresh", fg='cyan')
            return set()

        # Load all trajectory files
        trajectory_files = sorted([
            f for f in os.listdir(traj_dir)
            if f.startswith(f'{split}_trajectories.pt.iteration')
        ])

        if not trajectory_files:
            colorful_print(f"📂 No trajectory files found in {traj_dir}, starting fresh", fg='cyan')
            return set()

        colorful_print(f"🔍 Loading previously sampled tasks from {len(trajectory_files)} iteration files...", fg='cyan')

        previously_sampled = set()
        for traj_file in trajectory_files:
            filepath = os.path.join(traj_dir, traj_file)
            try:
                data = torch.load(filepath, weights_only=False)

                # Data is a dict with 'trajectories' and 'metadata' keys
                trajectories = data.get('trajectories', data) if isinstance(data, dict) else data

                # Extract task names from trajectories
                for trajectory in trajectories:
                    if isinstance(trajectory, list) and len(trajectory) > 0:
                        step = trajectory[0]
                        if isinstance(step, dict) and 'observation' in step:
                            obs = step['observation']
                            if hasattr(obs, 'task') and hasattr(obs.task, 'task_name'):
                                task_name = obs.task.task_name
                                previously_sampled.add(task_name)
            except Exception as e:
                colorful_print(f"⚠️  Error loading {traj_file}: {e}", fg='yellow')
                continue

        colorful_print(f"✓ Found {len(previously_sampled)} unique tasks across all previous iterations", fg='green')
        return previously_sampled

    def sample_tasks(
        self,
        tasks: pd.DataFrame,
        split: str,
        cumulative_distribution: Optional[Dict[int, int]] = None,
        total_trajectories: int = 0,
        multinode_rank: Optional[int] = None,
        multinode_total_workers: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Sample tasks based on split type, filtering out broken websites.

        IMPORTANT: URL corrections are already baked into the task file before this method is called.
        This method only needs to filter out 'does_not_work' websites.

        For TRAIN split:
            - Filters out known broken websites AT SAMPLING TIME (lazy filtering)
            - URL corrections already applied to task file (no correction needed here)
            - Supports uniform or ratio-based sampling strategies
            - Maintains subdomain distribution
            - Can resample different tasks from previously sampled websites

        For TEST split:
            - NO filtering of broken websites (preserve original test set)
            - URL corrections already applied to task file
            - Simple sequential sampling (first N tasks)
            - NO randomization
            - Ensures consistent evaluation

        Args:
            tasks: Full task dataset (with corrections already baked in)
            split: Either "train" or "test"
            cumulative_distribution: Cumulative count of trajectories per attempt level
            total_trajectories: Total number of trajectories collected so far
            multinode_rank: Rank in multinode setup (0 = master, >0 = worker)
            multinode_total_workers: Total number of workers in multinode setup

        Returns:
            Sampled tasks DataFrame
        """
        if split == "train":
            # Get sampling strategy from config (default to "uniform")
            train_tasks_sampler = getattr(self.env_config, 'train_tasks_sampler', 'uniform')

            if train_tasks_sampler == "uniform":
                # Uniform random sampling
                colorful_print("📚 Using uniform random sampling for training", fg='cyan')
                # Check if multinode mode
                if multinode_rank is not None and multinode_total_workers is not None and multinode_total_workers > 1:
                    # Multinode mode: divide tasks across workers
                    return self._sample_train_tasks_vanilla(tasks, multinode_rank, multinode_total_workers)
                else:
                    # Single node mode
                    return self._sample_train_tasks_vanilla(tasks)
            elif train_tasks_sampler == "ratio":
                # Difficulty ratio-based sampling
                colorful_print("📊 Using difficulty ratio-based sampling for training", fg='cyan')
                # Check if multinode mode
                if multinode_rank is not None and multinode_total_workers is not None and multinode_total_workers > 1:
                    # Multinode mode: divide tasks across workers
                    return self._sample_train_tasks_by_difficulty_ratio(tasks, multinode_rank, multinode_total_workers)
                else:
                    # Single node mode
                    return self._sample_train_tasks_by_difficulty_ratio(tasks)
            else:
                raise ValueError(f"Unknown train_tasks_sampler: {train_tasks_sampler}. Must be 'uniform' or 'ratio'.")
        else:
            colorful_print("📋 Using sequential sampling for testing (preserving original test set)", fg='cyan')
            colorful_print("   No URL filtering or correction at sampling time for test set", fg='cyan')
            return self._sample_test_tasks(tasks)

    def _sample_unique_tasks(
        self,
        tasks: pd.DataFrame,
        n: int,
        weights: np.ndarray = None,
        context: str = ""
    ) -> pd.DataFrame:
        """
        Sample exactly n unique tasks, ensuring no duplicates.

        If weights are provided, uses weighted sampling without replacement.
        If not enough unique tasks exist, raises an error.

        Args:
            tasks: Tasks DataFrame to sample from
            n: Number of unique tasks to sample
            weights: Optional sampling weights (must match tasks length after reset_index)
            context: Context string for logging

        Returns:
            Sampled tasks DataFrame with exactly n unique tasks
        """
        if len(tasks) == 0:
            return pd.DataFrame()

        # Check if we have enough unique tasks
        if 'task_name' in tasks.columns:
            unique_tasks = tasks.drop_duplicates(
                subset=['subdomain', 'website', 'difficulty', 'task_name']
            )
            num_unique = len(unique_tasks)
        else:
            unique_tasks = tasks
            num_unique = len(tasks)

        if num_unique < n:
            colorful_print(
                f"⚠️  {context}: Only {num_unique} unique tasks available, but {n} requested. "
                f"Using all {num_unique} unique tasks.",
                fg='red'
            )
            return unique_tasks.copy()

        # Sample without replacement to guarantee uniqueness
        if weights is not None:
            # Weighted sampling without replacement using numpy
            # Reset index for consistent indexing
            tasks_reset = tasks.reset_index(drop=True)

            # Normalize weights
            weights_sum = weights.sum()
            if weights_sum == 0:
                # Fallback to uniform if all weights are zero
                sampled = tasks_reset.sample(n=n, replace=False)
            else:
                probs = weights / weights_sum

                try:
                    # Use numpy choice for weighted sampling without replacement
                    sampled_indices = np.random.choice(
                        len(tasks_reset),
                        size=n,
                        replace=False,
                        p=probs
                    )
                    sampled = tasks_reset.iloc[sampled_indices]
                except Exception as e:
                    colorful_print(
                        f"⚠️  {context}: Weighted sampling failed: {e}, using uniform sampling",
                        fg='yellow'
                    )
                    sampled = tasks_reset.sample(n=n, replace=False)
        else:
            # Uniform sampling without replacement
            sampled = tasks.sample(n=n, replace=False)

        return sampled

    def _validate_and_correct_task_batch(self, tasks: pd.DataFrame, subdomain: str = None) -> pd.DataFrame:
        """
        Validate tasks batch (no-op since task files are already filtered).

        Args:
            tasks: Task DataFrame to validate
            subdomain: Optional subdomain name for progress bar description

        Returns:
            Task DataFrame (unmodified)
        """
        # Task files are already permanently filtered, so just return as-is
        return tasks.copy()

    def _sample_test_tasks(self, tasks: pd.DataFrame) -> pd.DataFrame:
        """
        Simple sequential sampling for test tasks.

        IMPORTANT: Always picks the first N tasks in order to ensure:
        - Consistent evaluation across runs
        - No randomization
        - Reproducible test results

        In multinode mode, slices the test set across workers:
        - Each worker gets an equal slice of the test set
        - Worker 0: tasks[0:256], Worker 1: tasks[256:512], etc.

        Each task is repeated test_tasks_repeat_times times (if > 0) to enable
        multiple evaluation runs per task for computing variance/confidence.

        If test_tasks_rollout_size is -1 or 0, uses ALL tasks instead of limiting.
        """
        test_size = self.env_config.test_tasks_rollout_size
        test_repeat_times = getattr(self.env_config, 'test_tasks_repeat_times', 0)

        # Use all tasks if test_size is -1 or 0
        use_all_tasks = (test_size <= 0)
        if use_all_tasks:
            test_size = len(tasks)
            colorful_print(f"✓ Using ALL {test_size} tasks (test_tasks_rollout_size={self.env_config.test_tasks_rollout_size})", fg='cyan')

        # Check for multinode test slicing parameters
        test_rank = getattr(self.env_config, 'test_rank', None)
        test_total_workers = getattr(self.env_config, 'test_total_workers', None)

        if test_rank is not None and test_total_workers is not None and test_total_workers > 1:
            # Check if proportional load weights are configured
            rank_load_weight = getattr(self.env_config, 'rank_load_weight', None)
            total_load_weight = getattr(self.env_config, 'total_load_weight', None)

            if rank_load_weight is not None and total_load_weight is not None:
                # Proportional round-robin distribution
                # Parse all rank weights from config (comma-separated string like "1,1" or "2,1,1")
                all_rank_weights_str = getattr(self.env_config, 'all_rank_weights', None)
                if all_rank_weights_str:
                    weights_list = [int(w) for w in all_rank_weights_str.split(',')]
                    rank_weights = {i: weights_list[i] for i in range(len(weights_list))}
                else:
                    # Fallback: assume equal weights if not provided
                    rank_weights = {i: 1 for i in range(test_total_workers)}
                total_weight = sum(rank_weights.values())

                # Each "round" distributes total_weight tasks: 2 to rank0, 1 to rank1, 1 to rank2
                # Rank 0: [0,1], [4,5], [8,9], ...
                # Rank 1: [2], [6], [10], ...
                # Rank 2: [3], [7], [11], ...

                rank_indices = []
                idx = 0

                # Calculate starting offset for this rank within each round
                offset = 0
                for r in range(test_rank):
                    offset += rank_weights.get(r, 1)

                my_weight = rank_weights.get(test_rank, 1)

                # Calculate how many tasks this rank should collect (proportional share)
                expected_count = (test_size * my_weight + total_weight - 1) // total_weight  # Ceiling division

                # Collect indices in round-robin fashion
                round_num = 0
                while idx < expected_count and round_num * total_weight < test_size:
                    # Calculate base index for this round
                    base_idx = round_num * total_weight + offset

                    # Collect my_weight consecutive tasks starting from base_idx
                    for i in range(my_weight):
                        task_idx = base_idx + i
                        if task_idx < test_size:
                            rank_indices.append(task_idx)
                            idx += 1
                        else:
                            break

                    round_num += 1

                sampled = tasks.iloc[rank_indices]

                # Show example indices (first few rounds)
                example_indices = rank_indices[:min(12, len(rank_indices))]
                colorful_print(
                    f"✓ Multinode test: Proportional round-robin - rank {test_rank} (weight {my_weight}/{total_weight}) "
                    f"gets {len(sampled)}/{expected_count} tasks (first indices: {example_indices}...)",
                    fg='green'
                )
            else:
                # Legacy round-robin distribution
                # This ensures balanced difficulty distribution even if tasks are grouped by difficulty
                # Rank 0 gets indices [0, 2, 4, ...], Rank 1 gets [1, 3, 5, ...], etc.
                rank_indices = list(range(test_rank, test_size, test_total_workers))
                sampled = tasks.iloc[rank_indices]

                colorful_print(
                    f"✓ Multinode test: Round-robin sampling - rank {test_rank} gets indices "
                    f"[{test_rank}, {test_rank+test_total_workers}, {test_rank+2*test_total_workers}, ...] "
                    f"({len(sampled)} unique tasks for rank {test_rank}/{test_total_workers})",
                    fg='green'
                )
        else:
            # Single node mode: take first N tasks (or all if test_size was adjusted)
            sampled = tasks.iloc[:test_size]
            if use_all_tasks:
                colorful_print(f"✓ Selected ALL {len(sampled)} test tasks (sequential)", fg='green')
            else:
                colorful_print(f"✓ Selected first {len(sampled)} unique test tasks (sequential)", fg='green')

        # Repeat each task test_tasks_repeat_times times (if specified)
        if test_repeat_times > 0:
            # Create a list to hold repeated tasks
            repeated_tasks = []
            for _ in range(test_repeat_times):
                repeated_tasks.append(sampled.copy())

            # Concatenate all repetitions
            sampled = pd.concat(repeated_tasks, ignore_index=True)

            colorful_print(
                f"✓ Repeated each task {test_repeat_times} times: {len(sampled)} total samples "
                f"({len(sampled) // test_repeat_times} unique tasks × {test_repeat_times} repeats)",
                fg='cyan'
            )

        return sampled

    def _sample_train_tasks_vanilla(
        self,
        tasks: pd.DataFrame,
        multinode_rank: Optional[int] = None,
        multinode_total_workers: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Simple uniform random sampling for training tasks.

        No website diversity weighting, no attempt-level weighting.
        Just uniformly sample train_tasks_rollout_size tasks from the full train set.

        In multinode mode, the total train_tasks_rollout_size is divided across all workers.
        Each worker samples a different random subset to ensure no overlap.

        Args:
            tasks: Full task dataset
            multinode_rank: Rank in multinode setup (0 = master, >0 = worker)
            multinode_total_workers: Total number of workers in multinode setup

        Returns:
            Randomly sampled tasks DataFrame
        """
        # Filter out previously sampled tasks if deduplication is enabled
        if self.deduplicate_across_iterations:
            previously_sampled = self._get_previously_sampled_tasks('train')
            if previously_sampled:
                original_count = len(tasks)
                tasks = tasks[~tasks['task_name'].isin(previously_sampled)].copy()
                filtered_count = original_count - len(tasks)
                colorful_print(
                    f"🔄 Deduplication: Filtered out {filtered_count} previously sampled tasks ({original_count} -> {len(tasks)})",
                    fg='cyan'
                )

        total_train_size = self.env_config.train_tasks_rollout_size

        # Calculate per-node sample size in multinode mode
        if multinode_rank is not None and multinode_total_workers is not None and multinode_total_workers > 1:
            # Check if proportional load weights are configured
            rank_load_weight = getattr(self.env_config, 'rank_load_weight', None)
            total_load_weight = getattr(self.env_config, 'total_load_weight', None)

            if rank_load_weight is not None and total_load_weight is not None:
                # Proportional division based on GPU counts
                train_size = int(total_train_size * rank_load_weight / total_load_weight)
                colorful_print(
                    f"📦 Multinode proportional sampling: Rank {multinode_rank} (weight {rank_load_weight}/{total_load_weight}) "
                    f"sampling {train_size}/{total_train_size} tasks",
                    fg='cyan'
                )
            else:
                # Legacy equal division
                tasks_per_worker = total_train_size // multinode_total_workers

                # Last worker gets any remainder
                if multinode_rank == multinode_total_workers - 1:
                    train_size = total_train_size - (tasks_per_worker * (multinode_total_workers - 1))
                else:
                    train_size = tasks_per_worker

                colorful_print(
                    f"📦 Multinode uniform sampling: Rank {multinode_rank}/{multinode_total_workers} "
                    f"sampling {train_size}/{total_train_size} tasks",
                    fg='cyan'
                )
        else:
            # Single node: sample all tasks
            train_size = total_train_size

        # Set random seed based on rank to ensure different samples per node
        # This prevents duplicate sampling across nodes
        if multinode_rank is not None:
            import numpy as np
            # Use rank to offset the random state so each node gets different samples
            # But still deterministic within the same iteration
            random_state = None  # Use current global random state (already seeded with timestamp in rollout.py)
        else:
            random_state = None

        # Sample unique tasks without replacement
        sampled = self._sample_unique_tasks(
            tasks,
            n=train_size,
            weights=None,
            context=f"Uniform sampling (rank {multinode_rank})" if multinode_rank is not None else "Uniform sampling"
        )

        # Double-check for duplicates (safety check)
        if 'task_name' in sampled.columns:
            original_count = len(sampled)
            sampled_dedup = sampled.drop_duplicates(
                subset=['subdomain', 'website', 'difficulty', 'task_name'],
                keep='first'
            )
            if len(sampled_dedup) < original_count:
                colorful_print(
                    f"⚠️  Found {original_count - len(sampled_dedup)} duplicates in uniform sampling, using {len(sampled_dedup)} unique tasks",
                    fg='yellow'
                )
                sampled = sampled_dedup

        # Print basic statistics
        colorful_print(f"📊 Uniform Sampling Statistics:", fg='cyan')
        colorful_print(f"  Total tasks in train set: {len(tasks):,}", fg='cyan')
        colorful_print(f"  Sampled tasks for this rank: {len(sampled):,}", fg='cyan')
        if multinode_rank is not None and multinode_total_workers is not None and multinode_total_workers > 1:
            colorful_print(f"  Total across all {multinode_total_workers} ranks: {total_train_size:,}", fg='cyan')

        # Show subdomain distribution
        if 'subdomain' in sampled.columns:
            subdomain_counts = sampled['subdomain'].value_counts()
            colorful_print(f"\n📊 Subdomain Distribution (sampled):", fg='cyan')
            for subdomain, count in subdomain_counts.items():
                percentage = count / len(sampled) * 100
                colorful_print(f"  {subdomain}: {count} ({percentage:.1f}%)", fg='cyan')

        # Show difficulty distribution
        if 'difficulty' in sampled.columns:
            difficulty_counts = sampled['difficulty'].value_counts().sort_index()
            colorful_print(f"\n📊 Difficulty Distribution (sampled):", fg='cyan')
            for difficulty, count in difficulty_counts.items():
                percentage = count / len(sampled) * 100
                colorful_print(f"  Difficulty {difficulty}: {count} ({percentage:.1f}%)", fg='cyan')

        # Create sync flag for workers in multinode mode (master only)
        if multinode_rank == 0 and multinode_total_workers is not None and multinode_total_workers > 1:
            sampling_complete_flag = os.path.join(self.save_path, 'multinode_flags/train_sampling_complete_rank_0')
            os.makedirs(os.path.dirname(sampling_complete_flag), exist_ok=True)
            with open(sampling_complete_flag, 'w') as f:
                f.write('')
            colorful_print(f"🚩 Created train_sampling_complete flag for workers (vanilla mode)", fg='green')

        return sampled

    def _sample_train_tasks_by_difficulty_ratio(
        self,
        tasks: pd.DataFrame,
        multinode_rank: Optional[int] = None,
        multinode_total_workers: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Sample training tasks based on difficulty ratio weights.

        Uses difficulty_ratio from config as a map of difficulty ranges to weights.
        Example: difficulty_ratio = {
            'easy': {'range': [1, 3], 'weight': 0.2},
            'medium': {'range': [4, 6], 'weight': 0.5},
            'hard': {'range': [7, 100], 'weight': 0.3}
        }

        In multinode mode, the total train_tasks_rollout_size is divided across all workers.
        Each worker samples a different random subset to ensure no overlap.

        Args:
            tasks: Full task dataset
            multinode_rank: Rank in multinode setup (0 = master, >0 = worker)
            multinode_total_workers: Total number of workers in multinode setup

        Returns:
            Sampled tasks DataFrame based on difficulty ratios
        """
        # Get difficulty_ratio from config
        difficulty_ratio = getattr(self.env_config, 'difficulty_ratio', None)
        if difficulty_ratio is None or len(difficulty_ratio) == 0:
            colorful_print("⚠️  No difficulty_ratio specified in config, falling back to uniform sampling", fg='yellow')
            return self._sample_train_tasks_vanilla(tasks, multinode_rank, multinode_total_workers)

        # Note: Task deduplication is disabled for ratio sampling to allow resampling
        # This ensures we can always fill the target sample size even after many iterations
        colorful_print("ℹ️  Ratio sampling allows task resampling across iterations", fg='cyan')

        # Simple random sampling without diversity weighting
        colorful_print("🎲 Using uniform random sampling within each difficulty range", fg='cyan')

        total_train_size = self.env_config.train_tasks_rollout_size

        # Calculate per-node sample size in multinode mode
        if multinode_rank is not None and multinode_total_workers is not None and multinode_total_workers > 1:
            # Check if proportional load weights are configured
            rank_load_weight = getattr(self.env_config, 'rank_load_weight', None)
            total_load_weight = getattr(self.env_config, 'total_load_weight', None)

            if rank_load_weight is not None and total_load_weight is not None:
                # Proportional division based on GPU counts
                train_size = int(total_train_size * rank_load_weight / total_load_weight)
                colorful_print(
                    f"📦 Multinode proportional ratio sampling: Rank {multinode_rank} (weight {rank_load_weight}/{total_load_weight}) "
                    f"sampling {train_size}/{total_train_size} tasks",
                    fg='cyan'
                )
            else:
                # Legacy equal division
                tasks_per_worker = total_train_size // multinode_total_workers

                # Last worker gets any remainder
                if multinode_rank == multinode_total_workers - 1:
                    train_size = total_train_size - (tasks_per_worker * (multinode_total_workers - 1))
                else:
                    train_size = tasks_per_worker

                colorful_print(
                    f"📦 Multinode ratio sampling: Rank {multinode_rank}/{multinode_total_workers} "
                    f"sampling {train_size}/{total_train_size} tasks",
                    fg='cyan'
                )
        else:
            # Single node: sample all tasks
            train_size = total_train_size

        # Set random seed based on rank to ensure different samples per node
        if multinode_rank is not None:
            import numpy as np
            random_state = None  # Use current global random state (already seeded with timestamp in rollout.py)
        else:
            random_state = None

        # Parse difficulty_ratio map format
        # Expected format: {'easy': {'range': [1, 3], 'weight': 0.2}, ...}
        difficulty_ranges = []
        total_weight = 0

        for category_name, config in difficulty_ratio.items():
            range_vals = config.get('range', [])
            weight = config.get('weight', 0)

            if len(range_vals) != 2:
                colorful_print(f"⚠️  Invalid range format for '{category_name}': {range_vals}, skipping", fg='yellow')
                continue

            difficulty_ranges.append({
                'name': category_name,
                'min': range_vals[0],
                'max': range_vals[1],
                'weight': weight
            })
            total_weight += weight

        if total_weight == 0:
            colorful_print("⚠️  All difficulty_ratio weights are 0, falling back to uniform sampling", fg='yellow')
            return self._sample_train_tasks_vanilla(tasks, multinode_rank, multinode_total_workers)

        # Calculate target samples per difficulty range
        range_sample_counts = []
        for range_config in difficulty_ranges:
            if range_config['weight'] > 0:
                target_count = int(train_size * range_config['weight'] / total_weight)
                range_sample_counts.append({
                    **range_config,
                    'target_count': target_count
                })

        # Distribute remainder samples to ranges with highest fractional parts
        allocated_samples = sum(r['target_count'] for r in range_sample_counts)
        remainder = train_size - allocated_samples
        if remainder > 0:
            fractional_parts = []
            for range_config in difficulty_ranges:
                if range_config['weight'] > 0:
                    fractional_part = (train_size * range_config['weight'] / total_weight) - int(train_size * range_config['weight'] / total_weight)
                    fractional_parts.append((range_config['name'], fractional_part))
            # Sort by fractional part (descending) and add 1 sample to top ranges
            fractional_parts.sort(key=lambda x: x[1], reverse=True)
            for i in range(remainder):
                range_name = fractional_parts[i][0]
                # Find and increment the target count
                for r in range_sample_counts:
                    if r['name'] == range_name:
                        r['target_count'] += 1
                        break

        # Print target distribution
        colorful_print(f"📊 Target Difficulty Distribution:", fg='cyan')
        for range_config in range_sample_counts:
            count = range_config['target_count']
            percentage = count / train_size * 100
            colorful_print(
                f"  {range_config['name']} [{range_config['min']}-{range_config['max']}]: {count} samples ({percentage:.1f}%), weight={range_config['weight']}",
                fg='cyan'
            )

        # Sample tasks from each difficulty range
        sampled_tasks = []
        for range_config in range_sample_counts:
            if range_config['target_count'] == 0:
                continue

            # Get tasks within this difficulty range
            min_diff = range_config['min']
            max_diff = range_config['max']
            range_tasks = tasks[(tasks['difficulty'] >= min_diff) & (tasks['difficulty'] <= max_diff)]

            if len(range_tasks) == 0:
                colorful_print(
                    f"❌ No tasks found for {range_config['name']} [{min_diff}-{max_diff}], skipping",
                    fg='red'
                )
                continue

            # Uniform random sampling within this difficulty range
            sampled = self._sample_unique_tasks(
                range_tasks,
                n=range_config['target_count'],
                weights=None,  # No diversity weighting, pure random sampling
                context=f"{range_config['name']} [{min_diff}-{max_diff}]"
            )

            sampled_tasks.append(sampled)

        # Combine all sampled tasks
        if len(sampled_tasks) == 0:
            colorful_print("❌ No tasks sampled, falling back to uniform sampling", fg='red')
            return self._sample_train_tasks_vanilla(tasks, multinode_rank, multinode_total_workers)

        sampled = pd.concat(sampled_tasks, ignore_index=True)

        # Shuffle the combined samples to mix difficulties
        sampled = sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Double-check for duplicates (safety check)
        if 'task_name' in sampled.columns:
            original_count = len(sampled)
            sampled_dedup = sampled.drop_duplicates(
                subset=['subdomain', 'website', 'difficulty', 'task_name'],
                keep='first'
            )
            if len(sampled_dedup) < original_count:
                colorful_print(
                    f"⚠️  Found {original_count - len(sampled_dedup)} duplicates in ratio sampling, using {len(sampled_dedup)} unique tasks",
                    fg='yellow'
                )
                sampled = sampled_dedup

        # Print basic statistics
        colorful_print(f"\n📊 Ratio Sampling Statistics:", fg='cyan')
        colorful_print(f"  Total tasks in train set: {len(tasks):,}", fg='cyan')
        colorful_print(f"  Sampled tasks for this rank: {len(sampled):,}", fg='cyan')
        if multinode_rank is not None and multinode_total_workers is not None and multinode_total_workers > 1:
            colorful_print(f"  Total across all {multinode_total_workers} ranks: {total_train_size:,}", fg='cyan')

        # Show subdomain distribution
        if 'subdomain' in sampled.columns:
            subdomain_counts = sampled['subdomain'].value_counts()
            colorful_print(f"\n📊 Subdomain Distribution (sampled):", fg='cyan')
            for subdomain, count in subdomain_counts.items():
                percentage = count / len(sampled) * 100
                colorful_print(f"  {subdomain}: {count} ({percentage:.1f}%)", fg='cyan')

        # Show actual difficulty distribution
        if 'difficulty' in sampled.columns:
            difficulty_counts = sampled['difficulty'].value_counts().sort_index()
            colorful_print(f"\n📊 Actual Difficulty Distribution (sampled):", fg='cyan')
            for difficulty, count in difficulty_counts.items():
                percentage = count / len(sampled) * 100
                colorful_print(f"  Difficulty {difficulty}: {count} ({percentage:.1f}%)", fg='cyan')

        # Create sync flag for workers in multinode mode (master only)
        if multinode_rank == 0 and multinode_total_workers is not None and multinode_total_workers > 1:
            sampling_complete_flag = os.path.join(self.save_path, 'multinode_flags/train_sampling_complete_rank_0')
            os.makedirs(os.path.dirname(sampling_complete_flag), exist_ok=True)
            with open(sampling_complete_flag, 'w') as f:
                f.write('')
            colorful_print(f"🚩 Created train_sampling_complete flag for workers (ratio mode)", fg='green')

        return sampled


    def _compute_uniform_over_websites_weights(
        self,
        candidate_tasks: pd.DataFrame
    ) -> np.ndarray:
        """
        Compute sampling weights for uniform-over-websites strategy.

        UPDATED: Increased website diversity impact to balance with attempt-level weights.

        Algorithm:
        1. Each unique website gets base weight = 1.0
        2. Within each website, tasks share the weight uniformly
        3. Task weight = 1.0 / num_tasks_in_website
        4. Weights are normalized to [0.1, 10.0] range for stronger diversity preference

        This gives strong preference to websites with fewer tasks while still
        maintaining attempt-level as the primary driver.

        Args:
            candidate_tasks: Tasks after difficulty filtering

        Returns:
            Array of weights (same length as candidate_tasks)
        """
        if len(candidate_tasks) == 0:
            return np.array([])

        # Reset index to ensure we have 0-based indexing for weights array
        candidate_tasks_reset = candidate_tasks.reset_index(drop=True)

        # Group tasks by website
        website_groups = candidate_tasks_reset.groupby('website')
        num_websites = len(website_groups)

        if num_websites == 0:
            return np.array([])

        # Calculate weight for each task
        weights = np.zeros(len(candidate_tasks_reset))

        for website, group_indices in website_groups.groups.items():
            num_tasks_in_website = len(group_indices)
            # Each website gets weight 1.0 divided by its task count
            task_weight = 1.0 / num_tasks_in_website
            weights[group_indices] = task_weight

        # Normalize weights to [0.1, 10.0] range for stronger impact
        # This gives website diversity a 100x range (vs previous 3x range)
        if weights.max() > 0:
            # Normalize to [0, 1]
            weights_normalized = weights / weights.max()
            # Scale to [0.1, 10.0] to increase variance and website diversity impact
            weights = 0.1 + (weights_normalized * 9.9)

        return weights

    def _compute_attempt_level_weights(
        self,
        candidate_tasks: pd.DataFrame,
        level_weights: Optional[Dict[int, float]] = None
    ) -> np.ndarray:
        """
        Compute sampling weights based on attempt_level.

        Default weights (exponentially decreasing):
        - attempt_level 0 (new): 100.0
        - attempt_level 1: 10.0
        - attempt_level 2: 1.0
        - attempt_level 3: 0.1
        - attempt_level 4: 0.01
        - attempt_level 5 (solved/max): 0.001

        Args:
            candidate_tasks: Tasks after difficulty filtering
            level_weights: Optional custom weights for each attempt level

        Returns:
            Array of weights (same length as candidate_tasks)
        """
        if len(candidate_tasks) == 0:
            return np.array([])

        # Reset index to ensure consistency with website weights
        candidate_tasks_reset = candidate_tasks.reset_index(drop=True)

        # Ensure attempt_level exists
        if 'attempt_level' not in candidate_tasks_reset.columns:
            return np.ones(len(candidate_tasks_reset))

        # Default weights: exponentially decreasing
        if level_weights is None:
            level_weights = {
                0: 100.0,  # New tasks: VERY high priority
                1: 10.0,   # 1 failure: high priority
                2: 1.0,    # 2 failures: normal priority
                3: 0.1,    # 3 failures: low priority
                4: 0.01,   # 4 failures: very low priority
                5: 0.001   # Solved or max failures: extremely low priority
            }

        # Map attempt_level to weights
        weights = candidate_tasks_reset['attempt_level'].map(
            lambda level: level_weights.get(level, 1.0)
        ).values

        return weights

    def _sample_with_combined_weights(
        self,
        candidate_tasks: pd.DataFrame,
        sample_count: int,
        subdomain: str
    ) -> pd.DataFrame:
        """
        Sample tasks using combined weights from website diversity and attempt-level strategies.

        UPDATED: Website diversity now has stronger effect (range [0.1, 10.0])
        while attempt level maintains high impact (range [0.001, 100.0]).
        Both factors significantly influence sampling.

        Algorithm:
        1. Compute website diversity weights (strong preference, range [0.1, 10.0])
        2. Compute attempt_level weights (strong preference, range [0.001, 100.0])
        3. Multiply weights together: combined_weight = website_weight × attempt_weight
        4. Sample based on: probability_i = combined_weight_i / sum(all combined_weights)

        Effect: Both attempt_level (100,000x range) and website diversity (100x range)
        strongly influence sampling, with attempt_level still being the primary driver.

        Args:
            candidate_tasks: Tasks after difficulty filtering
            sample_count: Number of tasks to sample
            subdomain: Subdomain name (for logging)

        Returns:
            Sampled tasks DataFrame
        """
        if len(candidate_tasks) == 0:
            return pd.DataFrame()

        # Step 1: Compute uniform-over-websites weights
        website_weights = self._compute_uniform_over_websites_weights(candidate_tasks)

        # Step 2: Compute attempt_level weights
        attempt_weights = self._compute_attempt_level_weights(candidate_tasks)

        # Step 3: Combine weights (multiplicative)
        combined_weights = website_weights * attempt_weights

        # Step 4: Normalize to probabilities
        total_weight = combined_weights.sum()
        if total_weight == 0:
            colorful_print(f"⚠️ {subdomain}: All combined weights are 0, using uniform sampling", fg='yellow')
            # Fallback to uniform sampling
            if len(candidate_tasks) >= sample_count:
                return candidate_tasks.sample(n=sample_count, replace=False)
            else:
                return candidate_tasks.sample(n=sample_count, replace=True)

        probabilities = combined_weights / total_weight

        # Step 5: Sample based on combined probabilities using unique sampling
        sampled_tasks = self._sample_unique_tasks(
            candidate_tasks,
            n=sample_count,
            weights=probabilities,
            context=subdomain
        )

        # Log sampling statistics
        colorful_print(
            f"  {subdomain}: Sampled {len(sampled_tasks)} unique tasks using combined weights "
            f"(website[0.1-10.0] × attempt[0.001-100])",
            fg='cyan'
        )

        return sampled_tasks

    def _sample_uniform_over_websites(
        self,
        candidate_tasks: pd.DataFrame,
        sample_count: int,
        subdomain: str
    ) -> pd.DataFrame:
        """
        Sample tasks uniformly over websites, not tasks.

        Algorithm:
        1. Each unique website gets equal probability
        2. Within each website, tasks are uniformly sampled

        Args:
            candidate_tasks: Tasks after difficulty filtering
            sample_count: Number of tasks to sample
            subdomain: Subdomain name (for logging)

        Returns:
            Sampled tasks DataFrame
        """
        if len(candidate_tasks) == 0:
            return pd.DataFrame()

        # Group tasks by website
        website_groups = candidate_tasks.groupby('website')
        unique_websites = list(website_groups.groups.keys())
        num_websites = len(unique_websites)

        if num_websites == 0:
            return pd.DataFrame()

        # Calculate samples per website (uniform distribution)
        base_samples_per_website = sample_count // num_websites
        remainder = sample_count % num_websites

        sampled_tasks_list = []
        websites_sampled = 0
        replacement_count = 0
        replacement_details = []

        # Randomly shuffle websites to fairly distribute remainder
        import random
        random.shuffle(unique_websites)

        for idx, website in enumerate(unique_websites):
            website_tasks = website_groups.get_group(website)

            # Distribute remainder to first few websites
            num_samples_for_website = base_samples_per_website + (1 if idx < remainder else 0)

            if num_samples_for_website > 0:
                # Sample unique tasks within this website
                sampled = self._sample_unique_tasks(
                    website_tasks,
                    n=num_samples_for_website,
                    weights=None,
                    context=f"Website {website[:50]}"
                )

                # Track if we got fewer tasks than requested
                if len(sampled) < num_samples_for_website:
                    replacement_count += 1
                    duplicates_created = num_samples_for_website - len(sampled)
                    replacement_details.append({
                        'website': website[:50],
                        'available': len(website_tasks),
                        'needed': num_samples_for_website,
                        'shortfall': duplicates_created
                    })

                sampled_tasks_list.append(sampled)
                websites_sampled += 1

        # Combine all sampled tasks
        if sampled_tasks_list:
            result = pd.concat(sampled_tasks_list, ignore_index=False)  # Keep original indices

            # Check for duplicates within this subdomain sampling
            if 'task_name' in result.columns:
                unique_in_subdomain = result.drop_duplicates(subset=['website', 'difficulty', 'task_name']).shape[0]
                if unique_in_subdomain != len(result):
                    duplicates_in_subdomain = len(result) - unique_in_subdomain
                    colorful_print(
                        f"  {subdomain}: Sampled {len(result)} tasks ({unique_in_subdomain} unique, {duplicates_in_subdomain} duplicates) "
                        f"from {num_websites} websites (~{base_samples_per_website} tasks/website)",
                        fg='yellow'
                    )
                    if replacement_count > 0:
                        total_shortfall = sum(d['shortfall'] for d in replacement_details)
                        colorful_print(
                            f"    ⚠️  {replacement_count} websites had insufficient unique tasks ({total_shortfall} tasks short)",
                            fg='yellow'
                        )
                        # Show top 5 websites with shortfalls
                        top_replacement = sorted(replacement_details, key=lambda x: x['shortfall'], reverse=True)[:5]
                        for detail in top_replacement:
                            colorful_print(
                                f"      {detail['website']}: {detail['available']} tasks available, needed {detail['needed']} ({detail['shortfall']} short)",
                                fg='yellow'
                            )
                else:
                    colorful_print(
                        f"  {subdomain}: Sampled {len(result)} unique tasks from {num_websites} websites "
                        f"(~{base_samples_per_website} tasks/website)",
                        fg='cyan'
                    )
            else:
                colorful_print(
                    f"  {subdomain}: Sampled {len(result)} tasks from {num_websites} websites "
                    f"(~{base_samples_per_website} tasks/website)",
                    fg='cyan'
                )
            return result
        else:
            return pd.DataFrame()

