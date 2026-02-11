"""
Task History Manager for tracking task attempt history via attempt_level attribute.

Attempt Level System:
- 0: New task (never attempted)
- 1-4: Failed attempts (increments with each failure)
- 5: Solved task OR max failed attempts

The attempt_level is an attribute added to the tasks DataFrame for adaptive sampling weights.
"""

import os
import torch
import pandas as pd
import numpy as np
import hashlib
from typing import List, Optional, Dict
from webgym.misc import colorful_print


class TaskHistoryManager:
    """
    Manages attempt_level attribute for tasks based on attempt history.

    The attempt_level is stored directly in the task file as a new column.
    """

    def __init__(self, save_path: str):
        """
        Initialize TaskHistoryManager.

        Args:
            save_path: Directory containing task files and trajectories
        """
        self.save_path = save_path

    def _get_task_id(self, task):
        """
        Get task ID from task data.

        Uses the sequential task_id field from the task file (0, 1, 2, ..., N-1).
        This is much simpler and more reliable than hash-based IDs.

        Args:
            task: Task row (pd.Series) with task_id field, or Task object with task_id attribute

        Returns:
            Integer task ID
        """
        # For DataFrame rows (pd.Series)
        if isinstance(task, pd.Series):
            if 'task_id' in task:
                return int(task['task_id'])
            else:
                raise ValueError("Task row missing 'task_id' field")

        # For Task objects
        if hasattr(task, 'task_id'):
            return int(task.task_id)

        raise ValueError("Task object missing 'task_id' attribute")

    def compute_attempt_levels_from_trajectories(
        self,
        tasks: pd.DataFrame,
        trajectories: List
    ) -> pd.DataFrame:
        """
        Compute attempt levels for tasks based on trajectory history.

        Does NOT modify the task file - only computes in-memory.

        Logic:
        - Never attempted: level 0
        - Failed 1-4 times: level 1-4
        - Succeeded OR failed 5+ times: level 5

        Args:
            tasks: Task DataFrame
            trajectories: List of all trajectories (source of truth)

        Returns:
            Task DataFrame with attempt_level column computed from trajectories
        """
        # Build task history from trajectories
        task_history = {}  # task_id -> {'attempts': count, 'solved': bool}

        for traj in trajectories:
            if not traj or len(traj) == 0:
                continue

            # Get task from first step
            if hasattr(traj[0]['observation'], 'task'):
                task_obj = traj[0]['observation'].task

                # Get task ID directly from task object
                task_id = self._get_task_id(task_obj)

                # Skip filtered tasks (task_id=-1) or invalid task_id
                if task_id is None or task_id < 0:
                    continue

                # Get final reward
                final_reward = traj[-1]['reward'].reward if traj[-1]['reward'] is not None else 0

                # Update history
                if task_id not in task_history:
                    task_history[task_id] = {'attempts': 0, 'solved': False}

                task_history[task_id]['attempts'] += 1
                if final_reward == 1:
                    task_history[task_id]['solved'] = True

        # Compute attempt levels for each task
        tasks = tasks.copy()

        # Ensure task_id column exists (remember if we added it)
        task_id_existed = 'task_id' in tasks.columns
        if not task_id_existed:
            tasks['task_id'] = tasks.apply(self._get_task_id, axis=1)

        def compute_level(row):
            task_id = row['task_id']
            if task_id not in task_history:
                return 0  # Never attempted

            history = task_history[task_id]
            if history['solved']:
                return 5  # Solved
            else:
                # Failed attempts, cap at 5
                return min(history['attempts'], 5)

        tasks['attempt_level'] = tasks.apply(compute_level, axis=1)

        # Only drop task_id if we added it (keep it if it was in original data)
        if not task_id_existed:
            tasks = tasks.drop(columns=['task_id'])

        colorful_print(f"📋 Computed attempt levels from {len(trajectories)} trajectories", fg='cyan')
        self._log_level_distribution(tasks)

        return tasks

    def compute_cumulative_distribution_from_trajectories(
        self,
        tasks: pd.DataFrame,
        trajectories: List
    ) -> Dict[int, int]:
        """
        Compute cumulative distribution of attempt levels from trajectories.

        This computes what attempt level each trajectory's task CURRENTLY has
        based on all trajectory outcomes (not the stored attempt_level attribute).

        Args:
            tasks: Task DataFrame with computed attempt_level column
            trajectories: List of trajectories

        Returns:
            Dict mapping attempt_level -> count of trajectories
        """
        # Build task_id -> attempt_level mapping from tasks DataFrame
        # Ensure task_id column exists
        if 'task_id' not in tasks.columns:
            tasks_temp = tasks.copy()
            tasks_temp['task_id'] = tasks_temp.apply(self._get_task_id, axis=1)
        else:
            tasks_temp = tasks

        task_level_map = dict(zip(tasks_temp['task_id'], tasks_temp['attempt_level']))

        # Count trajectories by their task's current attempt level
        cumulative_distribution = {}
        for traj in trajectories:
            if not traj or len(traj) == 0:
                continue

            if hasattr(traj[0]['observation'], 'task'):
                task_obj = traj[0]['observation'].task

                # Get task ID directly from task object
                task_id = self._get_task_id(task_obj)

                # Skip filtered tasks (task_id=-1) or invalid task_id
                if task_id is None or task_id < 0:
                    continue

                # Get current attempt level for this task
                current_level = task_level_map.get(task_id, 0)
                cumulative_distribution[current_level] = cumulative_distribution.get(current_level, 0) + 1

        return cumulative_distribution

    def _log_level_distribution(self, tasks: pd.DataFrame):
        """Log current distribution of tasks across attempt levels."""
        if 'attempt_level' not in tasks.columns:
            return

        distribution = tasks['attempt_level'].value_counts().to_dict()
        total = len(tasks)

        colorful_print(f"📊 Level distribution (total: {total} tasks):", fg='cyan')
        for pos in range(6):
            count = distribution.get(pos, 0)
            percentage = (count / total * 100) if total > 0 else 0

            if pos == 0:
                label = "New (never attempted)"
            elif pos < 5:
                label = f"Failed {pos} time(s)"
            else:
                label = "Solved or max failures"

            colorful_print(f"  Position {pos} ({label}): {count} ({percentage:.1f}%)", fg='cyan')

    def apply_adaptive_weighted_sampling(
        self,
        tasks: pd.DataFrame,
        level_weights: Optional[Dict[int, float]] = None,
        cumulative_distribution: Optional[Dict[int, int]] = None,
        total_trajectories: int = 0
    ) -> pd.DataFrame:
        """
        Apply adaptive sampling weights based on task attempt history.

        This superimposes the attempt-level distribution over the existing task distribution
        (which already reflects uniform-over-websites sampling).

        Default weights (exponentially decreasing):
        - Position 0 (new): 10.0 (highest priority)
        - Position 1: 5.0
        - Position 2: 2.5
        - Position 3: 1.0
        - Position 4: 0.5
        - Position 5 (solved/max): 0.1 (lowest priority)

        Args:
            tasks: DataFrame of tasks with attempt_level column
            level_weights: Optional custom weights for each attempt level
            cumulative_distribution: Cumulative count of trajectories per attempt level
            total_trajectories: Total number of trajectories collected so far

        Returns:
            Re-weighted sampled tasks DataFrame (same size as input)
        """
        if len(tasks) == 0:
            return tasks

        # Ensure attempt_level exists
        if 'attempt_level' not in tasks.columns:
            tasks['attempt_level'] = 0

        # Default weights: exponentially decreasing (more drastic)
        if level_weights is None:
            level_weights = {
                0: 100.0,  # New tasks: VERY high priority
                1: 10.0,   # 1 failure: high priority
                2: 1.0,    # 2 failures: normal priority
                3: 0.1,    # 3 failures: low priority
                4: 0.01,   # 4 failures: very low priority
                5: 0.001   # Solved or max failures: extremely low priority
            }

        # Calculate weights for each task
        weights = tasks['attempt_level'].map(lambda pos: level_weights.get(pos, 1.0))

        # Normalize weights to probabilities
        total_weight = weights.sum()
        if total_weight == 0:
            colorful_print("⚠️ All level weights are 0, falling back to uniform sampling", fg='yellow')
            return tasks

        probabilities = weights / total_weight

        # Sample tasks based on probabilities (same number as input)
        try:
            sampled_indices = np.random.choice(
                len(tasks),
                size=len(tasks),
                replace=False,
                p=probabilities.values
            )
            sampled_tasks = tasks.iloc[sampled_indices].reset_index(drop=True)

            # Log sampling statistics
            self._log_sampling_stats(tasks, sampled_tasks, level_weights, cumulative_distribution, total_trajectories)

            return sampled_tasks

        except Exception as e:
            colorful_print(f"⚠️ Adaptive-weighted sampling failed: {e}, returning original tasks", fg='yellow')
            return tasks

    def _log_sampling_stats(
        self,
        original_tasks: pd.DataFrame,
        sampled_tasks: pd.DataFrame,
        level_weights: Dict[int, float],
        cumulative_distribution: Optional[Dict[int, int]] = None,
        total_trajectories: int = 0
    ):
        """Log statistics about adaptive-weighted sampling."""
        colorful_print(f"🎯 Adaptive-weighted sampling applied to {len(sampled_tasks)} tasks:", fg='cyan')

        # Distribution before and after (for current iteration)
        for pos in range(6):
            before_count = (original_tasks['attempt_level'] == pos).sum()
            after_count = (sampled_tasks['attempt_level'] == pos).sum()
            before_pct = (before_count / len(original_tasks) * 100) if len(original_tasks) > 0 else 0
            after_pct = (after_count / len(sampled_tasks) * 100) if len(sampled_tasks) > 0 else 0
            weight = level_weights.get(pos, 1.0)

            # If cumulative stats provided, show cumulative percentage
            if cumulative_distribution is not None:
                cumulative_count = cumulative_distribution.get(pos, 0) + after_count
                cumulative_total = total_trajectories + len(sampled_tasks)
                cumulative_pct = (cumulative_count / cumulative_total * 100) if cumulative_total > 0 else 0
                colorful_print(
                    f"  Pos {pos}: {before_count}→{after_count} "
                    f"({before_pct:.1f}%→{after_pct:.1f}%) [weight={weight}] "
                    f"| Cumulative: {cumulative_count}/{cumulative_total} ({cumulative_pct:.1f}%)",
                    fg='cyan'
                )
            else:
                colorful_print(
                    f"  Pos {pos}: {before_count}→{after_count} "
                    f"({before_pct:.1f}%→{after_pct:.1f}%) [weight={weight}]",
                    fg='cyan'
                )
