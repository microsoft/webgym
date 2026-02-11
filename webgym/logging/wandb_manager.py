"""
WandB Management Module

Centralizes all WandB-related functionality including:
- Run detection and name matching
- Step synchronization
- Dynamic step management for monotonically increasing logs
"""

import wandb
import os
from typing import Optional, Tuple


def get_latest_wandb_run_info(project_name: str, entity_name: str, run_name: str) -> Tuple[Optional[str], int, bool]:
    """
    Enhanced version that checks if run_name matches latest run and handles new runs.

    Args:
        project_name: WandB project name
        entity_name: WandB entity name
        run_name: Requested run name

    Returns:
        Tuple of (run_id, max_step, is_new_run)
    """
    try:
        api = wandb.Api()

        # Get runs from the project, filtered by name
        runs = api.runs(f"{entity_name}/{project_name}", filters={"display_name": run_name})

        if not runs:
            print(f"No existing runs found with name '{run_name}' - will start new run")
            return None, 0, True

        # Get the most recent run with this specific name
        latest_run = runs[0]

        # Check if this run name matches what we want
        if latest_run.name != run_name:
            print(f"Latest run name '{latest_run.name}' != requested name '{run_name}' - will start new run")
            return None, 0, True

        # Get max step from the matching run's summary
        max_step = 0
        try:
            if hasattr(latest_run, 'summary') and latest_run.summary.get('global_step'):
                max_step = latest_run.summary.get('global_step', 0)

            # Try to check history for more accurate step count (only works for completed runs)
            try:
                if hasattr(latest_run, 'history'):
                    history = latest_run.history(samples=100)
                    if len(history) > 0:
                        step_columns = [col for col in history.columns if 'step' in col.lower() or col == '_step']
                        for col in step_columns:
                            try:
                                col_max = history[col].max()
                                if col_max is not None:
                                    max_step = max(max_step, int(col_max))
                            except:
                                continue
            except:
                # History access failed, continue with summary value
                pass
        except Exception as e:
            print(f"Warning: Could not get step info from run {latest_run.id}: {e}")

        print(f"Found matching run: {latest_run.id} (created: {latest_run.created_at}) with max_step: {max_step}")
        return latest_run.id, max_step, False

    except Exception as e:
        print(f"Error getting latest wandb run: {e}")
        return None, 0, True


def check_different_run_names(project_name: str, entity_name: str, current_run_name: str) -> list:
    """
    Check if there are any runs with different names (to inform user).

    Args:
        project_name: WandB project name
        entity_name: WandB entity name
        current_run_name: Current run name to exclude

    Returns:
        List of different run names found
    """
    try:
        api = wandb.Api()

        # Get all runs from the project (limited to recent ones)
        all_runs = api.runs(f"{entity_name}/{project_name}", per_page=20)

        if not all_runs:
            return []

        # Get unique run names that are different from current
        different_run_names = set()
        for run in all_runs:
            if run.name and run.name != current_run_name:
                different_run_names.add(run.name)

        return list(different_run_names)

    except Exception as e:
        print(f"Warning: Could not check for different run names: {e}")
        return []


def initialize_wandb_run(project_name: str, entity_name: str, run_name: str,
                        config: dict = None, wandb_key: str = None) -> Tuple[Optional[object], int]:
    """
    Initialize WandB run with proper resume handling and new run detection.

    Args:
        project_name: WandB project name
        entity_name: WandB entity name
        run_name: Requested run name
        config: Configuration dict to log
        wandb_key: WandB API key (optional)

    Returns:
        Tuple of (wandb_run, initial_step)
    """
    # Check for existing runs and different run names
    run_id, max_cloud_step, is_new_run = get_latest_wandb_run_info(
        project_name, entity_name, run_name
    )

    # Inform user about other existing runs if this is a new run
    if is_new_run:
        different_run_names = check_different_run_names(
            project_name, entity_name, run_name
        )
        if different_run_names:
            print(f"📝 Note: Found existing runs with different names: {different_run_names}")
            print(f"   Starting new run '{run_name}' as requested")

    # Set up WandB login
    if wandb_key:
        wandb.login(key=wandb_key)
    elif 'WANDB_API_KEY' in os.environ:
        wandb.login(key=os.environ['WANDB_API_KEY'])

    # Initialize WandB run
    if is_new_run:
        # Start completely new run
        wandb_run = wandb.init(
            project=project_name,
            entity=entity_name,
            name=run_name,
            config=config or {}
        )
        initial_step = 0  # Signal to start from step 1
        print(f"🆕 WandB new run started: '{run_name}' - will start logging from step 1")
    else:
        # Resume existing run
        wandb_run = wandb.init(
            project=project_name,
            entity=entity_name,
            name=run_name,
            id=run_id,
            resume="allow",
            config=config or {}
        )
        initial_step = max_cloud_step
        print(f"🔄 WandB resuming run: '{run_name}' (id: {run_id}) - cloud_max_step: {max_cloud_step}")

    return wandb_run, initial_step


class WandBStepManager:
    """
    Manages WandB step synchronization for monotonically increasing logs.

    This ensures that WandB steps are always increasing, even when:
    - Local training restarts
    - Process crashes and resumes
    - Switching between different experiments
    """

    def __init__(self, initial_step: int = 0):
        """
        Initialize the step manager.

        Args:
            initial_step: Starting step from cloud (0 for new runs, max_step for resume)
        """
        self.initial_step = initial_step
        self.current_step = None
        self.initialized = False

    def get_next_step(self, local_step: int = None, is_main_process: bool = True) -> int:
        """
        Get the next WandB step, ensuring monotonic increase.

        Args:
            local_step: Current local training step (for reference/debugging)
            is_main_process: Whether this is the main process (for logging)

        Returns:
            Next step to use for WandB logging
        """
        if not wandb.run:
            return 0

        try:
            if not self.initialized:
                # Check if this is a new run
                if self.initial_step == 0:
                    # New run - start from step 1
                    self.current_step = 1
                    self.initialized = True

                    if is_main_process:
                        print(f"🆕 WandB new run detected: Starting from step 1")
                        if local_step is not None:
                            print(f"   Local training step = {local_step} (unchanged for training logic)")

                    return self.current_step

                # Existing run - get current max step from cloud
                current_max_step = 0

                # Try to get from summary first (most reliable for active runs)
                if hasattr(wandb.run, 'summary') and wandb.run.summary.get('global_step'):
                    current_max_step = max(current_max_step, wandb.run.summary.get('global_step', 0))

                # For active runs, we rely on summary. History API only works on completed runs.
                # If summary doesn't have global_step, use initial_step as fallback
                if current_max_step == 0 and self.initial_step > 0:
                    current_max_step = self.initial_step

                # Start from max + 1
                self.current_step = current_max_step + 1
                self.initialized = True

                if is_main_process:
                    print(f"🔄 WandB step sync: Cloud max step = {current_max_step}, will start logging from step {self.current_step}")
                    if local_step is not None:
                        print(f"   Local training step = {local_step} (unchanged for training logic)")
            else:
                # Increment for subsequent logs
                self.current_step += 1

            return self.current_step

        except Exception as e:
            if is_main_process:
                print(f"Warning: WandB step detection failed: {e}, using fallback")
            # Fallback
            if self.current_step is None:
                self.current_step = max(1, self.initial_step + (local_step or 0))
            else:
                self.current_step += 1
            return self.current_step
