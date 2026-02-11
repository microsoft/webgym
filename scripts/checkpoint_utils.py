"""
Checkpoint system for webgym/environment/async_webgym.py

This module provides checkpoint callback and grace period code.
Uses incremental trajectory format with checkpoint files that get merged across ranks.

Workflow:
1. Checkpoint (50%): Save trajectories to temp .checkpoint file, track task IDs
2. Grace period (98%): Error incomplete tasks, append NEW trajectories to checkpoint file
3. Post-rollout (run.sh): Merge all rank checkpoint files into one iteration file

This ensures:
- Partial progress saved at checkpoints (recoverable if crash)
- No duplicate trajectories (task ID tracking)
- One final iteration file per rollout after merging ranks
"""

import os


class GracePeriodExpiredException(Exception):
    """
    Exception raised when grace period expires and trajectories have been saved.

    This allows proper cleanup (wandb logging, file renaming) instead of using os._exit(0)
    which would kill the process immediately without cleanup.
    """
    pass


def create_fixed_checkpoint_callback(trajectory_file, checkpoint_state, save_path=None, split=None, rank_suffix='', metadata=None, resume_checkpoint_path=None):
    """
    Checkpoint callback that saves trajectories to temp file and tracks task IDs.

    Args:
        trajectory_file: Unused (kept for API compatibility)
        checkpoint_state: Dict to track checkpoint state (must contain 'saved_task_ids' set and 'checkpoint_file')
        save_path: Base directory for incremental saving (e.g., /data/logs) - REQUIRED
        split: 'train' or 'test' for incremental saving - REQUIRED
        rank_suffix: Multinode rank suffix (e.g., '_rank_0') for incremental saving
        metadata: Optional dict with config metadata (solution_model, judge_model, dataset_file, etc.)
        resume_checkpoint_path: Optional path to existing checkpoint to resume from

    Returns:
        Checkpoint callback function
    """
    # Store resume checkpoint path for merging
    _resume_checkpoint_path = resume_checkpoint_path

    def checkpoint_save(completed_count, total_tasks, trajectories):
        """Save trajectories to checkpoint file and track task IDs"""
        percentage = int((completed_count / total_tasks) * 100)

        # Filter out None trajectories
        valid_trajectories = [t for t in trajectories if t is not None]

        # Get set of task IDs already saved
        saved_task_ids = checkpoint_state.get('saved_task_ids', set())

        # Find NEW trajectories by checking trajectory_index (unique per task slot, distinguishes repeats)
        new_trajectories = []
        new_task_ids = set()

        for traj in valid_trajectories:
            if not traj or len(traj) == 0:
                continue

            # Get trajectory_index from trajectory (unique per task slot, even for repeated tasks)
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
                    new_trajectories.append(traj)
                    new_task_ids.add(traj_id)
            except Exception as e:
                # If we can't get trajectory_index, save it to be safe
                print(f"⚠️ Could not extract trajectory_index from trajectory, saving anyway: {e}")
                new_trajectories.append(traj)

        if len(new_trajectories) == 0:
            print(f"💾 Checkpoint {percentage}%: No new trajectories since last checkpoint")
            return

        # Save to checkpoint file (temp file that will be extended at grace period)
        import torch
        import os
        from pathlib import Path
        from webgym.utils.trajectory_storage import get_next_iteration_number, _strip_page_metadata

        traj_dir = os.path.join(save_path, f'{split}_trajectories')
        os.makedirs(traj_dir, exist_ok=True)

        # Determine checkpoint filepath - reuse existing path if available, or use resume path, or create new
        is_new_checkpoint = False  # Track if we're creating a brand new checkpoint (not reusing/resuming)
        if checkpoint_state.get('checkpoint_file') and os.path.exists(checkpoint_state['checkpoint_file']):
            # Reuse the same checkpoint file from previous checkpoint in this run
            checkpoint_filepath = checkpoint_state['checkpoint_file']
            import re
            match = re.search(r'\.iteration(\d+)', checkpoint_filepath)
            iteration_num = int(match.group(1)) if match else get_next_iteration_number(save_path, split)
        elif _resume_checkpoint_path and os.path.exists(_resume_checkpoint_path):
            # Use resume checkpoint path
            checkpoint_filepath = _resume_checkpoint_path
            import re
            match = re.search(r'\.iteration(\d+)', checkpoint_filepath)
            iteration_num = int(match.group(1)) if match else get_next_iteration_number(save_path, split)
        else:
            # Create new checkpoint filename (first checkpoint in this run)
            is_new_checkpoint = True
            iteration_num = get_next_iteration_number(save_path, split)
            checkpoint_filename = f'{split}_trajectories.pt.iteration{iteration_num}{rank_suffix}.checkpoint'
            checkpoint_filepath = os.path.join(traj_dir, checkpoint_filename)

            # If a stale checkpoint file exists from a previous crashed run, remove it
            # to prevent cross-run trajectory contamination
            if os.path.exists(checkpoint_filepath):
                print(f"⚠️ Found stale checkpoint from previous crashed run: {checkpoint_filepath}")
                print(f"   Removing stale checkpoint to prevent cross-run contamination")
                os.remove(checkpoint_filepath)

        # Strip unnecessary page_metadata before saving
        from webgym.utils.trajectory_storage import _extract_keypoint_step_ids
        new_trajectories = _strip_page_metadata(new_trajectories)
        new_trajectories = _extract_keypoint_step_ids(new_trajectories)

        # Load existing checkpoint trajectories if file exists (for accumulation across checkpoints within the same run)
        all_trajectories = []
        existing_metadata = {}
        if os.path.exists(checkpoint_filepath):
            try:
                existing_data = torch.load(checkpoint_filepath, weights_only=False)
                if isinstance(existing_data, dict) and 'trajectories' in existing_data:
                    all_trajectories = existing_data['trajectories']
                    existing_metadata = existing_data.get('metadata', {})
                else:
                    all_trajectories = existing_data
                print(f"   Loaded {len(all_trajectories)} existing trajectories from checkpoint for merging")

                # CRITICAL: Populate saved_task_ids from existing trajectories to prevent duplicates
                for existing_traj in all_trajectories:
                    if existing_traj and len(existing_traj) > 0:
                        try:
                            existing_task = existing_traj[0]['observation'].task
                            existing_traj_id = getattr(existing_task, 'trajectory_index', None)
                            if existing_traj_id is not None:
                                saved_task_ids.add(f"traj_{existing_traj_id}")
                            else:
                                # Fallback for old trajectories
                                existing_task_id = getattr(existing_task, 'task_id', None)
                                if existing_task_id:
                                    saved_task_ids.add(existing_task_id)
                        except Exception:
                            pass
                print(f"   Populated saved_task_ids with {len(saved_task_ids)} existing trajectory IDs")
            except Exception as e:
                print(f"⚠️ Could not load existing checkpoint for merging: {e}")
                all_trajectories = []

        # Merge with new trajectories
        all_trajectories.extend(new_trajectories)

        # Save checkpoint with metadata if provided
        if metadata:
            import datetime
            save_metadata = metadata.copy() if metadata else existing_metadata.copy()
            if 'timestamp' not in save_metadata:
                save_metadata['timestamp'] = datetime.datetime.now().isoformat()
            if 'iteration' not in save_metadata:
                save_metadata['iteration'] = iteration_num
            save_data = {'trajectories': all_trajectories, 'metadata': save_metadata}
        else:
            save_data = all_trajectories

        torch.save(save_data, checkpoint_filepath)

        # Store checkpoint file path for grace period handler
        checkpoint_state['checkpoint_file'] = checkpoint_filepath

        # Update saved task IDs
        saved_task_ids.update(new_task_ids)
        checkpoint_state['saved_task_ids'] = saved_task_ids

        print(
            f"💾 Checkpoint {percentage}%: Saved {len(new_trajectories)} NEW trajectories to {checkpoint_filepath} "
            f"(total in file: {len(all_trajectories)}, total tracked: {len(saved_task_ids)})"
        )

    return checkpoint_save


def handle_grace_period_expiry_fixed(
    grace_period_expired,
    completed_count,
    total_tasks,
    future_to_task,
    trajectories,
    tasks,
    checkpoint_state,
    trajectory_file,
    _create_dummy_trajectory_with_error,
    save_path=None,
    split=None,
    rank_suffix='',
    metadata=None
):
    """
    Grace period handler that saves ALL trajectories to one iteration file.

    Args:
        grace_period_expired: Boolean flag
        completed_count: Number of completed tasks
        total_tasks: Total number of tasks
        future_to_task: Dict mapping futures to task info
        trajectories: List of all trajectories
        tasks: List of task info dicts
        checkpoint_state: Unused (kept for API compatibility)
        trajectory_file: Unused (kept for API compatibility)
        _create_dummy_trajectory_with_error: Function to create error trajectories
        save_path: Base directory for incremental saving (e.g., /data/logs) - REQUIRED
        split: 'train' or 'test' for incremental saving - REQUIRED
        rank_suffix: Multinode rank suffix (e.g., '_rank_0') for incremental saving
        metadata: Optional dict with config metadata (solution_model, judge_model, dataset_file, etc.)
    """

    if not grace_period_expired:
        return

    print("=" * 80)
    print("💾 GRACE PERIOD EXPIRED - SAVING ALL TASKS (INCLUDING INCOMPLETE)")
    print(f"Completed: {completed_count}/{total_tasks}")
    print("=" * 80)

    # Determine which tasks are complete vs incomplete
    completed_task_indices = {
        future_to_task[f]['index'] for f in future_to_task
        if f.done()
    }

    # Get set of task IDs already saved at checkpoint
    saved_task_ids = checkpoint_state.get('saved_task_ids', set()) if checkpoint_state else set()
    checkpoint_file = checkpoint_state.get('checkpoint_file', None) if checkpoint_state else None

    print(f"Tasks already saved at checkpoint: {len(saved_task_ids)}")
    print(f"Tasks completed now: {len(completed_task_indices)}")
    print(f"Tasks to process: {total_tasks}")

    # Mark ALL incomplete tasks with error trajectories
    incomplete_count = 0
    for task_idx in range(total_tasks):
        # If task didn't complete yet and has no trajectory, create error trajectory
        if task_idx not in completed_task_indices and trajectories[task_idx] is None:
            task_info = tasks[task_idx]
            error_type = "GRACE_PERIOD_INCOMPLETE"
            detailed_error = f"[{error_type}] Task did not complete before grace period expired at {(completed_count/total_tasks)*100:.0f}% threshold"

            trajectories[task_idx] = _create_dummy_trajectory_with_error(
                task_info['task'], task_info['output_dir'], error_type, detailed_error
            )
            incomplete_count += 1

    if incomplete_count > 0:
        print(f"⚠️ Marked {incomplete_count} incomplete tasks with error trajectories")

    # Find NEW trajectories (not already saved at checkpoint)
    all_valid = [t for t in trajectories if t is not None]
    new_trajectories = []
    new_task_ids = set()
    skipped_count = 0

    for traj in all_valid:
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

            # Only add if not already saved at checkpoint
            if traj_id:
                if traj_id not in saved_task_ids:
                    new_trajectories.append(traj)
                    new_task_ids.add(traj_id)
                else:
                    skipped_count += 1
            else:
                # Can't verify if saved, include it to be safe
                new_trajectories.append(traj)
        except Exception as e:
            print(f"⚠️ Could not extract trajectory_index, saving anyway: {e}")
            new_trajectories.append(traj)

    print(f"New trajectories to append: {len(new_trajectories)} (skipped {skipped_count} already in checkpoint)")
    print(f"  - Newly completed tasks: {len([t for t in new_trajectories if t])}")
    print(f"  - Incomplete/errored tasks: {incomplete_count}")

    # Append new trajectories to checkpoint file
    import torch
    import datetime
    from webgym.utils.trajectory_storage import _strip_page_metadata, _extract_keypoint_step_ids

    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"📂 Loading existing checkpoint: {checkpoint_file}")
        checkpoint_data = torch.load(checkpoint_file, weights_only=False)

        # Handle both old format (list) and new format (dict with trajectories/metadata)
        if isinstance(checkpoint_data, dict) and 'trajectories' in checkpoint_data:
            existing_trajectories = checkpoint_data['trajectories']
            existing_metadata = checkpoint_data.get('metadata', {})
        else:
            existing_trajectories = checkpoint_data
            existing_metadata = {}

        print(f"   Found {len(existing_trajectories)} trajectories in checkpoint")

        # CRITICAL: Populate saved_task_ids from existing trajectories to prevent duplicates
        existing_ids_added = 0
        for existing_traj in existing_trajectories:
            if existing_traj and len(existing_traj) > 0:
                try:
                    existing_task = existing_traj[0]['observation'].task
                    existing_traj_id = getattr(existing_task, 'trajectory_index', None)
                    if existing_traj_id is not None:
                        saved_task_ids.add(f"traj_{existing_traj_id}")
                        existing_ids_added += 1
                    else:
                        existing_task_id = getattr(existing_task, 'task_id', None)
                        if existing_task_id:
                            saved_task_ids.add(existing_task_id)
                            existing_ids_added += 1
                except Exception:
                    pass
        print(f"   Populated saved_task_ids with {existing_ids_added} existing trajectory IDs")

        # Re-filter new_trajectories against updated saved_task_ids
        filtered_new = []
        for traj in new_trajectories:
            if traj and len(traj) > 0:
                try:
                    task = traj[0]['observation'].task
                    traj_id = getattr(task, 'trajectory_index', None)
                    if traj_id is not None:
                        traj_id = f"traj_{traj_id}"
                    else:
                        traj_id = getattr(task, 'task_id', None)
                    if traj_id and traj_id in saved_task_ids:
                        continue  # Skip duplicate
                except Exception:
                    pass
            filtered_new.append(traj)

        if len(filtered_new) < len(new_trajectories):
            print(f"   Filtered out {len(new_trajectories) - len(filtered_new)} duplicates after loading checkpoint")
        new_trajectories = filtered_new

        # Strip page_metadata and extract keypoint IDs from new trajectories
        new_trajectories = _strip_page_metadata(new_trajectories)
        new_trajectories = _extract_keypoint_step_ids(new_trajectories)

        # Append new trajectories
        existing_trajectories.extend(new_trajectories)

        # Save back to checkpoint file with metadata
        if metadata or existing_metadata:
            save_metadata = metadata.copy() if metadata else existing_metadata.copy()
            if 'timestamp' not in save_metadata:
                save_metadata['timestamp'] = datetime.datetime.now().isoformat()
            save_data = {'trajectories': existing_trajectories, 'metadata': save_metadata}
        else:
            save_data = existing_trajectories

        torch.save(save_data, checkpoint_file)
        print(f"💾 Appended {len(new_trajectories)} new trajectories to checkpoint")
        print(f"   Total trajectories in {checkpoint_file}: {len(existing_trajectories)}")

        # Rename checkpoint to final file BEFORE raising exception (in case cleanup hangs)
        if checkpoint_file.endswith('.checkpoint'):
            final_file = checkpoint_file[:-len('.checkpoint')]
            os.rename(checkpoint_file, final_file)
            print(f"📦 Renamed checkpoint to final: {final_file}")
            final_path = final_file
        else:
            final_path = checkpoint_file
    else:
        # No checkpoint file exists (save_traj_progress was false), save as .iteration file
        print(f"⚠️ No checkpoint file found, saving all {len(all_valid)} trajectories as iteration file")
        from webgym.utils.trajectory_storage import get_next_iteration_number

        traj_dir = os.path.join(save_path, f'{split}_trajectories')
        os.makedirs(traj_dir, exist_ok=True)

        iteration_num = get_next_iteration_number(save_path, split)
        # Use .iteration format (not .checkpoint) when save_traj_progress is disabled
        iteration_filename = f'{split}_trajectories.pt.iteration{iteration_num}{rank_suffix}'
        iteration_file = os.path.join(traj_dir, iteration_filename)

        # Strip page_metadata and extract keypoint IDs before saving
        all_valid = _strip_page_metadata(all_valid)
        all_valid = _extract_keypoint_step_ids(all_valid)

        # Save with metadata if provided
        if metadata:
            save_metadata = metadata.copy()
            if 'timestamp' not in save_metadata:
                save_metadata['timestamp'] = datetime.datetime.now().isoformat()
            if 'iteration' not in save_metadata:
                save_metadata['iteration'] = iteration_num
            save_data = {'trajectories': all_valid, 'metadata': save_metadata}
        else:
            save_data = all_valid

        torch.save(save_data, iteration_file)
        print(f"💾 Saved {len(all_valid)} trajectories to {iteration_file}")
        final_path = iteration_file

    # Free memory
    import gc
    del new_trajectories
    if 'existing_trajectories' in locals():
        del existing_trajectories
    gc.collect()

    # Raise exception to signal grace period completion
    # This allows proper cleanup (wandb logging, etc.) instead of os._exit(0) which kills immediately
    print("🚪 Grace period expired - raising exception for proper cleanup")
    raise GracePeriodExpiredException(f"Grace period expired after saving {len(all_valid)} trajectories to {final_path}")
