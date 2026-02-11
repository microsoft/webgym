"""
Incremental trajectory storage utilities.

Instead of saving all trajectories to a single large file (which causes EOFErrors
during concurrent read/write), we save each rollout iteration to a separate file
in a directory structure.

Structure:
  {save_path}/train_trajectories/
    ├── train_trajectories.pt.iteration1
    ├── train_trajectories.pt.iteration2
    └── ...

File format:
  Each .pt file contains a dict: {'trajectories': [...], 'metadata': {...}}
"""
import os
import re
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path


def get_next_iteration_number(base_dir: str, split: str) -> int:
    """
    Find the next iteration number by scanning existing iteration files.

    Args:
        base_dir: Base directory containing trajectory files (e.g., /data/logs)
        split: 'train' or 'test'

    Returns:
        Next iteration number (1 if no files exist)
    """
    traj_dir = os.path.join(base_dir, f'{split}_trajectories')

    # Create directory if it doesn't exist
    os.makedirs(traj_dir, exist_ok=True)

    # Find all iteration files
    if not os.path.exists(traj_dir):
        return 1

    max_iteration = 0
    # Only match final merged files (without rank suffix like _rank_0)
    pattern = re.compile(rf'{split}_trajectories\.pt\.iteration(\d+)$')

    for filename in os.listdir(traj_dir):
        match = pattern.match(filename)
        if match:
            iteration = int(match.group(1))
            max_iteration = max(max_iteration, iteration)

    return max_iteration + 1


def _strip_page_metadata(trajectories: List[List[Dict]]) -> List[List[Dict]]:
    """
    Strip unnecessary page_metadata fields to reduce storage size.

    Keeps only 'title' and 'url' which are the only fields used by the code.
    Removes: microdata, jsonld, meta_tags (which can be 10KB+ per step).

    Args:
        trajectories: List of trajectories (each trajectory is a list of steps)

    Returns:
        Trajectories with stripped page_metadata
    """
    for traj in trajectories:
        if traj is None:
            continue
        for step in traj:
            if step is None or 'observation' not in step:
                continue

            obs = step['observation']
            if not hasattr(obs, 'page_metadata') or obs.page_metadata is None:
                continue

            # Keep only title and url (the only fields actually used)
            stripped_metadata = {
                'title': obs.page_metadata.get('title', 'Unknown'),
                'url': obs.page_metadata.get('url', 'Unknown')
            }
            obs.page_metadata = stripped_metadata

    return trajectories


def _extract_keypoint_step_ids(trajectories: List[List[Dict]]) -> List[List[Dict]]:
    """
    Extract and add keypoint detection step IDs to each trajectory.

    For each trajectory, finds the step indices where submit=True (keypoint detection)
    and adds them as a 'keypoint_step_ids' field to trajectory metadata.

    Args:
        trajectories: List of trajectories (each trajectory is a list of steps)

    Returns:
        Trajectories with keypoint_step_ids added to first step's trajectory_metadata
    """
    for traj in trajectories:
        if traj is None or len(traj) == 0:
            continue

        # Find step indices where submit=True (keypoint detection)
        keypoint_ids = []
        for i, step in enumerate(traj):
            if step is None:
                continue
            reward = step.get('reward')
            if reward is not None and hasattr(reward, 'submit') and reward.submit:
                keypoint_ids.append(i)

        # Store keypoint_step_ids in trajectory-level metadata in the first step
        if len(traj) > 0 and traj[0] is not None:
            if 'trajectory_metadata' not in traj[0]:
                traj[0]['trajectory_metadata'] = {}
            traj[0]['trajectory_metadata']['keypoint_step_ids'] = keypoint_ids

    return trajectories


def save_trajectories_incremental(
    trajectories: List[Dict],
    base_dir: str,
    split: str,
    rank_suffix: str = '',
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save trajectories to an incremental iteration file in dict format.

    Always saves as: {'trajectories': [...], 'metadata': {...}}

    Automatically strips unnecessary page_metadata fields to reduce storage size
    while keeping title and url which are the only fields used by the code.

    Also extracts and saves keypoint detection step IDs for each trajectory.

    Args:
        trajectories: List of trajectory dictionaries to save
        base_dir: Base directory (e.g., /data/logs)
        split: 'train' or 'test'
        rank_suffix: Multinode rank suffix (e.g., '_rank_0', '_rank_1', or '')
        metadata: Optional config metadata to save alongside trajectories
                  (e.g., solution_model, judge_model, dataset_file, timestamp).
                  If None, an empty dict will be created with timestamp and iteration.

    Returns:
        Path to saved file
    """
    import datetime

    traj_dir = os.path.join(base_dir, f'{split}_trajectories')
    os.makedirs(traj_dir, exist_ok=True)

    # Get next iteration number
    iteration_num = get_next_iteration_number(base_dir, split)

    # Create filename with iteration and optional rank suffix
    # e.g., train_trajectories.pt.iteration5_rank_0
    base_filename = f'{split}_trajectories.pt.iteration{iteration_num}{rank_suffix}'
    filepath = os.path.join(traj_dir, base_filename)

    # Strip unnecessary page_metadata before saving to reduce file size
    trajectories = _strip_page_metadata(trajectories)

    # Extract and add keypoint detection step IDs to each trajectory
    trajectories = _extract_keypoint_step_ids(trajectories)

    # Initialize metadata if not provided
    if metadata is None:
        metadata = {}

    # Add timestamp if not present
    if 'timestamp' not in metadata:
        metadata['timestamp'] = datetime.datetime.now().isoformat()
    if 'iteration' not in metadata:
        metadata['iteration'] = iteration_num

    # Always save as dict with trajectories and metadata
    save_data = {
        'trajectories': trajectories,
        'metadata': metadata
    }

    # Save trajectories with metadata
    torch.save(save_data, filepath)

    return filepath


def _extract_trajectories_from_data(data: Any) -> List[Dict]:
    """
    Extract trajectories from loaded data.

    Expected format: Dict with 'trajectories' and 'metadata' keys

    Args:
        data: Loaded data from torch.load

    Returns:
        List of trajectories
    """
    if isinstance(data, dict) and 'trajectories' in data:
        return data['trajectories']
    else:
        raise ValueError(
            f"Invalid trajectory data format. Expected dict with 'trajectories' key, got {type(data)}. "
            f"Keys: {data.keys() if isinstance(data, dict) else 'N/A'}"
        )


def load_last_iteration_trajectories(
    base_dir: str,
    split: str,
    return_metadata: bool = False
) -> List[Dict] | tuple[List[Dict], Optional[Dict]]:
    """
    Load only the LAST iteration's trajectories (most recent).

    This is useful when you only need the most recent performance data.

    Args:
        base_dir: Base directory (e.g., /data/logs)
        split: 'train' or 'test'
        return_metadata: If True, also return metadata dict (or None if not present)

    Returns:
        List of trajectories from the last iteration only
        OR (trajectories, metadata) tuple if return_metadata=True

    Raises:
        FileNotFoundError: If trajectory directory doesn't exist
    """
    traj_dir = os.path.join(base_dir, f'{split}_trajectories')

    if not os.path.exists(traj_dir):
        raise FileNotFoundError(
            f"Trajectory directory not found: {traj_dir}\n"
            f"Please ensure trajectories are saved using the incremental format."
        )

    # Find all iteration files (excluding rank-specific files)
    pattern = re.compile(rf'{split}_trajectories\.pt\.iteration(\d+)$')
    iteration_files = []

    for filename in os.listdir(traj_dir):
        match = pattern.match(filename)
        if match:
            iteration_num = int(match.group(1))
            filepath = os.path.join(traj_dir, filename)
            iteration_files.append((iteration_num, filepath))

    if not iteration_files:
        raise FileNotFoundError(
            f"No iteration files found in {traj_dir}"
        )

    # Sort by iteration number and get the last one
    iteration_files.sort(key=lambda x: x[0])
    last_iteration_num, last_filepath = iteration_files[-1]

    print(f"📂 Loading last iteration ({last_iteration_num}) from {traj_dir}...")
    data = torch.load(last_filepath, weights_only=False)

    # Handle both old and new formats
    trajectories = _extract_trajectories_from_data(data)
    metadata = data.get('metadata') if isinstance(data, dict) else None

    print(f"✓ Loaded {len(trajectories)} trajectories from iteration {last_iteration_num}")

    if return_metadata:
        return trajectories, metadata
    return trajectories


def load_all_trajectories(
    base_dir: str,
    split: str,
    return_file_list: bool = False,
    parallel: bool = True,
    num_workers: int = 4,
    last_n_iterations: Optional[int] = None
) -> List[Dict] | tuple[List[Dict], List[str]]:
    """
    Load all trajectories from all iteration files and combine them.

    Args:
        base_dir: Base directory (e.g., /data/logs)
        split: 'train' or 'test'
        return_file_list: If True, return (trajectories, file_list) tuple
        parallel: If True, load files in parallel (default: True)
        num_workers: Number of parallel workers (default: 4)
        last_n_iterations: If specified, only load the last N iterations (default: None = load all)

    Returns:
        Combined list of all trajectories from all iterations
        OR (trajectories, file_list) if return_file_list=True

    Raises:
        FileNotFoundError: If trajectory directory doesn't exist
    """
    traj_dir = os.path.join(base_dir, f'{split}_trajectories')

    if not os.path.exists(traj_dir):
        raise FileNotFoundError(
            f"Trajectory directory not found: {traj_dir}\n"
            f"Please ensure trajectories are saved using the incremental format.\n"
            f"To migrate legacy files, run:\n"
            f"  python scripts/migrate_to_incremental_trajectories.py --save_path {base_dir} --split {split}"
        )

    # Find all iteration files (excluding rank-specific files)
    pattern = re.compile(rf'{split}_trajectories\.pt\.iteration(\d+)$')
    iteration_files = []

    for filename in os.listdir(traj_dir):
        match = pattern.match(filename)
        if match:
            iteration_num = int(match.group(1))
            filepath = os.path.join(traj_dir, filename)
            iteration_files.append((iteration_num, filepath))

    # Sort by iteration number
    iteration_files.sort(key=lambda x: x[0])

    if not iteration_files:
        raise FileNotFoundError(
            f"No iteration files found in {traj_dir}\n"
            f"Directory exists but contains no valid trajectory files.\n"
            f"Expected files matching pattern: {split}_trajectories.pt.iteration[N]"
        )

    # Filter to last N iterations if specified
    total_iterations_available = len(iteration_files)
    if last_n_iterations is not None and last_n_iterations > 0:
        iteration_files = iteration_files[-last_n_iterations:]
        print(f"📂 Loading last {len(iteration_files)} of {total_iterations_available} iteration files from {traj_dir}...")
    else:
        print(f"📂 Loading {len(iteration_files)} iteration files from {traj_dir}...")

    if parallel and len(iteration_files) > 1:
        # Parallel loading for multiple files
        import concurrent.futures
        import time

        start_time = time.time()

        def load_single_file(iteration_info):
            iteration_num, filepath = iteration_info
            try:
                data = torch.load(filepath, weights_only=False)
                # Handle both old and new formats
                trajectories = _extract_trajectories_from_data(data)
                print(f"  ✓ Loaded iteration {iteration_num}: {len(trajectories)} trajectories")
                return (trajectories, filepath, None)
            except Exception as e:
                print(f"  ⚠️  Failed to load {filepath}: {e}")
                return (None, filepath, e)

        all_trajectories = []
        file_list = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(load_single_file, iteration_files))

        for trajectories, filepath, error in results:
            if trajectories is not None:
                all_trajectories.extend(trajectories)
                file_list.append(filepath)

        elapsed = time.time() - start_time
        print(f"✓ Total trajectories loaded: {len(all_trajectories)} from {len(file_list)} files in {elapsed:.2f}s (parallel)")
    else:
        # Sequential loading (fallback or single file)
        all_trajectories = []
        file_list = []

        for iteration_num, filepath in iteration_files:
            try:
                data = torch.load(filepath, weights_only=False)
                # Handle both old and new formats
                trajectories = _extract_trajectories_from_data(data)
                all_trajectories.extend(trajectories)
                file_list.append(filepath)
                print(f"  ✓ Loaded iteration {iteration_num}: {len(trajectories)} trajectories")
            except Exception as e:
                print(f"  ⚠️  Failed to load {filepath}: {e}")
                continue

        print(f"✓ Total trajectories loaded: {len(all_trajectories)} from {len(file_list)} files")

    if return_file_list:
        return all_trajectories, file_list
    return all_trajectories


def cleanup_old_iterations(
    base_dir: str,
    split: str,
    keep_last_n: int = 5
) -> None:
    """
    Clean up old iteration files, keeping only the last N iterations.

    Args:
        base_dir: Base directory (e.g., /data/logs)
        split: 'train' or 'test'
        keep_last_n: Number of recent iterations to keep
    """
    traj_dir = os.path.join(base_dir, f'{split}_trajectories')

    if not os.path.exists(traj_dir):
        return

    # Find all iteration files (excluding rank-specific files)
    pattern = re.compile(rf'{split}_trajectories\.pt\.iteration(\d+)$')
    iteration_files = []

    for filename in os.listdir(traj_dir):
        match = pattern.match(filename)
        if match:
            iteration_num = int(match.group(1))
            filepath = os.path.join(traj_dir, filename)
            iteration_files.append((iteration_num, filepath))

    # Sort by iteration number (oldest first)
    iteration_files.sort(key=lambda x: x[0])

    # Remove old files
    files_to_remove = iteration_files[:-keep_last_n] if len(iteration_files) > keep_last_n else []

    for iteration_num, filepath in files_to_remove:
        try:
            os.remove(filepath)
            print(f"🗑️  Removed old iteration file: {filepath}")
        except Exception as e:
            print(f"⚠️  Failed to remove {filepath}: {e}")


def get_trajectory_stats(base_dir: str, split: str) -> Dict[str, Any]:
    """
    Get statistics about stored trajectories.

    Args:
        base_dir: Base directory (e.g., /data/logs)
        split: 'train' or 'test'

    Returns:
        Dictionary with statistics
    """
    traj_dir = os.path.join(base_dir, f'{split}_trajectories')

    if not os.path.exists(traj_dir):
        return {
            'total_iterations': 0,
            'total_trajectories': 0,
            'total_size_mb': 0,
            'iterations': []
        }

    # Find all iteration files (excluding rank-specific files)
    pattern = re.compile(rf'{split}_trajectories\.pt\.iteration(\d+)$')
    iteration_files = []

    for filename in os.listdir(traj_dir):
        match = pattern.match(filename)
        if match:
            iteration_num = int(match.group(1))
            filepath = os.path.join(traj_dir, filename)
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB

            try:
                data = torch.load(filepath, weights_only=False)
                # Handle both old and new formats
                trajectories = _extract_trajectories_from_data(data)
                num_trajs = len(trajectories)
            except:
                num_trajs = -1  # Error loading

            iteration_files.append({
                'iteration': iteration_num,
                'filepath': filepath,
                'num_trajectories': num_trajs,
                'size_mb': file_size
            })

    # Sort by iteration
    iteration_files.sort(key=lambda x: x['iteration'])

    total_trajectories = sum(f['num_trajectories'] for f in iteration_files if f['num_trajectories'] > 0)
    total_size = sum(f['size_mb'] for f in iteration_files)

    return {
        'total_iterations': len(iteration_files),
        'total_trajectories': total_trajectories,
        'total_size_mb': total_size,
        'iterations': iteration_files
    }
