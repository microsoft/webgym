import matplotlib.pyplot as plt
import torch
import argparse
import os
import re
import glob
from collections import Counter, defaultdict
from math import comb

# ───── Dummy Classes from create_dummy_trajectories.py ─────
class DummyTask:
    """Dummy task object to mimic real task structure"""
    def __init__(self, task_row):
        self.task_name = task_row['task_name']
        self.subdomain = task_row['subdomain']
        self.website = task_row['website']
        self.difficulty = task_row['difficulty']
        self.domain = task_row.get('domain', '')

class DummyObservation:
    """Dummy observation to hold task"""
    def __init__(self, task):
        self.task = task

class DummyResponse:
    """Dummy response"""
    def __init__(self):
        self.raw_response = "This is a dummy response for testing sampling behavior."

# ───── Dummy Classes to Replace None Values ─────
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

# ───── Function to Clean Trajectories ─────
def clean_trajectories(trajs):
    """Replace None values with dummy objects"""
    cleaned_trajs = []

    for traj in trajs:
        # Skip None trajectories entirely
        if traj is None:
            cleaned_trajs.append([])
            continue

        cleaned_traj = []
        for step in traj:
            # Skip non-dict steps (e.g., strings or other unexpected types)
            if not isinstance(step, dict):
                continue
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

# ───── Original Script Parameters ─────
chunk_train = 512                                # training chunk size (trajectories per iteration)
chunk_test = 1168                                 # test chunk size (trajectories per test rollout)

def is_crashed(t, is_train=True):
    """
    Detect crashed trajectories based on action keys from actual data:
    1. Empty trajectory (crashed before any steps)
    2. invalid_step action (crashed immediately)
    3. error action (e.g., GRACE_PERIOD_INCOMPLETE, screenshot failures)
    4. Trajectory ended early (len < max_steps) without an answer action
       EXCEPT: Homepage loop pattern (3 consecutive gobacks at start)
    """
    # Check for empty trajectory (crashed before any steps)
    if not t or len(t) == 0:
        return True

    # Check for crash indicators in first action
    first_action_key = t[0]['action'].action.get('key', '')
    if first_action_key == 'invalid_step' or first_action_key == 'error':
        return True

    # Check if trajectory ended prematurely (before max_steps) without answer action
    if t and len(t) > 0:
        # Get max_steps from task
        if hasattr(t[0]['observation'], 'task') and hasattr(t[0]['observation'].task, 'max_steps'):
            max_steps = t[0]['observation'].task.max_steps
            # Check if trajectory is shorter than max_steps AND last action is not 'answer'
            if len(t) < max_steps:
                last_action_key = t[-1]['action'].action.get('key', '')
                if last_action_key != 'answer':
                    # Before counting as crashed, check if it's a homepage loop
                    # Homepage loop: last 3 steps are all consecutive goback actions
                    if len(t) >= 3:
                        # Check if last 3 steps all have goback action
                        last_three_all_goback = True
                        for step_idx in range(len(t) - 3, len(t)):
                            step_action = t[step_idx]['action'].action.get('key', '')
                            if step_action != 'goback':
                                last_three_all_goback = False
                                break

                        if last_three_all_goback:
                            # This is a homepage loop - NOT a crash
                            # Agent tried to go back from homepage 3 consecutive times
                            return False

                    # Not a homepage loop - this is a real crash/failure
                    return True

    return False

# ───── pass@k calculation helper ─────
def calculate_pass_at_k(n, c, k):
    """
    Calculate pass@k metric.

    Args:
        n: Total number of samples per task
        c: Number of correct samples
        k: Number of samples to evaluate

    Returns:
        pass@k probability (0 to 1)
    """
    if n < k:
        return 0.0
    if c == 0:
        return 0.0
    if c >= k:
        return 1.0

    # pass@k = 1 - C(n-c, k) / C(n, k)
    # This is the probability that at least one correct sample is in k random samples
    try:
        return 1.0 - (comb(n - c, k) / comb(n, k))
    except:
        return 0.0

# ───── Helper to load and split test data ─────
def load_and_split_test_data(traj_dir, task_file_dir):
    """
    Load test trajectories and filter for OOD only.

    Args:
        traj_dir: Directory containing trajectory files (run-specific directory)
        task_file_dir: Directory containing task definition files (base logs directory)

    Returns (ood_trajs_by_iteration, iteration_counts)
    """
    import glob
    import sys
    import json
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    # Load task definitions to create task_id → distribution mapping
    task_file = os.path.join(task_file_dir, 'test.jsonl')
    task_id_to_distribution = {}

    with open(task_file, 'r') as f:
        for line in f:
            task_def = json.loads(line.strip())
            # Construct composite task_id matching the format used in trajectories
            # Format: subdomain_website_difficulty_taskname
            subdomain = task_def.get('subdomain', '')
            website = task_def.get('website', '')
            difficulty = task_def.get('difficulty')
            task_name = task_def.get('task_name', '')

            if subdomain and website and difficulty is not None and task_name:
                # Keep the website as-is (including https:// if present)
                composite_task_id = f"{subdomain}_{website}_{difficulty}_{task_name}"
                distribution = task_def.get('distribution', 'ood')
                task_id_to_distribution[composite_task_id] = distribution

    test_traj_dir = os.path.join(traj_dir, 'test_trajectories')
    iteration_files = sorted(glob.glob(os.path.join(test_traj_dir, 'test_trajectories.pt.iteration*')),
                            key=lambda x: int(re.search(r'iteration(\d+)', x).group(1)))

    ood_trajs_by_iter = []
    iteration_counts = []  # Cumulative counts
    cumulative = 0

    for iter_file in iteration_files:
        data = torch.load(iter_file, map_location='cpu', weights_only=False)
        trajs = data['trajectories']
        trajs = clean_trajectories(trajs)

        ood_trajs = []

        for traj in trajs:
            if traj and len(traj) > 0:
                obs = traj[0].get('observation')
                # Get task_id from the trajectory
                task_id = None
                if obs and hasattr(obs, 'task'):
                    task_id = getattr(obs.task, 'task_id', None)

                # Look up distribution in the mapping - only keep OOD
                if task_id in task_id_to_distribution:
                    distribution = task_id_to_distribution[task_id]
                    if distribution == 'ood':
                        ood_trajs.append(traj)
                else:
                    ood_trajs.append(traj)  # Default to OOD if task_id not found
            else:
                ood_trajs.append(traj)

        ood_trajs_by_iter.append(ood_trajs)
        cumulative += len(trajs)
        iteration_counts.append(cumulative)

    return ood_trajs_by_iter, iteration_counts

# ───── aggregate helper (per-iteration) ─────
def aggregate(traj_dir, task_file_dir, split, is_train=True, train_tasks_per_iteration=None, train_websites_per_iteration=None, distribution_filter=None):
    """
    Aggregate metrics per iteration file (not fixed chunks).

    Args:
        traj_dir: Directory containing trajectory files (run-specific directory)
        task_file_dir: Directory containing task definition files (base logs directory)
        split: 'train' or 'test'
        ... (other args)

    Returns metrics dict, diversity dict, per-difficulty dicts, and iteration_counts (cumulative)
    """
    import glob
    import sys
    import json
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    # Load task_id → distribution mapping if we need to filter by distribution
    task_id_to_distribution = {}
    if distribution_filter is not None:
        if split == 'test':
            task_file = os.path.join(task_file_dir, 'test.jsonl')
        else:  # train
            task_file = os.path.join(task_file_dir, 'train.jsonl')
        with open(task_file, 'r') as f:
            for line in f:
                task_def = json.loads(line.strip())
                # Construct composite task_id matching the format used elsewhere
                # Format: subdomain_website_difficulty_taskname
                subdomain = task_def.get('subdomain', '')
                website = task_def.get('website', '')
                difficulty = task_def.get('difficulty')
                task_name = task_def.get('task_name', '')

                if subdomain and website and difficulty is not None and task_name:
                    # Keep the website as-is (including https:// if present)
                    composite_task_id = f"{subdomain}_{website}_{difficulty}_{task_name}"
                    distribution = task_def.get('distribution', 'ood')
                    task_id_to_distribution[composite_task_id] = distribution

        # Debug: count distributions
        dist_counts = {}
        for dist in task_id_to_distribution.values():
            dist_counts[dist] = dist_counts.get(dist, 0) + 1
        print(f"\nLoading {split} data with {distribution_filter} filter:")
        print(f"  Task file: {task_file}")
        print(f"  Total tasks in file: {len(task_id_to_distribution)}")
        print(f"  Distribution breakdown: {dist_counts}")
        print(f"  Sample composite task_id: {list(task_id_to_distribution.keys())[0] if task_id_to_distribution else 'none'}")

    # Find all iteration files
    trajectory_dir = os.path.join(traj_dir, f'{split}_trajectories')
    iteration_files = sorted(glob.glob(os.path.join(trajectory_dir, f'{split}_trajectories.pt.iteration*')),
                            key=lambda x: int(re.search(r'iteration(\d+)', x).group(1)))

    if not iteration_files:
        return None

    # Print header for train data (when no distribution filter)
    if is_train and distribution_filter is None:
        task_file = os.path.join(task_file_dir, 'train.jsonl')
        task_count = 0
        if os.path.exists(task_file):
            with open(task_file, 'r') as f:
                task_count = sum(1 for _ in f)
        print(f"\nLoading {split} data:")
        print(f"  Task file: {task_file}")
        print(f"  Total tasks in file: {task_count}")

    # Initialize metrics
    met = {k: [] for k in ("succ","act","memory_chars","steps","goback","same_screenshot","same_screenshot_success","crashed","blocked","no_memory_update","empty_memory_update","step_limited_success")}
    difficulty_to_threshold = {1: 5, 2: 8, 3: 10, 4: 15, 5: 20, 6: 30, 7: 50}

    succ_by_difficulty = {}
    samples_by_difficulty = {}
    chars_by_difficulty = {}
    memory_chars_by_difficulty = {}
    steps_by_difficulty = {}
    goback_by_difficulty = {}
    same_screenshot_by_difficulty = {}
    same_screenshot_success_by_difficulty = {}
    crash_by_difficulty = {}
    step_limited_by_difficulty = {}

    diversity = {k: [] for k in ("num_tasks", "num_websites", "tasks_overlap", "websites_overlap", "tasks_repetitive", "tasks_with_seen_websites")}
    all_previous_tasks = set()
    all_previous_websites = set()

    if not is_train and train_tasks_per_iteration is not None:
        diversity["test_tasks_in_train"] = []
        diversity["test_websites_in_train"] = []
        diversity["test_tasks_with_website_in_train"] = []

    tasks_per_iteration = []
    websites_per_iteration = []
    iteration_counts = []  # Cumulative trajectory counts
    cumulative = 0

    # Process each iteration file
    for iter_idx, iter_file in enumerate(iteration_files):
        data = torch.load(iter_file, map_location='cpu', weights_only=False)
        trajs = data['trajectories']
        trajs = clean_trajectories(trajs)

        # Filter by distribution if specified
        if distribution_filter is not None:
            filtered_trajs = []
            original_count = len(trajs)
            found_task_ids = set()
            matched_task_ids = set()
            missing_task_ids = set()

            for traj in trajs:
                if traj and len(traj) > 0:
                    obs = traj[0].get('observation')
                    # Construct composite task_id from task object (matching format used elsewhere)
                    task_id = None
                    if obs and hasattr(obs, 'task'):
                        task_obj = obs.task
                        if hasattr(task_obj, 'task_name') and hasattr(task_obj, 'subdomain') and hasattr(task_obj, 'website') and hasattr(task_obj, 'difficulty'):
                            task_id = f"{task_obj.subdomain}_{task_obj.website}_{task_obj.difficulty}_{task_obj.task_name}"

                    if task_id:
                        found_task_ids.add(task_id)

                    # Look up distribution in the mapping
                    if task_id in task_id_to_distribution:
                        if task_id_to_distribution[task_id] == distribution_filter:
                            filtered_trajs.append(traj)
                            matched_task_ids.add(task_id)
                    elif task_id:
                        missing_task_ids.add(task_id)

            print(f"  Iteration {iter_idx}: {distribution_filter} filter: {original_count} -> {len(filtered_trajs)} trajectories")
            if iter_idx == 0:  # Only print detailed info for first iteration
                print(f"    Task IDs found in trajectories: {len(found_task_ids)}")
                print(f"    Task IDs matched filter: {len(matched_task_ids)}")
                print(f"    Task IDs not in mapping: {len(missing_task_ids)}")
                if missing_task_ids:
                    print(f"    Sample missing task IDs: {list(missing_task_ids)[:3]}")
                if found_task_ids and not matched_task_ids:
                    sample_id = list(found_task_ids)[0]
                    if sample_id in task_id_to_distribution:
                        print(f"    Sample task '{sample_id}' has distribution: {task_id_to_distribution[sample_id]}")

            trajs = filtered_trajs
        elif is_train:
            # Print train iteration info (no filtering)
            print(f"  Iteration {iter_idx}: train: {len(trajs)} trajectories")

        c = trajs

        # Calculate current data point index BEFORE appending to iteration_counts
        current_data_point_idx = len(iteration_counts)

        cumulative += len(c)
        iteration_counts.append(cumulative)

        # If no trajectories after filtering, append 0 values for all metrics
        if not c:
            # Append 0 for all main metrics
            for key in met.keys():
                met[key].append(0)

            # Append 0 for diversity metrics
            for key in diversity.keys():
                diversity[key].append(0)

            # Append 0 for all existing difficulty-based metrics
            for diff in samples_by_difficulty.keys():
                samples_by_difficulty[diff].append(0)
            for diff in succ_by_difficulty.keys():
                succ_by_difficulty[diff].append(0)
            for diff in chars_by_difficulty.keys():
                chars_by_difficulty[diff].append(0)
            for diff in memory_chars_by_difficulty.keys():
                memory_chars_by_difficulty[diff].append(0)
            for diff in steps_by_difficulty.keys():
                steps_by_difficulty[diff].append(0)
            for diff in goback_by_difficulty.keys():
                goback_by_difficulty[diff].append(0)
            for diff in same_screenshot_by_difficulty.keys():
                same_screenshot_by_difficulty[diff].append(0)
            for diff in same_screenshot_success_by_difficulty.keys():
                same_screenshot_success_by_difficulty[diff].append(0)
            for diff in crash_by_difficulty.keys():
                crash_by_difficulty[diff].append(0)
            for diff in step_limited_by_difficulty.keys():
                step_limited_by_difficulty[diff].append(0)

            # Append empty sets for tasks and websites tracking
            tasks_per_iteration.append(set())
            websites_per_iteration.append(set())

            continue

        tot = len(c)

        # Extract task and website information for diversity metrics
        current_tasks = set()
        current_websites = set()
        current_tasks_list = []  # Track all tasks including duplicates
        for t in c:
            if t and len(t) > 0 and hasattr(t[0]['observation'], 'task'):
                task_obj = t[0]['observation'].task
                # Create unique task identifier
                if hasattr(task_obj, 'task_name'):
                    task_id = f"{task_obj.subdomain}_{task_obj.website}_{task_obj.difficulty}_{task_obj.task_name}"
                    current_tasks.add(task_id)
                    current_tasks_list.append(task_id)
                # Track website
                if hasattr(task_obj, 'website'):
                    current_websites.add(task_obj.website)

        # Calculate number of repetitive tasks
        task_counts = Counter(current_tasks_list)
        repetitive_tasks = sum(1 for count in task_counts.values() if count > 1)
        diversity["tasks_repetitive"].append(repetitive_tasks)

        # Track sample counts by difficulty
        difficulty_counts = {}
        for t in c:
            if t and len(t) > 0 and hasattr(t[0]['observation'], 'task'):
                task_obj = t[0]['observation'].task
                if hasattr(task_obj, 'difficulty'):
                    diff = task_obj.difficulty
                    if diff not in difficulty_counts:
                        difficulty_counts[diff] = 0
                    difficulty_counts[diff] += 1

        # Add counts to samples_by_difficulty
        for diff, count in difficulty_counts.items():
            if diff not in samples_by_difficulty:
                samples_by_difficulty[diff] = [0] * current_data_point_idx
            samples_by_difficulty[diff].append(count)

        # For difficulties not seen in this iteration, append 0
        for diff in samples_by_difficulty.keys():
            if diff not in difficulty_counts:
                samples_by_difficulty[diff].append(0)

        # Calculate diversity metrics
        diversity["num_tasks"].append(len(current_tasks))
        diversity["num_websites"].append(len(current_websites))

        # Calculate overlap with ALL previous iterations
        if iter_idx == 0:
            diversity["tasks_overlap"].append(0)
            diversity["websites_overlap"].append(0)
        else:
            tasks_overlap = len(current_tasks & all_previous_tasks)
            websites_overlap = len(current_websites & all_previous_websites)
            diversity["tasks_overlap"].append(tasks_overlap)
            diversity["websites_overlap"].append(websites_overlap)

        # Calculate # of tasks (unique) whose website was seen in previous iterations
        if iter_idx == 0:
            diversity["tasks_with_seen_websites"].append(0)
        else:
            tasks_with_seen_websites = 0
            for task_id in current_tasks:
                # Extract website from task_id (format: subdomain_website_difficulty_task_name)
                parts = task_id.split('_')
                if len(parts) >= 2:
                    website = parts[1]
                    if website in all_previous_websites:
                        tasks_with_seen_websites += 1
            diversity["tasks_with_seen_websites"].append(tasks_with_seen_websites)

        # Calculate test overlap with cumulative training set
        if not is_train and train_tasks_per_iteration is not None:
            cumulative_train_tasks = set()
            cumulative_train_websites = set()
            for j in range(iter_idx + 1):
                if j < len(train_tasks_per_iteration):
                    cumulative_train_tasks = cumulative_train_tasks | train_tasks_per_iteration[j]
                if j < len(train_websites_per_iteration):
                    cumulative_train_websites = cumulative_train_websites | train_websites_per_iteration[j]

            test_tasks_in_train = len(current_tasks & cumulative_train_tasks)
            test_websites_in_train = len(current_websites & cumulative_train_websites)
            diversity["test_tasks_in_train"].append(test_tasks_in_train)
            diversity["test_websites_in_train"].append(test_websites_in_train)

            test_tasks_with_website_in_train = 0
            for task_id in current_tasks:
                parts = task_id.split('_')
                if len(parts) >= 2:
                    website = parts[1]
                    if website in cumulative_train_websites:
                        test_tasks_with_website_in_train += 1
            diversity["test_tasks_with_website_in_train"].append(test_tasks_with_website_in_train)

        # Update cumulative sets
        all_previous_tasks = all_previous_tasks | current_tasks
        all_previous_websites = all_previous_websites | current_websites

        # Store for return (always store, not just for train)
        tasks_per_iteration.append(current_tasks)
        websites_per_iteration.append(current_websites)
        
        # Calculate crashed rate
        met["crashed"].append(sum(is_crashed(t, is_train) for t in c)/tot)

        # Calculate crash rate by difficulty
        difficulty_crash_stats = {}
        for t in c:
            if t and len(t) > 0 and hasattr(t[0]['observation'], 'task'):
                task_obj = t[0]['observation'].task
                if hasattr(task_obj, 'difficulty'):
                    diff = task_obj.difficulty
                    if diff not in difficulty_crash_stats:
                        difficulty_crash_stats[diff] = {'crashed': 0, 'total': 0}
                    difficulty_crash_stats[diff]['total'] += 1
                    if is_crashed(t, is_train):
                        difficulty_crash_stats[diff]['crashed'] += 1

        for diff, stats in difficulty_crash_stats.items():
            if diff not in crash_by_difficulty:
                crash_by_difficulty[diff] = [None] * current_data_point_idx
            crash_rate = stats['crashed'] / stats['total'] if stats['total'] > 0 else 0
            crash_by_difficulty[diff].append(crash_rate)

        for diff in crash_by_difficulty.keys():
            if diff not in difficulty_crash_stats:
                crash_by_difficulty[diff].append(None)

        # Calculate blocked rate
        blocked_count = sum(
            t[-1]['reward'].is_blocked
            for t in c if t and len(t) > 0
        )
        met["blocked"].append(blocked_count / tot if tot > 0 else 0)

        # Filter out crashed trajectories
        c_valid = [t for t in c if not is_crashed(t, is_train)]

        # Initialize difficulty stats dictionaries
        difficulty_stats = {}
        difficulty_goback_stats = {}

        if not c_valid:
            met["succ"].append(0)
            met["act"].append(0)
            met["memory_chars"].append(0)
            met["steps"].append(0)
            met["goback"].append(0)
            met["same_screenshot"].append(0)
            met["same_screenshot_success"].append(0)
            met["no_memory_update"].append(0)
            met["empty_memory_update"].append(0)
            for diff in succ_by_difficulty.keys():
                succ_by_difficulty[diff].append(None)
            for diff in chars_by_difficulty.keys():
                chars_by_difficulty[diff].append(None)
            for diff in memory_chars_by_difficulty.keys():
                memory_chars_by_difficulty[diff].append(None)
            for diff in steps_by_difficulty.keys():
                steps_by_difficulty[diff].append(None)
            for diff in goback_by_difficulty.keys():
                goback_by_difficulty[diff].append(None)
            for diff in same_screenshot_by_difficulty.keys():
                same_screenshot_by_difficulty[diff].append(None)
            for diff in same_screenshot_success_by_difficulty.keys():
                same_screenshot_success_by_difficulty[diff].append(None)
        else:
            tot_non_crashed = len(c_valid)

            # Filter out blocked trajectories
            c_non_blocked = [
                t for t in c_valid
                if not t[-1]['reward'].is_blocked
            ]

            if not c_non_blocked:
                met["succ"].append(0)
                met["act"].append(0)
                met["memory_chars"].append(0)
                met["steps"].append(0)
                met["goback"].append(0)
                met["same_screenshot"].append(0)
                met["same_screenshot_success"].append(0)
                met["no_memory_update"].append(0)
                met["empty_memory_update"].append(0)
                for diff in succ_by_difficulty.keys():
                    succ_by_difficulty[diff].append(None)
                for diff in chars_by_difficulty.keys():
                    chars_by_difficulty[diff].append(None)
                for diff in memory_chars_by_difficulty.keys():
                    memory_chars_by_difficulty[diff].append(None)
                for diff in steps_by_difficulty.keys():
                    steps_by_difficulty[diff].append(None)
                for diff in goback_by_difficulty.keys():
                    goback_by_difficulty[diff].append(None)
                for diff in same_screenshot_by_difficulty.keys():
                    same_screenshot_by_difficulty[diff].append(None)
                for diff in same_screenshot_success_by_difficulty.keys():
                    same_screenshot_success_by_difficulty[diff].append(None)
            else:
                tot_non_blocked = len(c_non_blocked)
                tot_st = sum(len(t) for t in c_non_blocked)

                # Calculate success rate
                met["succ"].append(sum(t[-1]['reward'].reward==1 for t in c_non_blocked)/tot_non_blocked)

                # Helper function to extract memory chars from response
                def get_memory_chars(response_text):
                    """Extract character count from Memory: line in response"""
                    if not response_text:
                        return 0
                    for line in response_text.split('\n'):
                        if line.strip().startswith('Memory:'):
                            # Return length of text after "Memory:"
                            memory_content = line.split('Memory:', 1)[1] if 'Memory:' in line else ''
                            return len(memory_content.strip())
                    return 0

                # Calculate metrics by difficulty
                for t in c_non_blocked:
                    if t and len(t) > 0 and hasattr(t[0]['observation'], 'task'):
                        task_obj = t[0]['observation'].task
                        if hasattr(task_obj, 'difficulty'):
                            diff = task_obj.difficulty
                            if diff not in difficulty_stats:
                                difficulty_stats[diff] = {'success': 0, 'total': 0, 'chars': 0, 'memory_chars': 0, 'steps': 0}
                            difficulty_stats[diff]['total'] += 1
                            if t[-1]['reward'].reward == 1:
                                difficulty_stats[diff]['success'] += 1
                            difficulty_stats[diff]['chars'] += sum(len(s['response'].raw_response) for s in t)
                            difficulty_stats[diff]['memory_chars'] += sum(get_memory_chars(s['response'].raw_response) for s in t)
                            difficulty_stats[diff]['steps'] += len(t)

                # Add success rates to succ_by_difficulty
                for diff, stats in difficulty_stats.items():
                    if diff not in succ_by_difficulty:
                        succ_by_difficulty[diff] = [None] * current_data_point_idx
                    success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
                    succ_by_difficulty[diff].append(success_rate)

                # Add average chars per response
                for diff, stats in difficulty_stats.items():
                    if diff not in chars_by_difficulty:
                        chars_by_difficulty[diff] = [None] * current_data_point_idx
                    avg_chars = stats['chars'] / stats['steps'] if stats['steps'] > 0 else 0
                    chars_by_difficulty[diff].append(avg_chars)

                # Add average memory chars per response
                for diff, stats in difficulty_stats.items():
                    if diff not in memory_chars_by_difficulty:
                        memory_chars_by_difficulty[diff] = [None] * current_data_point_idx
                    avg_memory_chars = stats['memory_chars'] / stats['steps'] if stats['steps'] > 0 else 0
                    memory_chars_by_difficulty[diff].append(avg_memory_chars)

                # Add average steps
                for diff, stats in difficulty_stats.items():
                    if diff not in steps_by_difficulty:
                        steps_by_difficulty[diff] = [None] * current_data_point_idx
                    avg_steps = stats['steps'] / stats['total'] if stats['total'] > 0 else 0
                    steps_by_difficulty[diff].append(avg_steps)

                # Calculate GoBack rate by difficulty
                for t in c_non_blocked:
                    if t and len(t) > 0 and hasattr(t[0]['observation'], 'task'):
                        task_obj = t[0]['observation'].task
                        if hasattr(task_obj, 'difficulty'):
                            diff = task_obj.difficulty
                            if diff not in difficulty_goback_stats:
                                difficulty_goback_stats[diff] = {'goback_count': 0, 'total_steps': 0}
                            goback_count = sum(1 for s in t if s['action'].action['key'] == 'goback')
                            difficulty_goback_stats[diff]['goback_count'] += goback_count
                            difficulty_goback_stats[diff]['total_steps'] += len(t)

                for diff, stats in difficulty_goback_stats.items():
                    if diff not in goback_by_difficulty:
                        goback_by_difficulty[diff] = [None] * current_data_point_idx
                    goback_rate = stats['goback_count'] / stats['total_steps'] if stats['total_steps'] > 0 else 0
                    goback_by_difficulty[diff].append(goback_rate)

                # Calculate Same-Screenshot rate by difficulty
                difficulty_same_screenshot_stats = {}
                for t in c_non_blocked:
                    if t and len(t) > 0 and hasattr(t[0]['observation'], 'task'):
                        task_obj = t[0]['observation'].task
                        if hasattr(task_obj, 'difficulty'):
                            diff = task_obj.difficulty
                            if diff not in difficulty_same_screenshot_stats:
                                difficulty_same_screenshot_stats[diff] = {'same_screenshot_count': 0, 'total_steps': 0}
                            # Count steps with same_as_next_screenshot = True
                            # For legacy trajectories without this field, treat as 0
                            same_screenshot_count = sum(1 for s in t if s.get('same_as_next_screenshot', False))
                            difficulty_same_screenshot_stats[diff]['same_screenshot_count'] += same_screenshot_count
                            difficulty_same_screenshot_stats[diff]['total_steps'] += len(t)

                for diff, stats in difficulty_same_screenshot_stats.items():
                    if diff not in same_screenshot_by_difficulty:
                        same_screenshot_by_difficulty[diff] = [None] * current_data_point_idx
                    same_screenshot_rate = stats['same_screenshot_count'] / stats['total_steps'] if stats['total_steps'] > 0 else 0
                    same_screenshot_by_difficulty[diff].append(same_screenshot_rate)

                # Calculate Same-Screenshot rate by difficulty for successful trajectories only
                difficulty_same_screenshot_success_stats = {}
                c_success = [t for t in c_non_blocked if t[-1]['reward'].reward == 1]
                for t in c_success:
                    if t and len(t) > 0 and hasattr(t[0]['observation'], 'task'):
                        task_obj = t[0]['observation'].task
                        if hasattr(task_obj, 'difficulty'):
                            diff = task_obj.difficulty
                            if diff not in difficulty_same_screenshot_success_stats:
                                difficulty_same_screenshot_success_stats[diff] = {'same_screenshot_count': 0, 'total_steps': 0}
                            # Count steps with same_as_next_screenshot = True
                            same_screenshot_count = sum(1 for s in t if s.get('same_as_next_screenshot', False))
                            difficulty_same_screenshot_success_stats[diff]['same_screenshot_count'] += same_screenshot_count
                            difficulty_same_screenshot_success_stats[diff]['total_steps'] += len(t)

                for diff, stats in difficulty_same_screenshot_success_stats.items():
                    if diff not in same_screenshot_success_by_difficulty:
                        same_screenshot_success_by_difficulty[diff] = [None] * current_data_point_idx
                    same_screenshot_rate = stats['same_screenshot_count'] / stats['total_steps'] if stats['total_steps'] > 0 else 0
                    same_screenshot_success_by_difficulty[diff].append(same_screenshot_rate)

                # For difficulties not seen in this iteration, append None
                for diff in succ_by_difficulty.keys():
                    if diff not in difficulty_stats:
                        succ_by_difficulty[diff].append(None)
                for diff in chars_by_difficulty.keys():
                    if diff not in difficulty_stats:
                        chars_by_difficulty[diff].append(None)
                for diff in memory_chars_by_difficulty.keys():
                    if diff not in difficulty_stats:
                        memory_chars_by_difficulty[diff].append(None)
                for diff in steps_by_difficulty.keys():
                    if diff not in difficulty_stats:
                        steps_by_difficulty[diff].append(None)
                for diff in goback_by_difficulty.keys():
                    if diff not in difficulty_goback_stats:
                        goback_by_difficulty[diff].append(None)
                for diff in same_screenshot_by_difficulty.keys():
                    if diff not in difficulty_same_screenshot_stats:
                        same_screenshot_by_difficulty[diff].append(None)
                for diff in same_screenshot_success_by_difficulty.keys():
                    if diff not in difficulty_same_screenshot_success_stats:
                        same_screenshot_success_by_difficulty[diff].append(None)

                # Token counting
                met["act"].append(sum(len(s['response'].raw_response) for t in c_non_blocked for s in t)/tot_st)

                # Memory chars counting
                met["memory_chars"].append(sum(get_memory_chars(s['response'].raw_response) for t in c_non_blocked for s in t)/tot_st)

                # Steps calculation
                met["steps"].append(tot_st/tot_non_blocked)

                # GoBack detection
                met["goback"].append(sum(s['action'].action['key']=='goback' for t in c_non_blocked for s in t)/tot_st)

                # Same-Screenshot detection (for legacy trajectories without field, treat as False)
                met["same_screenshot"].append(sum(s.get('same_as_next_screenshot', False) for t in c_non_blocked for s in t)/tot_st)

                # Same-Screenshot detection for successful trajectories only
                c_success = [t for t in c_non_blocked if t[-1]['reward'].reward == 1]
                if c_success:
                    tot_st_success = sum(len(t) for t in c_success)
                    met["same_screenshot_success"].append(sum(s.get('same_as_next_screenshot', False) for t in c_success for s in t)/tot_st_success)
                else:
                    met["same_screenshot_success"].append(0)

                # Memory update metrics
                def last_line_starts_with_memory_updated(step):
                    response = step['response'].raw_response
                    if not response:
                        return False
                    lines = response.strip().split('\n')
                    last_line = lines[-1].strip() if lines else ""
                    return last_line.startswith("Memory_Updated")

                well_formatted_count = sum(last_line_starts_with_memory_updated(s) for t in c_non_blocked for s in t)
                met["no_memory_update"].append(well_formatted_count / tot_st)

                def has_empty_memory_update(step):
                    response = step['response'].raw_response
                    if not response:
                        return False
                    lines = response.strip().split('\n')
                    last_line = lines[-1].strip() if lines else ""
                    if last_line.startswith("Memory_Updated"):
                        memory_content = last_line.replace("Memory_Updated:", "").strip()
                        return memory_content == "{}"
                    return False

                empty_memory_update_count = sum(has_empty_memory_update(s) for t in c_non_blocked for s in t)
                met["empty_memory_update"].append(empty_memory_update_count / tot_st)

        # Step-limited success rate (for test set)
        if not is_train:
            difficulty_stats = {}

            for t in c_valid:
                if t and len(t) > 0 and hasattr(t[0]['observation'], 'task'):
                    task_obj = t[0]['observation'].task
                    if hasattr(task_obj, 'difficulty'):
                        diff = task_obj.difficulty
                        if diff not in difficulty_stats:
                            difficulty_stats[diff] = {'success': 0, 'total': 0}

                        difficulty_stats[diff]['total'] += 1

                        threshold = difficulty_to_threshold.get(diff, 50)

                        if len(t) <= threshold and t[-1]['reward'].reward == 1:
                            difficulty_stats[diff]['success'] += 1

            total_success = sum(stats['success'] for stats in difficulty_stats.values())
            total_count = sum(stats['total'] for stats in difficulty_stats.values())
            if total_count > 0:
                met["step_limited_success"].append(total_success / total_count)
            else:
                met["step_limited_success"].append(0)

            for diff, stats in difficulty_stats.items():
                if diff not in step_limited_by_difficulty:
                    step_limited_by_difficulty[diff] = [None] * current_data_point_idx
                success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
                step_limited_by_difficulty[diff].append(success_rate)

            for diff in step_limited_by_difficulty.keys():
                if diff not in difficulty_stats:
                    step_limited_by_difficulty[diff].append(None)
        else:
            met["step_limited_success"].append(0)

    # Return results
    if is_train:
        return met, diversity, succ_by_difficulty, samples_by_difficulty, chars_by_difficulty, memory_chars_by_difficulty, steps_by_difficulty, goback_by_difficulty, same_screenshot_by_difficulty, same_screenshot_success_by_difficulty, crash_by_difficulty, tasks_per_iteration, websites_per_iteration, iteration_counts
    else:
        return met, diversity, succ_by_difficulty, samples_by_difficulty, chars_by_difficulty, memory_chars_by_difficulty, steps_by_difficulty, goback_by_difficulty, same_screenshot_by_difficulty, same_screenshot_success_by_difficulty, crash_by_difficulty, step_limited_by_difficulty, tasks_per_iteration, websites_per_iteration, iteration_counts

# ───── EMA smoothing function ─────
def apply_ema(data, alpha):
    """
    Apply exponential moving average smoothing to data.

    Args:
        data: List of values (may contain None)
        alpha: Smoothing factor (0 to 1)
               - 1.0 = no smoothing (original data)
               - 0.0 = maximum smoothing (almost flat)

    Returns:
        Smoothed list with same length as input
    """
    if not data or alpha == 1.0:
        return data

    smoothed = []
    ema_value = None

    for value in data:
        if value is None:
            smoothed.append(None)
        else:
            if ema_value is None:
                # First non-None value - initialize EMA
                ema_value = value
            else:
                # Update EMA: ema = alpha * current + (1 - alpha) * previous_ema
                ema_value = alpha * value + (1 - alpha) * ema_value
            smoothed.append(ema_value)

    return smoothed

# ───── Parse command line arguments ─────
parser = argparse.ArgumentParser(description='Visualize training and/or test results')
parser.add_argument('--data-path', type=str, required=True,
                    help='Data directory path containing task files (e.g., /data/shared).')
parser.add_argument('--log-path', type=str, required=True,
                    help='Log directory path containing trajectories (e.g., /data/exp1).')
parser.add_argument('--mode', type=str, choices=['train-only', 'test-only', 'train-test'],
                    default='train-test', help='Mode for visualization')
parser.add_argument('--ema', type=float, default=1.0,
                    help='EMA smoothing factor (0-1): 1.0=no smoothing, 0.0=max smoothing')
parser.add_argument('--run', type=str, default=None,
                    help='Specify which run to visualize (e.g., run_ablation_smalldomain). If not provided, uses base logs directory.')
args = parser.parse_args()

# Validate EMA parameter
if not 0.0 <= args.ema <= 1.0:
    print(f"Error: --ema must be between 0.0 and 1.0, got {args.ema}")
    exit(1)

# Validate data path
if not os.path.isdir(args.data_path):
    print(f"Error: Data directory does not exist: {args.data_path}")
    exit(1)

# Validate log path
if not os.path.isdir(args.log_path):
    print(f"Error: Log directory does not exist: {args.log_path}")
    exit(1)

data_path = args.data_path
base_logs_dir = args.log_path
print(f"📂 Using data directory: {data_path}")
print(f"📂 Using log directory: {base_logs_dir}")

if args.run:
    # Use specified run directory for trajectories
    logs_dir = os.path.join(base_logs_dir, args.run)

    # Validate that the run directory exists
    if not os.path.exists(logs_dir):
        print(f"Error: Run directory not found: {logs_dir}")
        print(f"Available runs in {base_logs_dir}:")
        if os.path.exists(base_logs_dir):
            runs = [d for d in os.listdir(base_logs_dir) if os.path.isdir(os.path.join(base_logs_dir, d)) and d.startswith('run_')]
            for run in sorted(runs):
                print(f"  - {run}")
        exit(1)

    print(f"📂 Using run directory for trajectories: {logs_dir}")
    print(f"📂 Using data directory for task files: {data_path}")
else:
    # Use base logs directory directly
    logs_dir = base_logs_dir
    print(f"📂 Using log directory for trajectories: {logs_dir}")
    print(f"📂 Using data directory for task files: {data_path}")

# if args.ema < 1.0:
#     print(f"📊 Applying EMA smoothing with alpha={args.ema}")

# ───── Load data based on mode ─────
train, train_diversity, train_succ_by_diff, train_samples_by_diff, train_chars_by_diff, train_steps_by_diff, train_goback_by_diff, train_same_screenshot_by_diff, train_same_screenshot_success_by_diff, train_crash_by_diff, train_tasks_per_iteration, train_websites_per_iteration, x_train = None, None, None, None, None, None, None, None, None, None, None, None, []
test_ood, test_ood_diversity, test_ood_succ_by_diff, test_ood_samples_by_diff, test_ood_chars_by_diff, test_ood_steps_by_diff, test_ood_goback_by_diff, test_ood_same_screenshot_by_diff, test_ood_same_screenshot_success_by_diff, test_ood_crash_by_diff, test_ood_step_limited_by_diff, test_ood_tasks_per_iteration, test_ood_websites_per_iteration, x_test_ood = None, None, None, None, None, None, None, None, None, None, None, None, None, []
n_train, n_test_ood = 0, 0

if args.mode in ['train-only', 'train-test']:
    # ───── 1) TRAIN ─────
    train, train_diversity, train_succ_by_diff, train_samples_by_diff, train_chars_by_diff, train_memory_chars_by_diff, train_steps_by_diff, train_goback_by_diff, train_same_screenshot_by_diff, train_same_screenshot_success_by_diff, train_crash_by_diff, train_tasks_per_iteration, train_websites_per_iteration, x_train = aggregate(logs_dir, data_path, 'train', is_train=True)
    n_train = len(train["succ"])
    # x_train is now cumulative trajectory counts from aggregate
    
    # Apply EMA smoothing to train data
    if args.ema < 1.0:
        for key in train.keys():
            train[key] = apply_ema(train[key], args.ema)
        for key in train_diversity.keys():
            train_diversity[key] = apply_ema(train_diversity[key], args.ema)
        for diff in train_succ_by_diff.keys():
            train_succ_by_diff[diff] = apply_ema(train_succ_by_diff[diff], args.ema)
        for diff in train_samples_by_diff.keys():
            train_samples_by_diff[diff] = apply_ema(train_samples_by_diff[diff], args.ema)
        for diff in train_chars_by_diff.keys():
            train_chars_by_diff[diff] = apply_ema(train_chars_by_diff[diff], args.ema)
        for diff in train_steps_by_diff.keys():
            train_steps_by_diff[diff] = apply_ema(train_steps_by_diff[diff], args.ema)
        for diff in train_goback_by_diff.keys():
            train_goback_by_diff[diff] = apply_ema(train_goback_by_diff[diff], args.ema)
        for diff in train_same_screenshot_success_by_diff.keys():
            train_same_screenshot_success_by_diff[diff] = apply_ema(train_same_screenshot_success_by_diff[diff], args.ema)
        for diff in train_crash_by_diff.keys():
            train_crash_by_diff[diff] = apply_ema(train_crash_by_diff[diff], args.ema)

if args.mode in ['test-only', 'train-test']:
    # ───── 2) TEST (OOD only) ─────
    # For test-only mode, we load train data for overlap metrics if available
    train_traj_dir = os.path.join(logs_dir, 'train_trajectories')
    if args.mode == 'test-only' and os.path.exists(train_traj_dir):
        _, _, train_succ_by_diff, train_samples_by_diff, train_chars_by_diff, train_memory_chars_by_diff, train_steps_by_diff, train_goback_by_diff, train_same_screenshot_by_diff, train_same_screenshot_success_by_diff, train_crash_by_diff, train_tasks_per_iteration, train_websites_per_iteration, _ = aggregate(logs_dir, data_path, 'train', is_train=True)
    elif args.mode == 'test-only':
        pass

    # Load test data (OOD only)
    ood_trajs_by_iter, test_iteration_counts = load_and_split_test_data(logs_dir, data_path)

    # Process OOD trajectories
    # We'll need to create a modified aggregate that works with pre-loaded trajs
    # For now, use distribution_filter approach (will need refactor for efficiency)
    test_ood, test_ood_diversity, test_ood_succ_by_diff, test_ood_samples_by_diff, test_ood_chars_by_diff, test_ood_memory_chars_by_diff, test_ood_steps_by_diff, test_ood_goback_by_diff, test_ood_same_screenshot_by_diff, test_ood_same_screenshot_success_by_diff, test_ood_crash_by_diff, test_ood_step_limited_by_diff, test_ood_tasks_per_iteration, test_ood_websites_per_iteration, x_test_ood = aggregate(logs_dir, data_path, 'test', is_train=False,
                                      train_tasks_per_iteration=train_tasks_per_iteration,
                                      train_websites_per_iteration=train_websites_per_iteration,
                                      distribution_filter='ood')
    n_test_ood = len(test_ood["succ"])

# Print task counts
if args.mode in ['train-only', 'train-test'] and train_tasks_per_iteration is not None:
    total_train_tasks = len(set().union(*train_tasks_per_iteration)) if train_tasks_per_iteration else 0
    print(f"Total unique train tasks: {total_train_tasks}")

if args.mode in ['test-only', 'train-test'] and test_ood_tasks_per_iteration is not None:
    total_ood_tasks = len(set().union(*test_ood_tasks_per_iteration)) if test_ood_tasks_per_iteration else 0
    print(f"Total unique OOD test tasks: {total_ood_tasks}")

if args.mode in ['test-only', 'train-test']:
    # Map test x-values to be evenly distributed across training x-range
    if args.mode == 'train-test' and x_train and len(x_train) > 0:
        # For OOD: evenly distribute test points from first to last train x value
        if x_test_ood and len(x_test_ood) > 0:
            n_test_ood_points = len(x_test_ood)
            if n_test_ood_points == 1:
                x_test_ood = [x_train[-1]]  # Single point at the end
            else:
                x_test_ood = [x_train[0] + (x_train[-1] - x_train[0]) * i / (n_test_ood_points - 1) for i in range(n_test_ood_points)]

    # Apply EMA smoothing to OOD test data
    if args.ema < 1.0:
        for key in test_ood.keys():
            test_ood[key] = apply_ema(test_ood[key], args.ema)
        for key in test_ood_diversity.keys():
            test_ood_diversity[key] = apply_ema(test_ood_diversity[key], args.ema)
        for diff in test_ood_succ_by_diff.keys():
            test_ood_succ_by_diff[diff] = apply_ema(test_ood_succ_by_diff[diff], args.ema)
        for diff in test_ood_samples_by_diff.keys():
            test_ood_samples_by_diff[diff] = apply_ema(test_ood_samples_by_diff[diff], args.ema)
        for diff in test_ood_chars_by_diff.keys():
            test_ood_chars_by_diff[diff] = apply_ema(test_ood_chars_by_diff[diff], args.ema)
        for diff in test_ood_steps_by_diff.keys():
            test_ood_steps_by_diff[diff] = apply_ema(test_ood_steps_by_diff[diff], args.ema)
        for diff in test_ood_goback_by_diff.keys():
            test_ood_goback_by_diff[diff] = apply_ema(test_ood_goback_by_diff[diff], args.ema)
        for diff in test_ood_same_screenshot_success_by_diff.keys():
            test_ood_same_screenshot_success_by_diff[diff] = apply_ema(test_ood_same_screenshot_success_by_diff[diff], args.ema)
        for diff in test_ood_crash_by_diff.keys():
            test_ood_crash_by_diff[diff] = apply_ema(test_ood_crash_by_diff[diff], args.ema)
        for diff in test_ood_step_limited_by_diff.keys():
            test_ood_step_limited_by_diff[diff] = apply_ema(test_ood_step_limited_by_diff[diff], args.ema)

# ───────────────────────────  PLOT  ────────────────────────────
# Layout: Row 1: Train Success, Train Chars, Train Steps, Train GoBack, Train Samples, Train Same-Screenshot, Train Same-Screenshot (Success)
#         Row 2: OOD Test Success, OOD Test Chars, OOD Test Steps, OOD Test GoBack, OOD Step-Limited Success, OOD Same-Screenshot, OOD Same-Screenshot (Success)
#         Row 3: # Tasks Seen Before, # Tasks with Websites Seen Before, Train Crash Rate, OOD Test Crash Rate, Block Rate Comparison
titles = ["Train Success by Difficulty","Train Avg. Chars Per Response","Train Avg. Steps","Train % GoBack Actions","Train Samples Collected by Difficulty","Train % Same Screenshot Steps","Train % Same Screenshot (Success)",
          "OOD Test Success by Difficulty","OOD Test Avg Chars Per Response","OOD Test Avg Steps","OOD Test % GoBack Actions","OOD Success Rate by Step Limit","OOD Test % Same Screenshot Steps","OOD Test % Same Screenshot (Success)",
          "# Tasks Seen Before","# Tasks with Websites Seen Before","Train Crash Rate by Difficulty","OOD Test Crash Rate","Block Rate Comparison"]
keys   = ["succ","act","steps","goback_by_diff","samples_by_diff","same_screenshot","same_screenshot_success",
          "ood_test_diff","ood_test_chars_by_diff","ood_test_steps_by_diff","ood_test_goback_by_diff","ood_step_limited","ood_same_screenshot","ood_same_screenshot_success",
          "tasks_overlap","websites_overlap","train_crashed","ood_test_crashed","block_comparison"]

# Determine if we should use markers (only when there's 1 iteration)
use_markers_train = len(x_train) == 1 if x_train else False
use_markers_test_ood = len(x_test_ood) == 1 if x_test_ood else False

# Compute # tasks with duplicate websites within each iteration
def count_tasks_with_duplicate_websites(tasks_per_iteration):
    """
    For each iteration, count how many tasks share a website with another task.

    Example: If iteration has task1(amazon), task2(amazon), task3(ebay),
    then count = 2 (both amazon tasks share the same website).
    """
    counts = []
    for tasks in tasks_per_iteration:
        # Extract website from each task_id (format: subdomain_website_difficulty_task_name)
        website_counts = {}
        for task_id in tasks:
            parts = task_id.split('_')
            if len(parts) >= 2:
                website = parts[1]  # website is the second component
                website_counts[website] = website_counts.get(website, 0) + 1

        # Count tasks where the website appears more than once
        duplicate_count = sum(count for count in website_counts.values() if count > 1)
        counts.append(duplicate_count)

    return counts

train_num_tasks_with_dup_websites = None
if train_tasks_per_iteration is not None:
    train_num_tasks_with_dup_websites = count_tasks_with_duplicate_websites(train_tasks_per_iteration)

test_ood_num_tasks_with_dup_websites = None
if test_ood_tasks_per_iteration is not None:
    test_ood_num_tasks_with_dup_websites = count_tasks_with_duplicate_websites(test_ood_tasks_per_iteration)

# 3x8 grid (Row 1: Train, Row 2: OOD Test, Row 3: Misc metrics)
fig, axes = plt.subplots(3,8,figsize=(64,18),sharex=True); axes = axes.flatten()

# Predefine colors for difficulty groups: easy (1-3), medium (4-6), hard (7+)
DIFFICULTY_COLORS = {
    'easy': '#2ca02c',    # green
    'medium': '#ff7f0e',  # orange
    'hard': '#d62728',    # red
}

def get_difficulty_group(diff):
    """Map individual difficulty (1-7+) to group (easy/medium/hard)"""
    if diff <= 3:
        return 'easy'
    elif diff <= 6:
        return 'medium'
    else:
        return 'hard'

def aggregate_by_difficulty_group(by_diff_dict, samples_by_diff_dict=None):
    """
    Aggregate per-difficulty metrics into easy/medium/hard groups.

    For success rates and other ratios, we compute weighted average using sample counts.
    For sample counts, we sum them up.

    Args:
        by_diff_dict: Dict mapping difficulty -> list of values per iteration
        samples_by_diff_dict: Optional dict mapping difficulty -> list of sample counts per iteration
                              (used for weighted averaging of rates)

    Returns:
        Dict mapping 'easy'/'medium'/'hard' -> list of aggregated values per iteration
    """
    if not by_diff_dict:
        return {}

    # Get the number of iterations from any difficulty's data
    n_iterations = max(len(v) for v in by_diff_dict.values()) if by_diff_dict else 0

    # Initialize result with groups
    result = {'easy': [], 'medium': [], 'hard': []}

    for iter_idx in range(n_iterations):
        for group in ['easy', 'medium', 'hard']:
            # Collect values and weights for this group at this iteration
            values = []
            weights = []

            for diff, vals in by_diff_dict.items():
                if get_difficulty_group(diff) == group and iter_idx < len(vals):
                    val = vals[iter_idx]
                    if val is not None:
                        values.append(val)
                        # Get weight (sample count) if available
                        if samples_by_diff_dict and diff in samples_by_diff_dict:
                            weight = samples_by_diff_dict[diff][iter_idx] if iter_idx < len(samples_by_diff_dict[diff]) else 1
                            weights.append(weight if weight else 1)
                        else:
                            weights.append(1)

            if values:
                # Weighted average
                total_weight = sum(weights)
                if total_weight > 0:
                    weighted_avg = sum(v * w for v, w in zip(values, weights)) / total_weight
                    result[group].append(weighted_avg)
                else:
                    result[group].append(None)
            else:
                result[group].append(None)

    return result

def aggregate_samples_by_difficulty_group(samples_by_diff_dict):
    """
    Aggregate sample counts by difficulty group (sum instead of average).
    """
    if not samples_by_diff_dict:
        return {}

    n_iterations = max(len(v) for v in samples_by_diff_dict.values()) if samples_by_diff_dict else 0

    result = {'easy': [], 'medium': [], 'hard': []}

    for iter_idx in range(n_iterations):
        for group in ['easy', 'medium', 'hard']:
            total = 0
            for diff, counts in samples_by_diff_dict.items():
                if get_difficulty_group(diff) == group and iter_idx < len(counts):
                    total += counts[iter_idx] if counts[iter_idx] else 0
            result[group].append(total)

    return result

def plot_by_difficulty_group(ax, x_vals_full, by_diff_dict, samples_by_diff_dict=None, is_sample_count=False):
    """
    Helper to plot metrics aggregated by difficulty group (easy/medium/hard).

    Args:
        ax: matplotlib axis
        x_vals_full: full x-axis values
        by_diff_dict: dict mapping difficulty -> list of values
        samples_by_diff_dict: optional dict for weighted averaging
        is_sample_count: if True, sum instead of average
    """
    if not by_diff_dict:
        return

    if is_sample_count:
        by_group = aggregate_samples_by_difficulty_group(by_diff_dict)
    else:
        by_group = aggregate_by_difficulty_group(by_diff_dict, samples_by_diff_dict)

    for group in ['easy', 'medium', 'hard']:
        if group in by_group:
            values = by_group[group]
            x_vals = []
            y_vals = []
            for i, val in enumerate(values):
                if val is not None and i < len(x_vals_full):
                    x_vals.append(x_vals_full[i])
                    y_vals.append(val)
            if x_vals:
                marker_args = {"marker": "o", "markersize": 3} if len(x_vals) == 1 else {}
                label = f"{group.capitalize()} (1-3)" if group == 'easy' else f"{group.capitalize()} (4-6)" if group == 'medium' else f"{group.capitalize()} (7+)"
                ax.plot(x_vals, y_vals, lw=3, **marker_args, label=label, color=DIFFICULTY_COLORS.get(group, 'gray'))

# Plot row 1 performance metrics (positions 0-2)
# Position 0: Train Success by Difficulty (moved from row 4)
ax = axes[0]
# Plot overall train success rate first (in black with thicker line)
if args.mode in ['train-only', 'train-test'] and train is not None:
    marker_args = {"marker": "o", "markersize": 8} if use_markers_train else {}
    ax.plot(x_train, train["succ"], lw=4, **marker_args, label="Overall", color="black", zorder=10)
# Plot difficulty-group-wise train success rates (easy/medium/hard)
if args.mode in ['train-only', 'train-test'] and train_succ_by_diff is not None:
    train_succ_by_group = aggregate_by_difficulty_group(train_succ_by_diff, train_samples_by_diff)
    for group in ['easy', 'medium', 'hard']:
        if group in train_succ_by_group:
            success_rates = train_succ_by_group[group]
            # Filter out None values for plotting
            x_vals = []
            y_vals = []
            for i, rate in enumerate(success_rates):
                if rate is not None and i < len(x_train):
                    x_vals.append(x_train[i])
                    y_vals.append(rate)
            if x_vals:  # Only plot if there's data
                marker_args = {"marker": "o", "markersize": 3} if (len(x_vals) == 1) else {}
                label = f"{group.capitalize()} (1-3)" if group == 'easy' else f"{group.capitalize()} (4-6)" if group == 'medium' else f"{group.capitalize()} (7+)"
                ax.plot(x_vals, y_vals, lw=3, **marker_args, label=label, color=DIFFICULTY_COLORS.get(group, 'gray'))
ax.set_title("Train Success by Difficulty", fontsize=14)
ax.set_xlabel("Training trajectories processed")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Position 1: Train Avg Chars Per Response (overall + by difficulty)
ax = axes[1]
if args.mode in ['train-only', 'train-test'] and train is not None:
    # Plot overall train avg chars (in black with thicker line)
    marker_args = {"marker": "o", "markersize": 4} if use_markers_train else {}

    ax.plot(x_train, train["act"], lw=4, **marker_args, label="Overall", color="black", zorder=10)
if args.mode in ['train-only', 'train-test'] and train_chars_by_diff is not None:
    plot_by_difficulty_group(ax, x_train, train_chars_by_diff, train_samples_by_diff)
ax.set_title("Train Avg. Chars Per Response", fontsize=14)
ax.set_xlabel("Training trajectories processed")
ax.set_ylabel("Avg chars per response")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Position 2: Train Avg Memory Chars Per Response (overall + by difficulty)
ax = axes[2]
if args.mode in ['train-only', 'train-test'] and train is not None:
    # Plot overall train avg memory chars (in black with thicker line)
    marker_args = {"marker": "o", "markersize": 4} if use_markers_train else {}

    ax.plot(x_train, train["memory_chars"], lw=4, **marker_args, label="Overall", color="black", zorder=10)
if args.mode in ['train-only', 'train-test'] and train_memory_chars_by_diff is not None:
    plot_by_difficulty_group(ax, x_train, train_memory_chars_by_diff, train_samples_by_diff)
ax.set_title("Train Avg. Memory Chars", fontsize=14)
ax.set_xlabel("Training trajectories processed")
ax.set_ylabel("Avg memory chars per response")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Position 3: Train Avg Steps (overall + by difficulty)
ax = axes[3]
if args.mode in ['train-only', 'train-test'] and train is not None:
    # Plot overall train avg steps (in black with thicker line)
    marker_args = {"marker": "o", "markersize": 4} if use_markers_train else {}

    ax.plot(x_train, train["steps"], lw=4, **marker_args, label="Overall", color="black", zorder=10)
if args.mode in ['train-only', 'train-test'] and train_steps_by_diff is not None:
    plot_by_difficulty_group(ax, x_train, train_steps_by_diff, train_samples_by_diff)
ax.set_title("Train Avg. Steps", fontsize=14)
ax.set_xlabel("Training trajectories processed")
ax.set_ylabel("Avg steps per trajectory")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Calculate test set difficulty counts for the legend
test_ood_difficulty_counts = {}
if args.mode in ['test-only', 'train-test']:
    if test_ood_samples_by_diff is not None:
        for diff, sample_counts in test_ood_samples_by_diff.items():
            # Test set is fixed, so we can take any iteration's count (they should all be the same)
            # Take the first iteration's count
            test_ood_difficulty_counts[diff] = sample_counts[0] if sample_counts else 0

# Position 4: Train GoBack Actions by Difficulty
ax = axes[4]
# Plot overall train goback rate (in black with thicker line)
if args.mode in ['train-only', 'train-test'] and train is not None:
    marker_args = {"marker": "o", "markersize": 4} if use_markers_train else {}
if args.mode in ['train-only', 'train-test'] and train is not None:
    marker_args = {"marker": "o", "markersize": 4} if use_markers_train else {}

    ax.plot(x_train, train["goback"], lw=4, **marker_args, label="Overall", color="black", zorder=10)
# Plot difficulty-wise train goback rates
if args.mode in ['train-only', 'train-test'] and train_goback_by_diff is not None:
    plot_by_difficulty_group(ax, x_train, train_goback_by_diff, train_samples_by_diff)
ax.set_title("Train % GoBack Actions", fontsize=14)
ax.set_xlabel("Training trajectories processed")
ax.set_ylabel("GoBack rate")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Position 5: Train Samples Collected by Difficulty
ax = axes[5]
if args.mode in ['train-only', 'train-test'] and train_samples_by_diff is not None:
    plot_by_difficulty_group(ax, x_train, train_samples_by_diff, is_sample_count=True)
ax.set_title("Train Samples Collected by Difficulty", fontsize=14)
ax.set_xlabel("Training trajectories processed")
ax.set_ylabel("Number of samples")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Position 6: Train Same-Screenshot Ratio by Difficulty
ax = axes[6]
# Plot overall train same-screenshot rate (in black with thicker line)
if args.mode in ['train-only', 'train-test'] and train is not None:
    marker_args = {"marker": "o", "markersize": 4} if use_markers_train else {}
if args.mode in ['train-only', 'train-test'] and train is not None:
    marker_args = {"marker": "o", "markersize": 4} if use_markers_train else {}

    ax.plot(x_train, train["same_screenshot"], lw=4, **marker_args, label="Overall", color="black", zorder=10)
# Plot difficulty-wise train same-screenshot rates
if args.mode in ['train-only', 'train-test'] and train_same_screenshot_by_diff is not None:
    plot_by_difficulty_group(ax, x_train, train_same_screenshot_by_diff, train_samples_by_diff)
ax.set_title("Train % Same Screenshot Steps", fontsize=14)
ax.set_xlabel("Training trajectories processed")
ax.set_ylabel("Same-screenshot rate")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Position 7: Train Same-Screenshot Ratio (Success) by Difficulty
ax = axes[7]
# Plot overall train same-screenshot rate for successful trajectories (in black with thicker line)
if args.mode in ['train-only', 'train-test'] and train is not None:
    marker_args = {"marker": "o", "markersize": 4} if use_markers_train else {}
if args.mode in ['train-only', 'train-test'] and train is not None:
    marker_args = {"marker": "o", "markersize": 4} if use_markers_train else {}

    ax.plot(x_train, train["same_screenshot_success"], lw=4, **marker_args, label="Overall", color="black", zorder=10)
# Plot difficulty-wise train same-screenshot rates for successful trajectories
if args.mode in ['train-only', 'train-test'] and train_same_screenshot_success_by_diff is not None:
    plot_by_difficulty_group(ax, x_train, train_same_screenshot_success_by_diff, train_samples_by_diff)
ax.set_title("Train % Same Screenshot (Success)", fontsize=14)
ax.set_xlabel("Training trajectories processed")
ax.set_ylabel("Same-screenshot rate (success only)")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Plot row 2 col 1: OOD Test Success by Difficulty (position 8)
ax = axes[8]
# Plot overall OOD test success rate first (in black with thicker line)
if args.mode in ['test-only', 'train-test'] and test_ood is not None:
    marker_args = {"marker": "o", "markersize": 4} if use_markers_test_ood else {}

    ax.plot(x_test_ood, test_ood["succ"], lw=4, **marker_args, label="Overall", color="black", zorder=10)
# Plot difficulty-wise OOD test success rates
if args.mode in ['test-only', 'train-test'] and test_ood_succ_by_diff is not None:
    plot_by_difficulty_group(ax, x_test_ood, test_ood_succ_by_diff, test_ood_samples_by_diff)
ax.set_title("OOD Test Success by Difficulty", fontsize=14)
ax.set_xlabel("Training trajectories processed")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Plot row 2 col 2: OOD Test Avg Chars Per Response by Difficulty (position 17)
ax = axes[9]
# Plot overall OOD test avg chars first (in black with thicker line)
if args.mode in ['test-only', 'train-test'] and test_ood is not None:
    marker_args = {"marker": "o", "markersize": 4} if use_markers_test_ood else {}

    ax.plot(x_test_ood, test_ood["act"], lw=4, **marker_args, label="Overall", color="black", zorder=10)
# Plot difficulty-wise OOD test avg chars
if args.mode in ['test-only', 'train-test'] and test_ood_chars_by_diff is not None:
    plot_by_difficulty_group(ax, x_test_ood, test_ood_chars_by_diff, test_ood_samples_by_diff)
ax.set_title("OOD Test Avg. Chars Per Response", fontsize=14)
ax.set_xlabel("Training trajectories processed")
ax.set_ylabel("Avg chars per response")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Plot row 2 col 3: OOD Test Avg Memory Chars by Difficulty (position 18)
ax = axes[10]
# Plot overall OOD test avg memory chars first (in black with thicker line)
if args.mode in ['test-only', 'train-test'] and test_ood is not None:
    marker_args = {"marker": "o", "markersize": 4} if use_markers_test_ood else {}

    ax.plot(x_test_ood, test_ood["memory_chars"], lw=4, **marker_args, label="Overall", color="black", zorder=10)
# Plot difficulty-wise OOD test avg memory chars
if args.mode in ['test-only', 'train-test'] and test_ood_memory_chars_by_diff is not None:
    plot_by_difficulty_group(ax, x_test_ood, test_ood_memory_chars_by_diff, test_ood_samples_by_diff)
ax.set_title("OOD Test Avg. Memory Chars", fontsize=14)
ax.set_xlabel("Training trajectories processed")
ax.set_ylabel("Avg memory chars per response")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Plot row 2 col 4: OOD Test Avg Steps by Difficulty (position 19)
ax = axes[11]
# Plot overall OOD test avg steps first (in black with thicker line)
if args.mode in ['test-only', 'train-test'] and test_ood is not None:
    marker_args = {"marker": "o", "markersize": 4} if use_markers_test_ood else {}

    ax.plot(x_test_ood, test_ood["steps"], lw=4, **marker_args, label="Overall", color="black", zorder=10)
# Plot difficulty-wise OOD test avg steps
if args.mode in ['test-only', 'train-test'] and test_ood_steps_by_diff is not None:
    plot_by_difficulty_group(ax, x_test_ood, test_ood_steps_by_diff, test_ood_samples_by_diff)
ax.set_title("OOD Test Avg. Steps", fontsize=14)
ax.set_xlabel("Training trajectories processed")
ax.set_ylabel("Avg steps per trajectory")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Plot row 2 col 5: OOD Test GoBack Actions by Difficulty (position 20)
ax = axes[12]
# Plot overall OOD test goback rate (in black with thicker line)
if args.mode in ['test-only', 'train-test'] and test_ood is not None:
    marker_args = {"marker": "o", "markersize": 4} if use_markers_test_ood else {}

    ax.plot(x_test_ood, test_ood["goback"], lw=4, **marker_args, label="Overall", color="black", zorder=10)
# Plot difficulty-wise OOD test goback rates
if args.mode in ['test-only', 'train-test'] and test_ood_goback_by_diff is not None:
    plot_by_difficulty_group(ax, x_test_ood, test_ood_goback_by_diff, test_ood_samples_by_diff)
ax.set_title("OOD Test % GoBack Actions", fontsize=14)
ax.set_xlabel("Training trajectories processed")
ax.set_ylabel("GoBack rate")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Plot row 2 col 6: OOD Success Rate by Step Limit (position 21)
ax = axes[13]
if args.mode in ['test-only', 'train-test'] and test_ood is not None:
    # Plot overall OOD step-limited success rate first (in black with thicker line)
    marker_args = {"marker": "o", "markersize": 4} if use_markers_test_ood else {}

    ax.plot(x_test_ood, test_ood["step_limited_success"], lw=4, **marker_args, label="Overall", color="black", zorder=10)

if args.mode in ['test-only', 'train-test'] and test_ood_step_limited_by_diff is not None:
    plot_by_difficulty_group(ax, x_test_ood, test_ood_step_limited_by_diff, test_ood_samples_by_diff)

ax.set_title("OOD Success Rate by Step Limit (Test)", fontsize=14)
ax.set_xlabel("Training trajectories processed")
ax.set_ylabel("Success rate")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend(fontsize=10, loc='best')
ax.grid(True)

# Plot row 2 col 7: OOD Test Same-Screenshot Ratio by Difficulty (position 22)
ax = axes[14]
# Plot overall OOD test same-screenshot rate (in black with thicker line)
if args.mode in ['test-only', 'train-test'] and test_ood is not None:
    marker_args = {"marker": "o", "markersize": 4} if use_markers_test_ood else {}

    ax.plot(x_test_ood, test_ood["same_screenshot"], lw=4, **marker_args, label="Overall", color="black", zorder=10)
# Plot difficulty-wise OOD test same-screenshot rates
if args.mode in ['test-only', 'train-test'] and test_ood_same_screenshot_by_diff is not None:
    plot_by_difficulty_group(ax, x_test_ood, test_ood_same_screenshot_by_diff, test_ood_samples_by_diff)
ax.set_title("OOD Test % Same Screenshot Steps", fontsize=14)
ax.set_xlabel("Training trajectories processed")
ax.set_ylabel("Same-screenshot rate")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Plot row 2 col 8: OOD Test Same-Screenshot Ratio (Success) by Difficulty (position 23)
ax = axes[15]
# Plot overall OOD test same-screenshot rate for successful trajectories (in black with thicker line)
if args.mode in ['test-only', 'train-test'] and test_ood is not None:
    marker_args = {"marker": "o", "markersize": 4} if use_markers_test_ood else {}

    ax.plot(x_test_ood, test_ood["same_screenshot_success"], lw=4, **marker_args, label="Overall", color="black", zorder=10)
# Plot difficulty-wise OOD test same-screenshot rates for successful trajectories
if args.mode in ['test-only', 'train-test'] and test_ood_same_screenshot_success_by_diff is not None:
    plot_by_difficulty_group(ax, x_test_ood, test_ood_same_screenshot_success_by_diff, test_ood_samples_by_diff)
ax.set_title("OOD Test % Same Screenshot (Success)", fontsize=14)
ax.set_xlabel("Training trajectories processed")
ax.set_ylabel("Same-screenshot rate (success only)")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Plot row 3 col 1: # Tasks Seen Before (position 24)
ax = axes[16]
if args.mode in ['train-only', 'train-test'] and train_diversity is not None:
    marker_args = {"marker": "o", "markersize": 3} if use_markers_train else {}
if args.mode in ['train-only', 'train-test'] and train_diversity is not None:
    marker_args = {"marker": "o", "markersize": 3} if use_markers_train else {}

    ax.plot(x_train, train_diversity["tasks_overlap"], lw=3, **marker_args, label="Train", color="#1f77b4")
if args.mode in ['test-only', 'train-test'] and test_ood_diversity is not None:
    marker_args = {"marker": "o", "markersize": 3} if use_markers_test_ood else {}

    ax.plot(x_test_ood, test_ood_diversity["tasks_overlap"], lw=3, **marker_args, label="OOD Test",  color="#d62728")
ax.set_title("# Tasks Seen Before", fontsize=14)
ax.set_xlabel("Training trajectories processed")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Plot row 3 col 2: # Tasks with Websites Seen Before (position 25)
ax = axes[17]
if args.mode in ['train-only', 'train-test'] and train_diversity is not None:
    marker_args = {"marker": "o", "markersize": 3} if use_markers_train else {}
    if "tasks_with_seen_websites" in train_diversity:
        ax.plot(x_train, train_diversity["tasks_with_seen_websites"], lw=3, **marker_args, label="Train", color="#1f77b4")
if args.mode in ['test-only', 'train-test'] and test_ood_diversity is not None:
    marker_args = {"marker": "o", "markersize": 3} if use_markers_test_ood else {}
    if "tasks_with_seen_websites" in test_ood_diversity:
        ax.plot(x_test_ood, test_ood_diversity["tasks_with_seen_websites"], lw=3, **marker_args, label="OOD Test",  color="#d62728")
ax.set_title("# Tasks with Websites Seen Before", fontsize=14)
ax.set_xlabel("Training trajectories processed")
ax.set_ylabel("Number of tasks")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Plot row 3 col 3: # Tasks with Duplicate Websites (NEW - position 26)
ax = axes[18]
if args.mode in ['train-only', 'train-test'] and train_num_tasks_with_dup_websites is not None and len(train_num_tasks_with_dup_websites) > 0:
    marker_args = {"marker": "o", "markersize": 3} if use_markers_train else {}
    ax.plot(x_train, train_num_tasks_with_dup_websites, lw=3, **marker_args, label="Train", color="#1f77b4")
if args.mode in ['test-only', 'train-test'] and test_ood_num_tasks_with_dup_websites is not None and len(test_ood_num_tasks_with_dup_websites) > 0:
    marker_args = {"marker": "o", "markersize": 3} if use_markers_test_ood else {}
    ax.plot(x_test_ood, test_ood_num_tasks_with_dup_websites, lw=3, **marker_args, label="OOD Test", color="#d62728")
ax.set_title("# Tasks with Duplicate Websites", fontsize=14)
ax.set_xlabel("Training trajectories processed")
ax.set_ylabel("Number of tasks")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Plot row 3 col 4: Train Crash Rate by Difficulty (position 27)
ax = axes[19]
# Plot overall train crash rate first (in black with thicker line)
if args.mode in ['train-only', 'train-test'] and train is not None:
    marker_args = {"marker": "o", "markersize": 4} if use_markers_train else {}
if args.mode in ['train-only', 'train-test'] and train is not None:
    marker_args = {"marker": "o", "markersize": 4} if use_markers_train else {}

    ax.plot(x_train, train["crashed"], lw=4, **marker_args, label="Overall", color="black", zorder=10)
# Plot difficulty-wise train crash rates
if args.mode in ['train-only', 'train-test'] and train_crash_by_diff is not None:
    plot_by_difficulty_group(ax, x_train, train_crash_by_diff, train_samples_by_diff)
ax.set_title("Train Crash Rate by Difficulty", fontsize=14)
ax.set_xlabel("Training trajectories processed")
ax.set_ylabel("Crash rate")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Plot row 3 col 5: OOD Test Crash Rate by Difficulty (position 28)
ax = axes[20]
# Plot overall OOD test crash rate first (in black with thicker line)
if args.mode in ['test-only', 'train-test'] and test_ood is not None:
    marker_args = {"marker": "o", "markersize": 4} if use_markers_test_ood else {}

    ax.plot(x_test_ood, test_ood["crashed"], lw=4, **marker_args, label="Overall", color="black", zorder=10)
# Plot difficulty-wise test crash rates
if args.mode in ['test-only', 'train-test'] and test_ood_crash_by_diff is not None:
    plot_by_difficulty_group(ax, x_test_ood, test_ood_crash_by_diff, test_ood_samples_by_diff)
ax.set_title("OOD Test Crash Rate by Difficulty", fontsize=14)
ax.set_xlabel("Training trajectories processed")
ax.set_ylabel("Crash rate")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# Plot row 2 col 6: Block Rate Comparison (position 21)
ax = axes[21]
if args.mode in ['train-only', 'train-test'] and train is not None:
    marker_args = {"marker": "o", "markersize": 3} if use_markers_train else {}
if args.mode in ['train-only', 'train-test'] and train is not None:
    marker_args = {"marker": "o", "markersize": 3} if use_markers_train else {}

    ax.plot(x_train, train["blocked"], lw=3, **marker_args, label="Train", color="#1f77b4")
if args.mode in ['test-only', 'train-test'] and test_ood is not None:
    marker_args = {"marker": "o", "markersize": 3} if use_markers_test_ood else {}

    ax.plot(x_test_ood, test_ood["blocked"], lw=3, **marker_args, label="OOD Test", color="#d62728")
ax.set_title("Block Rate Comparison", fontsize=14)
ax.set_xlabel("Training trajectories processed")
ax.set_ylabel("Block rate")
if ax.get_lines():  # Only show legend if there are lines plotted
    ax.legend()
ax.grid(True)

# No figure title (removed as requested)

# Always use the same filename regardless of EMA value
output_filename = "metrics.png"

# Set the same x-axis range for all plots (use train x-axis range with padding)
if x_train and len(x_train) > 0:
    x_min = 0
    x_max = x_train[-1]
    # Add 2% padding on the right to keep endpoints inside the plot
    padding = (x_max - x_min) * 0.02
    for ax in axes.flat:
        ax.set_xlim(x_min, x_max + padding)

fig.tight_layout()
plt.savefig(output_filename)
print(f"Saved plot to {output_filename}")
