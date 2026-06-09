"""Load AReaL JSON trajectory dumps from test-rollout and rollout directories."""

import glob
import json
import os
import re
from collections import defaultdict

# Step-limit thresholds per difficulty level (from reference visualize_results.py)
DIFFICULTY_STEP_THRESHOLD = {1: 5, 2: 8, 3: 10, 4: 15, 5: 20, 6: 30, 7: 50}


def _get_difficulty_group(diff: int) -> str:
    if diff <= 3:
        return "easy"
    elif diff <= 6:
        return "medium"
    return "hard"


def _is_crashed(steps: list[dict]) -> bool:
    """Detect crashed episode using action-key heuristics (mirrors reference)."""
    if not steps:
        return True
    for step in steps:
        act = step.get("action")
        if act is None:
            return True
        action_dict = act.get("action")
        if action_dict is None:
            return True
        key = action_dict.get("key", "")
        if key in ("invalid_step", "error"):
            return True
    # Premature termination: only 1 step and not a submit
    if len(steps) == 1:
        act = steps[0].get("action", {})
        action_dict = act.get("action", {})
        if action_dict.get("key") != "submit":
            return True
    return False


def _extract_step_metrics(steps: list[dict]) -> dict:
    """Extract per-step aggregate metrics from step data."""
    total_chars = 0
    total_memory_chars = 0
    n_goback = 0
    n_steps = len(steps)
    for step in steps:
        # Chars per response
        resp = step.get("response")
        if resp and isinstance(resp, dict):
            raw = resp.get("raw_response", "")
            total_chars += len(raw)
            # Memory chars: look for memory section patterns
            # Format 1: "Memory: {...}" (current webgym format)
            mem_match = re.search(r"^Memory:\s*(\{.*?\})\s*$", raw, re.MULTILINE)
            # Format 2: "<memory>...</memory>" (legacy format)
            if not mem_match:
                mem_match = re.search(r"<memory>(.*?)</memory>", raw, re.DOTALL)
            if mem_match:
                total_memory_chars += len(mem_match.group(1))
        # GoBack action
        act = step.get("action")
        if act and isinstance(act, dict):
            action_dict = act.get("action")
            if action_dict and isinstance(action_dict, dict):
                if action_dict.get("key") == "goback":
                    n_goback += 1
    return {
        "avg_chars": total_chars / n_steps if n_steps else 0,
        "avg_memory_chars": total_memory_chars / n_steps if n_steps else 0,
        "goback_rate": n_goback / n_steps if n_steps else 0,
    }


def _load_version_dir(version_dir: str) -> list[dict]:
    """Load all JSON episode files from a single version directory."""
    episodes = []
    for filepath in glob.glob(os.path.join(version_dir, "*.json")):
        try:
            with open(filepath) as f:
                data = json.load(f)
            ti = data.get("task_info", {})
            diff = ti.get("difficulty")
            if diff is not None:
                try:
                    diff = int(diff)
                except (ValueError, TypeError):
                    diff = 5
            else:
                diff = 5
            reward = float(data.get("reward", 0))
            n_steps = int(data.get("n_steps", 0))
            steps = data.get("steps", [])
            step_metrics = _extract_step_metrics(steps)
            is_crashed = _is_crashed(steps)
            # Step-limited success: success within difficulty threshold
            threshold = DIFFICULTY_STEP_THRESHOLD.get(diff, 50)
            step_limited_success = reward > 0 and n_steps <= threshold
            episodes.append(
                {
                    "task_id": data.get("task_id", ""),
                    "reward": reward,
                    "n_steps": n_steps,
                    "difficulty": diff,
                    "difficulty_group": _get_difficulty_group(diff),
                    "is_blocked": bool(data.get("is_blocked", False)),
                    "website": ti.get("website", ""),
                    "task_name": ti.get("task_name", ""),
                    "steps": steps,
                    "avg_chars": step_metrics["avg_chars"],
                    "avg_memory_chars": step_metrics["avg_memory_chars"],
                    "goback_rate": step_metrics["goback_rate"],
                    "is_crashed": is_crashed,
                    "step_limited_success": step_limited_success,
                }
            )
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return episodes


def load_rollout(logs_dir: str, split: str = "test") -> list[dict]:
    """Load all versions of rollout trajectories.

    Args:
        logs_dir: Trial directory (e.g., .../trial0)
        split: "test" for test-rollout, "train" for rollout

    Returns:
        List of version dicts sorted by version number, each containing:
        {
            "version": int,
            "episodes": [episode_dict, ...],
            "stats": {computed aggregate stats}
        }
    """
    subdir = "test-rollout" if split == "test" else "rollout"
    rollout_dir = os.path.join(logs_dir, subdir)
    if not os.path.isdir(rollout_dir):
        print(f"Directory not found: {rollout_dir}")
        return []

    versions = []
    for entry in sorted(os.listdir(rollout_dir)):
        version_path = os.path.join(rollout_dir, entry)
        if not os.path.isdir(version_path):
            continue
        try:
            version_num = int(entry)
        except ValueError:
            continue

        episodes = _load_version_dir(version_path)
        if not episodes:
            continue

        # Compute aggregate stats
        non_blocked = [e for e in episodes if not e["is_blocked"]]
        n_total = len(episodes)
        n_valid = len(non_blocked)
        n_blocked = n_total - n_valid

        overall_reward = sum(e["reward"] for e in non_blocked) / n_valid if n_valid else 0
        overall_success = sum(1 for e in non_blocked if e["reward"] > 0) / n_valid if n_valid else 0
        overall_steps = sum(e["n_steps"] for e in non_blocked) / n_valid if n_valid else 0
        n_crashed = sum(1 for e in non_blocked if e["is_crashed"])
        overall_chars = sum(e["avg_chars"] for e in non_blocked) / n_valid if n_valid else 0
        overall_memory_chars = sum(e["avg_memory_chars"] for e in non_blocked) / n_valid if n_valid else 0
        overall_goback = sum(e["goback_rate"] for e in non_blocked) / n_valid if n_valid else 0
        overall_step_limited = (
            sum(1 for e in non_blocked if e["step_limited_success"]) / n_valid
            if n_valid
            else 0
        )

        # Per-difficulty group
        by_group = defaultdict(list)
        for e in non_blocked:
            by_group[e["difficulty_group"]].append(e)

        group_stats = {}
        for group in ["easy", "medium", "hard"]:
            eps = by_group.get(group, [])
            if eps:
                group_stats[group] = {
                    "reward": sum(e["reward"] for e in eps) / len(eps),
                    "success_rate": sum(1 for e in eps if e["reward"] > 0) / len(eps),
                    "avg_steps": sum(e["n_steps"] for e in eps) / len(eps),
                    "n_tasks": len(eps),
                    "n_crashed": sum(1 for e in eps if e["is_crashed"]),
                    "crash_rate": sum(1 for e in eps if e["is_crashed"]) / len(eps),
                    "avg_chars": sum(e["avg_chars"] for e in eps) / len(eps),
                    "avg_memory_chars": sum(e["avg_memory_chars"] for e in eps) / len(eps),
                    "goback_rate": sum(e["goback_rate"] for e in eps) / len(eps),
                    "step_limited_success": (
                        sum(1 for e in eps if e["step_limited_success"]) / len(eps)
                    ),
                }

        versions.append(
            {
                "version": version_num,
                "episodes": episodes,
                "stats": {
                    "n_total": n_total,
                    "n_valid": n_valid,
                    "n_blocked": n_blocked,
                    "n_crashed": n_crashed,
                    "overall_reward": overall_reward,
                    "overall_success": overall_success,
                    "overall_steps": overall_steps,
                    "overall_chars": overall_chars,
                    "overall_memory_chars": overall_memory_chars,
                    "overall_goback": overall_goback,
                    "overall_step_limited": overall_step_limited,
                    "blocked_rate": n_blocked / n_total if n_total else 0,
                    "crash_rate": n_crashed / n_valid if n_valid else 0,
                    "by_group": group_stats,
                },
            }
        )

    versions.sort(key=lambda v: v["version"])
    return versions
