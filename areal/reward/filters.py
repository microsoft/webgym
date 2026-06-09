"""Trajectory accept/reject filters for rollout-side filtering.

These functions are passed as `should_accept_fn` string paths to
`prepare_batch()` and evaluated on rollout workers after `arun_episode`.
"""


def fbc_accept(traj: dict) -> bool:
    """Accept trajectory only if reward > 0 (Filtered Behavioral Cloning)."""
    if "rewards" in traj:
        return bool((traj["rewards"] > 0).any())
    return True


def grpo_accept(traj: dict) -> bool:
    """Accept grouped trajectory only if rewards have variance (GRPO).

    Rejects groups where all episodes are all-correct or all-incorrect,
    since these produce zero advantage after group normalization.
    For single-episode trajectories (n_samples=1), always accepts.
    """
    if "rewards" not in traj or "episode_id" not in traj:
        return True
    rewards = traj["rewards"]
    episode_ids = traj["episode_id"]
    # Extract one reward per unique episode
    unique_eps = episode_ids.unique()
    if unique_eps.numel() <= 1:
        return True  # Single episode, no group filtering
    ep_rewards = []
    for eid in unique_eps:
        mask = episode_ids == eid
        ep_rewards.append(float(rewards[mask][0].item()))
    # Reject if all same (zero variance → no GRPO signal)
    return len(set(ep_rewards)) > 1
