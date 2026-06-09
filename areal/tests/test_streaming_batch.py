"""Benchmark: concat-then-split vs streaming per-microbatch assembly.

The old pipeline:
  1. concat_padded_tensors(504 trajs) → [504, max_seq_len]  (~45 min for large batches)
  2. split_padded_tensor_dict_into_mb_list → reindex [504, max_seq_len] → list of [n_i, max_seq_len]
  3. pack_tensor_dict per MB → remove padding → [total_tokens_i, ...]

The new pipeline (streaming):
  1. Extract seq_lens from each trajectory (O(N) metadata scan, instant)
  2. FFD bin-packing on seq_lens → group_indices (instant)
  3. For each microbatch group: concat only those 2-5 trajs, pack → [total_tokens_i, ...]

The key insight: concat_padded_tensors pads ALL sequences to the global max_seq_len,
then split immediately undoes this by re-grouping. Streaming skips the global pad entirely.

Usage:
  python -m pytest areal/tests/test_streaming_batch.py -v -s
  python areal/tests/test_streaming_batch.py  # direct run for quick benchmark
"""

import time

import numpy as np
import torch

from areal.utils import datapack
from areal.utils.data import (
    MicroBatchList,
    MicroBatchSpec,
    allocate_balanced_mbs,
    concat_padded_tensors,
    pack_tensor_dict,
    pad_mb_list,
    split_padded_tensor_dict_into_mb_list,
    unsqueeze_mb_list,
)


def make_fake_trajectories(
    n_episodes: int = 504,
    min_steps: int = 1,
    max_steps: int = 10,
    min_step_tokens: int = 2000,
    max_step_tokens: int = 3000,
    hidden_dim: int = 3584,  # Qwen3-VL-8B hidden size
    seed: int = 42,
) -> list[dict[str, torch.Tensor]]:
    """Generate fake multi-step trajectories mimicking WebGym rollout output.

    Each trajectory has shape [n_steps, seq_len] where seq_len varies per episode.
    This matches what RolloutController receives from the workflow.
    """
    rng = np.random.RandomState(seed)
    trajs = []
    for _ in range(n_episodes):
        n_steps = rng.randint(min_steps, max_steps + 1)
        # Each step adds tokens; cumulative sequence grows
        step_lens = []
        for t in range(n_steps):
            if t == 0:
                step_lens.append(rng.randint(min_step_tokens, max_step_tokens))
            else:
                step_lens.append(step_lens[-1] + rng.randint(500, 1500))

        seq_len = step_lens[-1]
        input_ids = torch.randint(0, 32000, (n_steps, seq_len), dtype=torch.long)
        attention_mask = torch.zeros(n_steps, seq_len, dtype=torch.long)
        for t in range(n_steps):
            attention_mask[t, : step_lens[t]] = 1
        loss_mask = torch.zeros(n_steps, seq_len, dtype=torch.float32)
        rewards = torch.zeros(n_steps, dtype=torch.float32)
        rewards[-1] = float(rng.random() > 0.5)
        logprobs = torch.randn(n_steps, seq_len)

        trajs.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "rewards": rewards,
                "logprobs": logprobs,
            }
        )
    return trajs


# ---------------------------------------------------------------------------
# Streaming assembly: skip global concat, build microbatches directly
# ---------------------------------------------------------------------------


def streaming_assemble_mb_list(
    trajectories: list[dict[str, torch.Tensor]],
    mb_spec: MicroBatchSpec,
) -> MicroBatchList:
    """Assemble microbatches directly from a list of variable-length trajectories.

    Skips the expensive global concat_padded_tensors step entirely.
    Instead:
      1. Compute per-sequence lengths from each trajectory (instant)
      2. Run FFD bin packing to group sequences into microbatches
      3. For each microbatch, concat only its few sequences (2-5), pad within the group
    """
    # Step 1: Flatten all sequences and their metadata
    # Each trajectory has shape [n_steps, seq_len_i] — n_steps sequences per episode
    all_seq_lens: list[int] = []
    seq_to_traj: list[tuple[int, int]] = []  # (traj_idx, step_idx)
    for traj_idx, traj in enumerate(trajectories):
        am = traj["attention_mask"]
        n_steps = am.shape[0]
        for step_idx in range(n_steps):
            seq_len = int(am[step_idx].sum().item())
            all_seq_lens.append(seq_len)
            seq_to_traj.append((traj_idx, step_idx))

    n_seqs = len(all_seq_lens)

    # Step 2: FFD bin packing (same algorithm, just on metadata)
    group_indices = allocate_balanced_mbs(mb_spec, all_seq_lens)
    group_indices = sorted([sorted(g) for g in group_indices])

    # Build forward/backward index mapping
    forward_indices = datapack.flat2d(group_indices)
    backward_indices = np.zeros(n_seqs, dtype=np.int64)
    backward_indices[forward_indices] = np.arange(n_seqs)

    # Step 3: Assemble each microbatch by directly indexing into trajectories.
    # Per-sequence tensors have shape [n_steps, seq_len, ...] (2D rows).
    # Per-episode scalars like rewards have shape [n_steps] (0D per step).
    # We classify by checking ndim of a single row (trajectory[key][step_idx]).
    sample_traj = trajectories[0]
    seq_tensor_keys = []  # Keys where each step is a 1D+ tensor (seq_len, ...)
    scalar_tensor_keys = []  # Keys where each step is a scalar (0D)
    non_tensor_keys = []
    for key, val in sample_traj.items():
        if torch.is_tensor(val) and val.ndim >= 2:
            # val shape is [n_steps, seq_len, ...]; each row is [seq_len, ...]
            seq_tensor_keys.append(key)
        elif torch.is_tensor(val) and val.ndim == 1:
            # val shape is [n_steps]; each element is a scalar
            scalar_tensor_keys.append(key)
        else:
            non_tensor_keys.append(key)

    mbs: list[dict[str, torch.Tensor]] = []
    group_lens: list[int] = []

    # Pre-compute the padded row width for each sequence (= trajectory's seq_len dim)
    all_row_widths: list[int] = []
    for traj_idx, traj in enumerate(trajectories):
        width = traj["attention_mask"].shape[1]  # padded width of this trajectory
        n_steps = traj["attention_mask"].shape[0]
        for _ in range(n_steps):
            all_row_widths.append(width)

    for group_idx_list in group_indices:
        mb_seq_lens = [all_seq_lens[i] for i in group_idx_list]
        group_lens.append(sum(mb_seq_lens))
        # Pad to the max *row width* in this group (not max token count)
        mb_max_width = max(all_row_widths[i] for i in group_idx_list)

        mb_dict: dict[str, torch.Tensor] = {}

        # Sequence tensors: gather rows, pad to local max width, stack → [n_seqs, mb_max_width, ...]
        for key in seq_tensor_keys:
            rows = []
            for seq_idx in group_idx_list:
                traj_idx, step_idx = seq_to_traj[seq_idx]
                row = trajectories[traj_idx][key][step_idx]  # [row_width_i, ...]
                if row.shape[0] < mb_max_width:
                    pad_size = mb_max_width - row.shape[0]
                    padding = torch.zeros(
                        (pad_size, *row.shape[1:]), dtype=row.dtype
                    )
                    row = torch.cat([row, padding], dim=0)
                rows.append(row)
            mb_dict[key] = torch.stack(rows, dim=0)

        # Scalar tensors (rewards etc): gather scalars → [n_seqs]
        for key in scalar_tensor_keys:
            vals = []
            for seq_idx in group_idx_list:
                traj_idx, step_idx = seq_to_traj[seq_idx]
                vals.append(trajectories[traj_idx][key][step_idx].unsqueeze(0))
            mb_dict[key] = torch.cat(vals, dim=0)

        mbs.append(mb_dict)

    # Build a minimal "data" dict (just metadata, not the full concat)
    # This is needed for MicroBatchList but won't be used in the streaming path
    data_placeholder = {"_n_seqs": n_seqs, "_streaming": True}

    return MicroBatchList(
        data=data_placeholder,
        mb_spec=mb_spec,
        mbs=mbs,
        forward_indices=forward_indices.tolist()
        if isinstance(forward_indices, np.ndarray)
        else forward_indices,
        backward_indices=backward_indices.tolist(),
        group_lens=group_lens,
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def benchmark(n_episodes: int = 504, max_tokens_per_mb: int = 16384):
    """Compare old (concat-then-split) vs new (streaming) assembly."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {n_episodes} episodes, max_tokens_per_mb={max_tokens_per_mb}")
    print(f"{'='*60}")

    trajs = make_fake_trajectories(n_episodes=n_episodes)
    total_steps = sum(t["attention_mask"].shape[0] for t in trajs)
    total_tokens = sum(int(t["attention_mask"].sum().item()) for t in trajs)
    max_seq_len = max(t["attention_mask"].shape[1] for t in trajs)
    print(
        f"Generated: {n_episodes} episodes, {total_steps} total steps, "
        f"{total_tokens:,} tokens, max_seq_len={max_seq_len}"
    )

    mb_spec = MicroBatchSpec(max_tokens_per_mb=max_tokens_per_mb)

    # --- Old path: concat → split → pack ---
    t0 = time.monotonic()
    batch = concat_padded_tensors(trajs)
    t1 = time.monotonic()
    old_concat_time = t1 - t0

    mb_list_old = split_padded_tensor_dict_into_mb_list(batch, mb_spec)
    t2 = time.monotonic()
    old_split_time = t2 - t1

    # Pack each microbatch (simulating what _prepare_mb_list does)
    old_packed = []
    for mb in mb_list_old.mbs:
        old_packed.append(pack_tensor_dict(mb))
    t3 = time.monotonic()
    old_pack_time = t3 - t2
    old_total = t3 - t0

    print(f"\nOLD (concat → split → pack):")
    print(f"  concat_padded_tensors: {old_concat_time:.3f}s")
    print(f"  split_into_mb_list:    {old_split_time:.3f}s")
    print(f"  pack_tensor_dict:      {old_pack_time:.3f}s")
    print(f"  TOTAL:                 {old_total:.3f}s")
    print(f"  n_microbatches:        {len(mb_list_old.mbs)}")
    print(
        f"  concat tensor shape:   {batch['input_ids'].shape} "
        f"({batch['input_ids'].numel() * 8 / 1e9:.2f} GB for int64)"
    )

    # --- New path: streaming assembly ---
    t4 = time.monotonic()
    mb_list_new = streaming_assemble_mb_list(trajs, mb_spec)
    t5 = time.monotonic()
    new_assembly_time = t5 - t4

    # Pack each microbatch
    new_packed = []
    for mb in mb_list_new.mbs:
        new_packed.append(pack_tensor_dict(mb))
    t6 = time.monotonic()
    new_pack_time = t6 - t5
    new_total = t6 - t4

    print(f"\nNEW (streaming assembly):")
    print(f"  streaming_assemble:    {new_assembly_time:.3f}s")
    print(f"  pack_tensor_dict:      {new_pack_time:.3f}s")
    print(f"  TOTAL:                 {new_total:.3f}s")
    print(f"  n_microbatches:        {len(mb_list_new.mbs)}")

    speedup = old_total / new_total if new_total > 0 else float("inf")
    print(f"\nSpeedup: {speedup:.1f}x ({old_total:.3f}s → {new_total:.3f}s)")

    # Verify correctness: same total tokens per microbatch
    old_group_lens = sorted(mb_list_old.group_lens)
    new_group_lens = sorted(mb_list_new.group_lens)
    assert len(old_group_lens) == len(new_group_lens), (
        f"Different number of microbatches: {len(old_group_lens)} vs {len(new_group_lens)}"
    )
    # Token counts should match (same FFD algorithm)
    assert old_group_lens == new_group_lens, (
        f"Token distribution mismatch:\n  old: {old_group_lens[:5]}...\n  new: {new_group_lens[:5]}..."
    )
    print("Correctness: token distribution matches across microbatches")

    # Verify packed shapes match
    old_total_packed = sum(p["input_ids"].shape[0] for p in old_packed)
    new_total_packed = sum(p["input_ids"].shape[0] for p in new_packed)
    assert old_total_packed == new_total_packed, (
        f"Total packed tokens differ: {old_total_packed} vs {new_total_packed}"
    )
    print(f"Correctness: total packed tokens match ({old_total_packed})")

    # Memory comparison
    old_concat_bytes = sum(
        v.numel() * v.element_size()
        for v in batch.values()
        if torch.is_tensor(v)
    )
    new_mb_bytes = sum(
        v.numel() * v.element_size()
        for mb in mb_list_new.mbs
        for v in mb.values()
        if torch.is_tensor(v)
    )
    print(
        f"\nPeak CPU memory (tensors only):"
        f"\n  OLD concat tensor: {old_concat_bytes / 1e9:.2f} GB"
        f"\n  NEW all MBs total: {new_mb_bytes / 1e9:.2f} GB"
        f"\n  Ratio: {old_concat_bytes / new_mb_bytes:.1f}x more in old path"
    )

    return speedup


def test_streaming_vs_concat_small():
    """Quick correctness test with small batch."""
    speedup = benchmark(n_episodes=20, max_tokens_per_mb=16384)
    assert speedup > 1.0, f"Expected streaming to be faster, got {speedup:.1f}x"


def test_streaming_vs_concat_medium():
    """Medium batch benchmark."""
    speedup = benchmark(n_episodes=100, max_tokens_per_mb=16384)
    assert speedup > 1.0, f"Expected streaming to be faster, got {speedup:.1f}x"


def test_streaming_vs_concat_large():
    """Large batch benchmark (similar to production batch=504)."""
    speedup = benchmark(n_episodes=504, max_tokens_per_mb=16384)
    assert speedup > 1.5, f"Expected significant speedup, got {speedup:.1f}x"


if __name__ == "__main__":
    benchmark(n_episodes=20, max_tokens_per_mb=16384)
    benchmark(n_episodes=100, max_tokens_per_mb=16384)
    benchmark(n_episodes=504, max_tokens_per_mb=16384)
