# webgym/data/replay_buffer.py
import numpy as np
from torch.utils.data import Dataset

# Use TYPE_CHECKING to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from webgym.models.web_agent import WebAgent

class ReplayBuffer(Dataset):
    def __init__(self, trajectories, agent: 'WebAgent', capacity=None,
                filter_successful_only=False, include_reward_in_sample=True, shuffle=False,
                filter_same_screenshot=True):
        """
        Initialize dataset from trajectories using WebAgent with context management.

        Args:
            trajectories: List of trajectory data
            agent: The WebAgent instance (contains context manager)
            capacity: Optional capacity limit (if None, no limit)
            filter_successful_only: If True, only samples with trajectory_reward=1 are accessible
                                   (filtering happens at sample-time, not during initialization)
            include_reward_in_sample: If True, include trajectory_reward in returned samples
            shuffle: If True, shuffle trajectories before processing
            filter_same_screenshot: If True, filter out steps with same screenshot from successful trajectories
        """
        self.trajectories = trajectories
        self.agent = agent
        self.capacity = capacity
        self.filter_successful_only = filter_successful_only
        self.include_reward_in_sample = include_reward_in_sample
        self.filter_same_screenshot = filter_same_screenshot

        self.context_manager = agent.context_manager
        # Get conversation builder from model interface (new API)
        self.conversation_builder = self.context_manager.get_model_interface().conversation_builder

        # Filter out dummy trajectories before processing
        filtered_trajectories = self._filter_dummy_trajectories(trajectories)

        # Calculate success rate for logging (but don't filter yet)
        try:
            import deepspeed
            rank = deepspeed.comm.get_rank()
        except:
            rank = 0

        if rank == 0:
            successful_count = sum(1 for traj in filtered_trajectories if traj[-1]['reward'].reward == 1)
            success_rate = successful_count / len(filtered_trajectories) * 100 if filtered_trajectories else 0
            print(f"Trajectory success rate: {successful_count}/{len(filtered_trajectories)} = {success_rate:.1f}%")
            if self.filter_successful_only:
                print(f"Sample-time filtering enabled: Only steps from successful trajectories will be accessible")

        # Process ALL trajectories (no pre-filtering)
        state_strings, action_strings, trajectory_rewards, same_as_next_screenshot = self._reconstruct_prompt(filtered_trajectories)

        # Create all samples and identify training-eligible ones
        # Training-eligible: successful trajectory + different screenshot (if filtering enabled)
        all_samples = []
        training_indices = []

        for i in range(len(state_strings)):
            reward = trajectory_rewards[i]
            same_screenshot = same_as_next_screenshot[i]

            if self.filter_same_screenshot:
                is_trainable = (reward == 1 and not same_screenshot)
            else:
                is_trainable = (reward == 1)

            all_samples.append({
                "message": state_strings[i],
                "action": action_strings[i],
                "trajectory_reward": reward,
                "same_as_next_screenshot": same_screenshot,
            })

            if is_trainable:
                training_indices.append(i)

        if rank == 0:
            print(f"📊 Training-eligible samples: {len(training_indices)} (from successful trajectories)")


        # Shuffle samples if requested
        if shuffle and all_samples:
            import random
            random.shuffle(all_samples)

        # Apply capacity limit
        if self.capacity is not None and len(all_samples) > self.capacity:
            all_samples = all_samples[-self.capacity:]

        self.samples = all_samples

        if self.filter_successful_only:
            self.training_indices = training_indices
            self.successful_indices = training_indices

            if rank == 0:
                total_reward_1 = sum(1 for s in self.samples if s['trajectory_reward'] == 1)
                total_reward_0 = sum(1 for s in self.samples if s['trajectory_reward'] == 0)

                print(f"Created {len(self.samples)} total samples")
                print(f"  - {total_reward_1} from successful trajectories (reward=1)")
                print(f"  - {total_reward_0} from failed trajectories (reward=0, excluded)")
                print(f"  - {len(self.training_indices)} training-eligible samples")
        else:
            self.training_indices = list(range(len(self.samples)))
            self.successful_indices = self.training_indices
            if rank == 0:
                print(f"Created {len(self.samples)} total samples (filtering disabled)")
    
    def __len__(self):
        # Return length of filtered samples (only successful if filter_successful_only=True)
        return len(self.successful_indices)

    def __getitem__(self, idx):
        # Map requested index to actual sample index via successful_indices
        actual_idx = self.successful_indices[idx]
        sample = self.samples[actual_idx]

        if self.include_reward_in_sample:
            return sample
        else:
            return {
                "message": sample["message"],
                "action": sample["action"]
            }

    def get_training_samples(self, num_samples=None, recency_bias_power=1.0):
        """Get training-eligible samples with recency-weighted sampling.

        Samples directly from training_indices (successful trajectory steps),
        so recency bias is applied only to the training pool.

        Args:
            num_samples: Number of samples to retrieve.
                         If None or >= available, return all.
            recency_bias_power: Power for recency weighting (higher = more recent bias).
                               If 1.0, uses uniform random sampling.

        Returns:
            List of training samples (dicts with 'message' and 'action')
        """
        indices = self.training_indices

        if num_samples is not None and num_samples < len(indices):
            positions = np.arange(len(indices))
            if recency_bias_power != 1.0:
                weights = (positions + 1) ** recency_bias_power
                weights = weights / weights.sum()
                sampled = np.random.choice(
                    positions, size=num_samples, replace=False, p=weights
                )
            else:
                sampled = np.random.choice(
                    positions, size=num_samples, replace=False
                )
            selected = [indices[pos] for pos in sampled]
        else:
            selected = indices

        return [self.samples[i] for i in selected]

    def _filter_dummy_trajectories(self, trajectories):
        """
        Filter out dummy trajectories created for invalid URLs.
        
        A dummy trajectory is identified by:
        - Having only one step
        - The action being of type "invalid_url" 
        - The observation having empty image_path
        - The action_string containing "Invalid URL"
        """
        filtered_trajectories = []
        dummy_count = 0
        
        for trajectory in trajectories:
            if self._is_dummy_trajectory(trajectory):
                dummy_count += 1
                continue  # Skip this trajectory
            
            filtered_trajectories.append(trajectory)
        
        if dummy_count > 0:
            try:
                import deepspeed
                rank = deepspeed.comm.get_rank()
            except:
                rank = 0  # Default to rank 0 if deepspeed not available

            if rank == 0:
                print(f"Filtered out {dummy_count} dummy trajectories from {len(trajectories)} total")
                print(f"Using {len(filtered_trajectories)} valid trajectories for training")
        
        return filtered_trajectories
    
    def _is_dummy_trajectory(self, trajectory):
        """
        Check if a trajectory is a dummy trajectory created for invalid URLs.
        """
        # Must have exactly one step to be a dummy trajectory
        if len(trajectory) != 1:
            return False
        
        step = trajectory[0]
        
        # Check observation for empty image_path (indicator of invalid URL)
        if hasattr(step, 'get'):
            observation = step.get('observation')
        else:
            observation = getattr(step, 'observation', None)
            
        if observation and hasattr(observation, 'image_path'):
            if observation.image_path == "":
                return True
        
        # Check action for invalid_url type
        if hasattr(step, 'get'):
            action = step.get('action')
        else:
            action = getattr(step, 'action', None)
            
        if action:
            # Check action dict for invalid_url key
            if hasattr(action, 'action') and isinstance(action.action, dict):
                if action.action.get('key') == 'invalid_url':
                    return True
            
            # Check action string for "Invalid URL" text
            if hasattr(action, 'action_string') and action.action_string:
                if "Invalid URL" in action.action_string:
                    return True
        
        return False
    
    def _reconstruct_prompt(self, trajectories):
        """Load saved prompts from trajectories (stored as JSON message lists with file:// paths)"""
        import json

        try:
            import deepspeed
            rank = deepspeed.comm.get_rank()
        except:
            rank = 0

        if rank == 0:
            print(f"Loading prompts from {len(trajectories)} trajectories...")

        messages_list = []
        responses = []
        trajectory_rewards = []
        same_as_next_screenshot = []  # Track if step has same screenshot as next step

        total_steps = 0
        legacy_steps = 0  # Track steps from legacy trajectories (without precomputed field)
        for i, traj in enumerate(trajectories):
            # Print progress every 1000 trajectories on rank 0
            if rank == 0 and i > 0 and i % 1000 == 0:
                print(f"  Processed {i}/{len(trajectories)} trajectories ({total_steps} steps so far)...")

            for j, step in enumerate(traj):
                # Parse saved raw_prompt (JSON string with message lists and file:// paths)
                if step['response'] is not None and hasattr(step['response'], 'raw_prompt') and step['response'].raw_prompt:
                    try:
                        messages = json.loads(step['response'].raw_prompt)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Step {j} in trajectory {i} has invalid JSON in raw_prompt: {e}. "
                            "This may indicate corrupted trajectory data. "
                            "Please re-run rollout to collect new trajectories."
                        )
                else:
                    raise ValueError(
                        f"Step {j} in trajectory {i} missing raw_prompt. "
                        "Old trajectories without raw_prompt are not supported. "
                        "Please re-run rollout to collect new trajectories."
                    )

                messages_list.append(messages)

                if step['response'] is not None:
                    responses.append(step['response'].raw_response)
                else:
                    responses.append("")

                trajectory_rewards.append(traj[-1]['reward'].reward)

                # Read precomputed same_as_next_screenshot value from trajectory step
                # For legacy trajectories without this field, assume False (different screenshot)
                if 'same_as_next_screenshot' in step:
                    # Use precomputed value (new trajectories)
                    same_screenshot = step['same_as_next_screenshot']
                else:
                    # Legacy trajectory - assume different screenshot (False)
                    # This avoids expensive image loading and pixel comparison
                    legacy_steps += 1
                    same_screenshot = False

                same_as_next_screenshot.append(same_screenshot)

                total_steps += 1

        if rank == 0:
            print(f"  Completed: {len(trajectories)} trajectories → {total_steps} training steps")
            print(f"  ✅ Loaded {len(messages_list)} training samples with message lists")

            # Log statistics about same-screenshot steps
            same_screenshot_count = sum(same_as_next_screenshot)
            if total_steps > 0:
                same_screenshot_pct = same_screenshot_count / total_steps * 100
                print(f"  📊 Same-screenshot steps: {same_screenshot_count}/{total_steps} ({same_screenshot_pct:.1f}%)")

            # Log legacy trajectory handling
            if legacy_steps > 0:
                legacy_pct = legacy_steps / total_steps * 100
                print(f"  ⚠️  Legacy steps (computed on-demand): {legacy_steps}/{total_steps} ({legacy_pct:.1f}%)")
                print(f"     Consider re-running rollout to precompute screenshot comparisons for better performance")
            else:
                print(f"  ✅ All steps have precomputed screenshot comparisons")

        return messages_list, responses, trajectory_rewards, same_as_next_screenshot