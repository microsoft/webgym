import sys
# ignore --local_rank input as we're using hydra
sys.argv = [arg for arg in sys.argv if not arg.startswith("--local_rank")]

import os
import json
import yaml
import glob
import hydra
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from webgym.misc import colorful_print
from webgym.data.replay_buffer import ReplayBuffer
from webgym.models import WebAgent
from webgym.logging import get_latest_wandb_run_info
from pathlib import Path
from huggingface_hub import login
import tempfile
import shutil
import wandb

# ───── Trajectory Cleaning ─────

def clean_trajectories(trajs):
    """Filter out trajectories that have None values or are too short (< 2 steps, i.e., require >= 2 steps)"""
    cleaned_trajs = []

    for traj in trajs:
        # Skip empty trajectories
        if not traj:
            continue

        # Filter out very short trajectories (require at least 2 steps)
        if len(traj) < 2:
            continue

        # Check if any step has None values
        has_none_values = False
        for step in traj:
            if not step:
                has_none_values = True
                break
            if (step.get('action') is None or
                step.get('observation') is None or
                step.get('response') is None):
                has_none_values = True
                break

        # Only keep trajectories without None values
        if not has_none_values:
            cleaned_trajs.append(traj)

    return cleaned_trajs




class DataPreparationManager:
    """
    Manages data preparation for LLaMA-Factory training.
    Converts WebGym training samples to LLaMA-Factory format and saves them.
    """

    def __init__(self, agent, save_path: str, algorithm_config: DictConfig):
        self.agent = agent
        self.save_path = save_path
        self.algorithm_config = algorithm_config

        # Create output directory
        self.output_dir = os.path.join(save_path, "llamafactory_data")
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"🚀 DataPreparationManager initialized")
        print(f"   Output directory: {self.output_dir}")

    def prepare_all_data(self, replay_buffer: ReplayBuffer, positive_samples_to_train: int = None, recency_bias_power: float = 1.0, val_split_ratio: float = 0.0) -> tuple:
        """
        Prepare data from replay buffer by sampling successful steps with recency bias.

        Args:
            replay_buffer: ReplayBuffer containing all training data
            positive_samples_to_train: Number of successful steps to sample with recency bias.
                                       If None or >= available, use all.
            recency_bias_power: Power for recency weighting (higher = more recent bias)
            val_split_ratio: Fraction of data to use for validation (0.0 to 1.0)

        Returns:
            Tuple of (train_dataset_path, val_dataset_path or None)
        """
        print(f"\n===== Preparing Data for LLaMA-Factory =====")
        print(f"Model: {self.agent.policy_config.base_model}")
        print(f"Total steps (all): {len(replay_buffer.samples)}")
        print(f"Training-eligible steps: {len(replay_buffer)}")

        # Sample from training-eligible steps with recency weighting
        sampled = replay_buffer.get_training_samples(
            num_samples=positive_samples_to_train,
            recency_bias_power=recency_bias_power
        )
        print(f"📊 Sampled {len(sampled)} training steps (recency_bias_power={recency_bias_power})")

        # Convert sampled data to LLaMA-Factory format
        training_data = []
        print("🔄 Converting samples to LLaMA-Factory format...")

        for sample in sampled:
            llamafactory_sample = self._convert_to_llamafactory_format(
                sample["message"], sample["action"]
            )
            if llamafactory_sample:
                training_data.append(llamafactory_sample)

        print(f"✅ Converted {len(training_data)} samples to LLaMA-Factory format")

        # Shuffle data before splitting or saving (always shuffle)
        import random
        random.shuffle(training_data)
        print(f"🔀 Shuffled {len(training_data)} samples")

        # Split into train/val if requested
        val_dataset_path = None
        if val_split_ratio > 0 and len(training_data) > 1:
            val_size = int(len(training_data) * val_split_ratio)
            val_size = max(1, val_size)  # At least 1 validation sample

            val_data = training_data[:val_size]
            train_data = training_data[val_size:]

            print(f"📊 Split data: {len(train_data)} train, {len(val_data)} validation")

            # Save train dataset
            train_dataset_path = os.path.join(self.output_dir, "finetune_train.json")
            with open(train_dataset_path, 'w', encoding='utf-8') as f:
                json.dump(train_data, f, indent=2, ensure_ascii=False)

            # Save validation dataset
            val_dataset_path = os.path.join(self.output_dir, "finetune_val.json")
            with open(val_dataset_path, 'w', encoding='utf-8') as f:
                json.dump(val_data, f, indent=2, ensure_ascii=False)

            print(f"💾 Saved train dataset to: {train_dataset_path}")
            print(f"💾 Saved validation dataset to: {val_dataset_path}")
        else:
            # No validation split - save all as training data
            train_dataset_path = os.path.join(self.output_dir, "finetune.json")
            with open(train_dataset_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)

            print(f"💾 Saved dataset to: {train_dataset_path}")
            print(f"📊 Dataset contains {len(training_data)} training samples")

        return train_dataset_path, val_dataset_path

    def _escape_multimodal_tokens(self, text: str) -> str:
        """
        Escape HTML-like tags in text that could be confused with multimodal tokens.
        Only escapes <video>, <audio> tags - NOT <image> since we use that for actual images.

        This prevents validation errors when webpage content contains text like:
        "HTML5 <video> player" or "<audio> element"
        """
        if not text:
            return text

        # Escape <video> and </video> tags
        text = text.replace("<video>", "&lt;video&gt;")
        text = text.replace("</video>", "&lt;/video&gt;")

        # Escape <audio> and </audio> tags
        text = text.replace("<audio>", "&lt;audio&gt;")
        text = text.replace("</audio>", "&lt;/audio&gt;")

        return text

    def _convert_to_llamafactory_format(self, messages, raw_response) -> dict:
        """
        Convert a single WebGym sample to LLaMA-Factory ShareGPT format.

        Strategy: Output standard multi-turn ShareGPT format with all conversation history.
        LLaMA-Factory's mask_history=true will handle masking previous turns during training.

        Args:
            messages: List of message dicts (system + user + assistant) from raw_prompt
            raw_response: String containing the assistant's response for current step
        """
        try:
            images = []
            system_prompt = ""
            conversations = []

            # Parse messages and build standard multi-turn format
            for msg in messages:
                role = msg.get("role")

                if role == "system":
                    system_prompt = self._escape_multimodal_tokens(msg.get("content", ""))

                elif role == "user":
                    content = msg.get("content", [])

                    # Build user message value by processing content items in order
                    # Images become <image> tokens, text is preserved as-is
                    value_parts = []

                    if isinstance(content, list):
                        # Process items in order to build the message value
                        for item in content:
                            if item.get("type") == "image_url":
                                # Add image to images list and insert <image> token placeholder
                                image_url = item.get("image_url", {}).get("url", "")
                                if image_url.startswith("file://"):
                                    images.append(image_url[7:])
                                    value_parts.append("<image>")
                            elif item.get("type") == "text":
                                # Add text content (escaped for multimodal tokens)
                                text = item.get("text", "")
                                value_parts.append(self._escape_multimodal_tokens(text))
                    elif isinstance(content, str):
                        # Handle text-only user messages
                        value_parts.append(self._escape_multimodal_tokens(content))

                    conversations.append({
                        "from": "human",
                        "value": "".join(value_parts)
                    })

                elif role == "assistant":
                    # This is a historical assistant message
                    content = msg.get("content", "")
                    # Assistant content is typically a string, but handle both cases
                    if isinstance(content, str):
                        assistant_text = self._escape_multimodal_tokens(content)
                    else:
                        # If it's a list (uncommon for assistant), extract text
                        assistant_text = self._escape_multimodal_tokens(str(content))

                    conversations.append({
                        "from": "gpt",
                        "value": assistant_text
                    })

            # Add the final assistant response (the current action being trained)
            escaped_response = self._escape_multimodal_tokens(raw_response)
            conversations.append({
                "from": "gpt",
                "value": escaped_response
            })

            # Create ShareGPT format sample
            # IMPORTANT: Always include "system" field (even if empty) to match dataset_info.json schema
            # This ensures compatibility with LLaMA-Factory's dataset loader
            sample = {
                "conversations": conversations,
                "system": system_prompt,  # Always include, defaults to "" if no system prompt
            }

            # Add images if present
            if images:
                sample["images"] = images

            return sample

        except Exception as e:
            print(f"⚠️ Error converting sample: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_llamafactory_config(self, train_dataset_path: str, val_dataset_path: str, log_config: DictConfig, num_train_samples: int = None) -> str:
        """
        Create LLaMA-Factory training configuration.

        Args:
            train_dataset_path: Path to the training dataset
            val_dataset_path: Path to the validation dataset (or None if no validation)
            log_config: Logging configuration with wandb settings
            num_train_samples: Number of training samples (used for max_steps calculation)

        Returns:
            Path to the configuration file
        """
        # Get output model name from config (defaults to "model.pt" if not specified)
        model_output_name = getattr(self.algorithm_config, 'model_output_name', 'model.pt')

        # Output directory for the trained model
        output_dir = os.path.join(self.save_path, model_output_name)

        # Check if we should force HuggingFace initialization
        init_from_hf = getattr(self.algorithm_config, 'init_model_from_hf', False)

        # Determine model path - force base model if init_from_hf is True
        if init_from_hf:
            model_path = self.agent.policy_config.base_model
            print(f"🆕 Config set to init_model_from_hf=true - forcing initialization from HuggingFace base model")
            print(f"   Base model: {model_path}")
            print(f"   Local checkpoint at {output_dir} will be ignored for initialization")
        else:
            model_path = self.agent.policy_config.base_model
            if hasattr(self.agent, 'updated_model_path') and self.agent.updated_model_path:
                model_path = self.agent.updated_model_path

        # Check if we should resume from checkpoint (optimizer states AND trainer_state.json exist)
        # Note: We only resume optimizer, not scheduler, to avoid epoch counting issues
        # IMPORTANT: We must check for trainer_state.json existence to avoid resume errors
        should_resume = False
        if init_from_hf:
            # Skip resume logic when forcing HuggingFace initialization
            should_resume = False
        elif os.path.isdir(model_path):
            # CRITICAL: First check if trainer_state.json exists
            # Without this file, HuggingFace Trainer cannot resume training
            trainer_state_path = os.path.join(model_path, "trainer_state.json")
            if not os.path.exists(trainer_state_path):
                print(f"📦 No trainer_state.json found - will load model weights only from: {model_path}")
                print(f"   (Model files exist but checkpoint is not resumable)")
                should_resume = False
            else:
                # trainer_state.json exists - now check for optimizer states
                # Check for standard HuggingFace optimizer.pt
                optimizer_path = os.path.join(model_path, "optimizer.pt")
                has_optimizer = os.path.exists(optimizer_path)

                # Also check for DeepSpeed checkpoint format (global_step* directories)
                has_deepspeed_checkpoint = False
                if not has_optimizer:
                    # Look for DeepSpeed checkpoint directories (global_step*)
                    deepspeed_dirs = glob.glob(os.path.join(model_path, "global_step*"))
                    if deepspeed_dirs:
                        # Check if any contains optimizer states
                        for ds_dir in deepspeed_dirs:
                            optim_files = glob.glob(os.path.join(ds_dir, "*_optim_states.pt"))
                            if optim_files:
                                has_deepspeed_checkpoint = True
                                break

                    # Also check for 'latest' file pointing to a checkpoint
                    latest_file = os.path.join(model_path, "latest")
                    if os.path.exists(latest_file):
                        has_deepspeed_checkpoint = True

                if has_optimizer or has_deepspeed_checkpoint:
                    should_resume = True
                    checkpoint_type = "DeepSpeed" if has_deepspeed_checkpoint else "HuggingFace"
                    print(f"🔄 Found {checkpoint_type} optimizer states AND trainer_state.json, will resume training: {model_path}")
                else:
                    print(f"📦 Found trainer_state.json but no optimizer states - will load model weights only from: {model_path}")
        else:
            print(f"🆕 Using base model (first iteration): {model_path}")

        # Setup WandB configuration for LLaMA-Factory
        # Use "rl" as project name (changed from "llamafactory")
        project_name = "rl"
        run_name = log_config.run_name

        # Check if run exists and get run_id for resume
        run_id, max_cloud_step, is_new_run = get_latest_wandb_run_info(
            project_name=project_name,
            entity_name=log_config.entity_name,
            run_name=run_name
        )

        # Construct full project path for display and environment variable
        full_project_path = f"{log_config.entity_name}/{project_name}"

        if is_new_run:
            print(f"🆕 WandB: Starting new run '{run_name}' in project '{full_project_path}'")
            wandb_run_id = None
            wandb_resume = None
        else:
            print(f"🔄 WandB: Resuming run '{run_name}' (id: {run_id}) in project '{full_project_path}'")
            wandb_run_id = run_id
            wandb_resume = "allow"

        # Create dataset_info.json for LLaMA-Factory
        dataset_info_path = os.path.join(self.output_dir, "dataset_info.json")

        # Determine train dataset filename
        train_filename = os.path.basename(train_dataset_path)

        dataset_info = {
            "webgym_training": {
                "file_name": train_filename,  # Relative path within dataset_dir
                "formatting": "sharegpt",
                "columns": {
                    "messages": "conversations",  # ShareGPT uses "conversations"
                    "system": "system",  # Separate system field
                    "images": "images"
                },
                "tags": {
                    "role_tag": "from",  # ShareGPT uses "from" for role
                    "content_tag": "value",  # ShareGPT uses "value" for content
                    "user_tag": "human",  # User messages use "human" tag
                    "assistant_tag": "gpt",  # Assistant messages use "gpt" tag
                    "observation_tag": "observation"  # Historical messages use "observation" (not trained)
                }
            }
        }

        # Add validation dataset if provided
        if val_dataset_path:
            val_filename = os.path.basename(val_dataset_path)
            dataset_info["webgym_validation"] = {
                "file_name": val_filename,
                "formatting": "sharegpt",
                "columns": {
                    "messages": "conversations",  # ShareGPT uses "conversations"
                    "system": "system",  # Separate system field
                    "images": "images"
                },
                "tags": {
                    "role_tag": "from",  # ShareGPT uses "from" for role
                    "content_tag": "value",  # ShareGPT uses "value" for content
                    "user_tag": "human",  # User messages use "human" tag
                    "assistant_tag": "gpt",  # Assistant messages use "gpt" tag
                    "observation_tag": "observation"  # Historical messages use "observation" (not trained)
                }
            }

        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2)

        print(f"📋 Created dataset registry: {dataset_info_path}")

        # Build deepspeed config path
        deepspeed_config_path = os.path.join(
            os.path.dirname(__file__),
            "config/deepspeed",
            self.algorithm_config.deepspeed_config_filename
        )

        # Set trust_remote_code via environment variable (for model loading only, not datasets)
        os.environ["HF_TRUST_REMOTE_CODE"] = "1"

        # Detect GPU count (accounts for multi-node training)
        num_nodes = getattr(self.algorithm_config, 'num_nodes', 1)
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                  capture_output=True, text=True, check=True)
            num_gpus_per_node = len(result.stdout.strip().split('\n'))
            num_gpus = num_gpus_per_node * num_nodes  # Total GPUs across all nodes
            print(f"🖥️ Detected {num_gpus_per_node} GPUs per node × {num_nodes} nodes = {num_gpus} total GPUs")
        except:
            num_gpus = 8 * num_nodes  # Default: 8 GPUs per node
            print(f"🖥️ Using default GPU count: 8 GPUs per node × {num_nodes} nodes = {num_gpus} total GPUs")

        # Create LLaMA-Factory training configuration
        config = {
            "stage": "sft",
            "do_train": True,
            "dataset": "webgym_training",  # Dataset name from dataset_info.json
            "dataset_dir": self.output_dir,  # Directory containing dataset_info.json
            "template": self._get_model_template(),
            "finetuning_type": "full",  # Full finetuning
            "output_dir": output_dir,
            "overwrite_output_dir": True,  # Required by LLaMA-Factory when output dir exists
            "trust_remote_code": True,
            "mask_history": True,  # Only train on last assistant turn (mask previous turns)
        }

        # Print mask_history configuration
        print(f"🎯 mask_history=True: Only the last assistant turn will be trained")
        print(f"   All previous turns in conversation history will be masked out")

        # Add evaluation settings if validation dataset is provided
        if val_dataset_path and hasattr(self.algorithm_config, 'do_eval') and self.algorithm_config.do_eval:
            config["do_eval"] = True
            config["eval_dataset"] = "webgym_validation"  # Validation dataset name from dataset_info.json
            config["eval_strategy"] = getattr(self.algorithm_config, 'eval_strategy', 'steps')
            if hasattr(self.algorithm_config, 'eval_steps'):
                eval_steps_value = self.algorithm_config.eval_steps

                # Check if eval_steps is a fraction (0 < x < 1) - convert to actual step count
                if isinstance(eval_steps_value, (int, float)) and 0 < eval_steps_value < 1 and num_train_samples:
                    # Calculate steps per epoch using detected GPU count
                    per_device_batch_size = self.algorithm_config.per_device_train_batch_size
                    grad_accum_steps = self.algorithm_config.gradient_accumulation_steps
                    effective_batch_size = per_device_batch_size * grad_accum_steps * num_gpus
                    steps_per_epoch = num_train_samples // effective_batch_size

                    # Calculate actual eval step (e.g., 95% of epoch)
                    actual_eval_steps = int(steps_per_epoch * eval_steps_value)
                    actual_eval_steps = max(1, actual_eval_steps)  # Ensure at least 1 eval is performed

                    config["eval_steps"] = actual_eval_steps
                    print(f"📊 Converted eval_steps from fraction {eval_steps_value} to actual step count: {actual_eval_steps}")
                    print(f"   Steps per epoch: {steps_per_epoch}, evaluating at step {actual_eval_steps} ({eval_steps_value*100:.0f}% of epoch)")
                else:
                    config["eval_steps"] = eval_steps_value
            print(f"✅ Evaluation enabled with validation dataset")

        # Set model path (always required) and optionally resume checkpoint
        config["model_name_or_path"] = model_path

        # Check if we should actually resume or just load weights
        actual_resume = should_resume
        checkpoint_epoch = 0  # Track checkpoint epoch for later adjustment

        if should_resume:
            # Read checkpoint's global_step to check if resuming would cause issues
            trainer_state_path = os.path.join(model_path, "trainer_state.json")
            if os.path.exists(trainer_state_path):
                try:
                    with open(trainer_state_path, 'r') as f:
                        trainer_state = json.load(f)
                        checkpoint_global_step = trainer_state.get("global_step", 0)
                        checkpoint_epoch = trainer_state.get("epoch", 0)

                        print(f"📊 Checkpoint state: global_step={checkpoint_global_step}, epoch={checkpoint_epoch}")

                        # Always use ignore_data_skip when resuming to handle dataset size changes
                        config["ignore_data_skip"] = True

                        # CRITICAL: Update scheduler.pt to allow learning rate changes between iterations
                        # When resuming, HuggingFace Trainer loads scheduler state which includes base_lrs
                        # This overrides the new learning_rate config value. We need to update base_lrs
                        # in the scheduler state to match the new learning rate.
                        scheduler_path = os.path.join(model_path, "scheduler.pt")
                        if os.path.exists(scheduler_path):
                            try:
                                scheduler_state = torch.load(scheduler_path, weights_only=False)
                                old_lr = scheduler_state.get('base_lrs', [None])[0]
                                new_lr = self.algorithm_config.learning_rate

                                if old_lr != new_lr:
                                    print(f"🔧 Updating scheduler learning rate: {old_lr} → {new_lr}")
                                    # Update base_lrs for all parameter groups
                                    num_groups = len(scheduler_state.get('base_lrs', []))
                                    scheduler_state['base_lrs'] = [new_lr] * num_groups
                                    scheduler_state['_last_lr'] = [new_lr] * num_groups

                                    # Save updated scheduler
                                    torch.save(scheduler_state, scheduler_path)
                                    print(f"   ✅ Scheduler state updated with new learning rate")
                                else:
                                    print(f"✅ Scheduler learning rate already matches config: {new_lr}")
                            except Exception as e:
                                print(f"⚠️  Warning: Failed to update scheduler learning rate: {e}")
                                print(f"   Learning rate may not change from checkpoint value")

                        print(f"✅ Config set to resume from checkpoint (includes optimizer states): {model_path}")
                        print(f"   Note: ignore_data_skip=True allows training on new dataset from beginning")
                except Exception as e:
                    print(f"⚠️ Warning: Failed to read trainer_state.json ({e}), will load weights only")
                    actual_resume = False

            if actual_resume:
                config["resume_from_checkpoint"] = model_path

        if not actual_resume:
            print(f"✅ Config set to load model weights only: {model_path}")

        # Add training hyperparameters
        config.update({
            # Training hyperparameters (from config)
            "cutoff_len": self.algorithm_config.cutoff_len,
            "per_device_train_batch_size": self.algorithm_config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.algorithm_config.gradient_accumulation_steps,
            "learning_rate": self.algorithm_config.learning_rate,
            "max_grad_norm": self.algorithm_config.max_grad_norm,
            "num_train_epochs": self.algorithm_config.num_train_epochs,
            "logging_steps": self.algorithm_config.logging_steps,
            "save_strategy": self.algorithm_config.save_strategy,
            "lr_scheduler_type": self.algorithm_config.lr_scheduler_type,
        })

        # Add per_device_eval_batch_size if specified in config
        if hasattr(self.algorithm_config, 'per_device_eval_batch_size'):
            config["per_device_eval_batch_size"] = self.algorithm_config.per_device_eval_batch_size
            print(f"📊 Evaluation batch size per device: {self.algorithm_config.per_device_eval_batch_size}")

        # Add weight_decay if specified in config
        if hasattr(self.algorithm_config, 'weight_decay'):
            config["weight_decay"] = self.algorithm_config.weight_decay
            print(f"⚖️  Weight decay: {self.algorithm_config.weight_decay}")

        # Add warmup settings (use warmup_steps if available, otherwise warmup_ratio)
        if hasattr(self.algorithm_config, 'warmup_steps'):
            config["warmup_steps"] = self.algorithm_config.warmup_steps
        elif hasattr(self.algorithm_config, 'warmup_ratio'):
            config["warmup_ratio"] = self.algorithm_config.warmup_ratio

        # Add save settings if specified
        # Only set save_steps if save_strategy is "steps"
        save_strategy = self.algorithm_config.save_strategy
        if save_strategy == "steps" and hasattr(self.algorithm_config, 'save_steps'):
            save_steps_value = self.algorithm_config.save_steps

            # Check if save_steps is a fraction (0 < x < 1) - convert to actual step count
            if isinstance(save_steps_value, (int, float)) and 0 < save_steps_value < 1 and num_train_samples:
                # Calculate steps per epoch using detected GPU count
                per_device_batch_size = self.algorithm_config.per_device_train_batch_size
                grad_accum_steps = self.algorithm_config.gradient_accumulation_steps
                effective_batch_size = per_device_batch_size * grad_accum_steps * num_gpus
                steps_per_epoch = num_train_samples // effective_batch_size

                # Calculate actual save step (e.g., 95% of epoch)
                actual_save_steps = int(steps_per_epoch * save_steps_value)
                actual_save_steps = max(1, actual_save_steps)  # Ensure at least 1 checkpoint is saved

                config["save_steps"] = actual_save_steps
                print(f"💾 Converted save_steps from fraction {save_steps_value} to actual step count: {actual_save_steps}")
                print(f"   Steps per epoch: {steps_per_epoch}, saving at step {actual_save_steps} ({save_steps_value*100:.0f}% of epoch)")
            else:
                config["save_steps"] = save_steps_value
        elif save_strategy == "epoch":
            # Explicitly disable save_steps when using epoch-based saving to avoid conflicts
            config["save_steps"] = None
        if hasattr(self.algorithm_config, 'save_total_limit'):
            config["save_total_limit"] = self.algorithm_config.save_total_limit

        config.update({
            # Vision-specific settings (from config)
            "preprocessing_num_workers": self.algorithm_config.preprocessing_num_workers,
            "dataloader_num_workers": self.algorithm_config.dataloader_num_workers,

            # Performance settings (from config)
            "bf16": self.algorithm_config.bf16,
            "remove_unused_columns": self.algorithm_config.remove_unused_columns,
            "report_to": self.algorithm_config.report_to,
            "save_only_model": self.algorithm_config.save_only_model,
            "dataloader_pin_memory": self.algorithm_config.dataloader_pin_memory,

            # Memory optimization (from config)
            "gradient_checkpointing": self.algorithm_config.gradient_checkpointing,
            "deepspeed": deepspeed_config_path,

            # Disable built-in plotting (from config)
            "plot_loss": self.algorithm_config.plot_loss,

            # WandB settings - use run_name from config
            "run_name": run_name,
        })

        # Note: WandB project/entity/run_id are set via environment variables (WANDB_PROJECT, etc.)
        # They should NOT be added to config dict as HfArgumentParser doesn't recognize them

        # Adjust num_train_epochs and max_steps when resuming to ensure we train for desired epochs
        if actual_resume and checkpoint_epoch > 0:
            desired_new_epochs = self.algorithm_config.num_train_epochs
            # Add checkpoint epochs to desired new epochs (keep as float to avoid truncation)
            total_epochs = checkpoint_epoch + desired_new_epochs
            config["num_train_epochs"] = total_epochs
            print(f"📊 Adjusted num_train_epochs: {desired_new_epochs} → {total_epochs:.2f} (checkpoint was at epoch {checkpoint_epoch:.2f})")

            # CRITICAL FIX: Set max_steps explicitly to handle dataset size changes between iterations
            # When resuming from a checkpoint trained on a larger dataset (e.g., 65K samples, step 675)
            # to a smaller dataset (e.g., 8K samples, 246 steps total), HuggingFace Trainer will
            # exit immediately because checkpoint_global_step (675) > calculated_max_steps (246).
            # Solution: Set max_steps = checkpoint_global_step + steps_needed_for_new_epochs
            if num_train_samples and checkpoint_global_step > 0:
                # Calculate steps per epoch based on actual dataset size (using detected GPU count)
                per_device_batch_size = self.algorithm_config.per_device_train_batch_size
                grad_accum_steps = self.algorithm_config.gradient_accumulation_steps
                effective_batch_size = per_device_batch_size * grad_accum_steps * num_gpus
                steps_per_epoch = num_train_samples // effective_batch_size

                # Calculate steps needed for the desired new epochs
                desired_new_steps = int(steps_per_epoch * desired_new_epochs)

                # Set max_steps to checkpoint + new steps
                new_max_steps = checkpoint_global_step + desired_new_steps
                config["max_steps"] = new_max_steps

                print(f"📊 Dataset size changed between iterations - explicitly setting max_steps:")
                print(f"   Training samples: {num_train_samples}")
                print(f"   Effective batch size: {effective_batch_size} (per_device={per_device_batch_size} × grad_accum={grad_accum_steps} × {num_gpus} GPUs)")
                print(f"   Steps per epoch: {steps_per_epoch}")
                print(f"   Checkpoint global_step: {checkpoint_global_step}")
                print(f"   Steps for {desired_new_epochs} new epochs: {desired_new_steps}")
                print(f"   New max_steps: {new_max_steps} (checkpoint {checkpoint_global_step} + new {desired_new_steps})")
            elif checkpoint_global_step > 0:
                print(f"⚠️  Warning: num_train_samples not provided, cannot calculate max_steps adjustment")
                print(f"   Training may exit immediately if checkpoint step > calculated max_steps")

        # WandB configuration will be written to wandb_env.sh and sourced by run.sh
        full_project_path = f"{log_config.entity_name}/{project_name}"

        if wandb_run_id:
            print(f"📊 WandB will resume run '{run_name}' (id: {wandb_run_id}) in project '{full_project_path}'")
        else:
            print(f"📊 WandB will start new run '{run_name}' in project '{full_project_path}'")

        # Create configuration file in YAML format
        config_path = os.path.join(self.output_dir, "train_config.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"📝 Created LLaMA-Factory config: {config_path}")

        # Write WandB environment variables to a shell script for run.sh to source
        wandb_env_path = os.path.join(self.output_dir, "wandb_env.sh")
        with open(wandb_env_path, 'w') as f:
            f.write("# WandB environment variables (auto-generated by update_prepare.py)\n")
            f.write(f"export WANDB_PROJECT='{project_name}'\n")
            f.write(f"export WANDB_ENTITY='{log_config.entity_name}'\n")
            f.write(f"export WANDB_RUN_NAME='{run_name}'\n")
            f.write("export WANDB_RESUME='allow'\n")
            if wandb_run_id:
                f.write(f"export WANDB_RUN_ID='{wandb_run_id}'\n")
        print(f"📝 Created WandB env script: {wandb_env_path}")
        print(f"📁 Train dataset: {train_dataset_path}")
        if val_dataset_path:
            print(f"📁 Validation dataset: {val_dataset_path}")
        print(f"📊 Output directory: {output_dir}")

        return config_path

    def _get_model_template(self) -> str:
        """Get the appropriate template for the model (only Qwen3-VL supported)"""
        # Check if this is a Thinking variant by looking at the base model name
        base_model = self.agent.policy_config.base_model
        if 'Thinking' in base_model or 'thinking' in base_model.lower():
            return "qwen3_vl"  # Template for Qwen3-VL-Thinking (with thinking support)
        else:
            return "qwen3_vl_nothink"  # Template for Qwen3-VL-Instruct (no thinking)


@hydra.main(version_base=None, config_path="config/main", config_name="update")
def main(config: DictConfig):
    print("🚀 Starting single-pass data preparation for LLaMA-Factory training (no negative gradient)")

    # Handle HuggingFace login
    token_env_var = config.policy_config.huggingface_token_env_var
    token_value = os.environ.get(token_env_var)
    min_token_length = config.algorithm_config.min_token_length

    if token_value and len(str(token_value)) > min_token_length:
        try:
            login(token_value)
            # Suppress success message during update preparation
        except Exception as e:
            print(f"HuggingFace login failed: {e}")
    else:
        print(f"Warning: Invalid or missing HuggingFace token in {token_env_var}")

    # Get model_config from config
    model_config = dict(config.model_config) if hasattr(config, 'model_config') else {'model_type': 'qwen3-instruct'}

    # Add interaction_mode to model_config for convenience
    model_config['interaction_mode'] = config.context_config.get('interaction_mode', 'coordinates')

    # Create agent (no training mode needed for data preparation)
    agent = WebAgent(
        policy_config=config.policy_config,
        context_config=config.context_config,
        model_config=model_config,
        save_path=config.save_path,
        vllm_server_url=config.vllm_server_url,
        verbose=False  # Suppress verbose output during update preparation
    )

    # Load existing trajectories from save path using incremental loading
    # This loads iteration files from the train_trajectories/ directory
    # Only load last 4 iterations to save memory
    from webgym.utils import load_all_trajectories

    train_trajectories = load_all_trajectories(
        base_dir=config.save_path,
        split='train',
        last_n_iterations=4  # Only load last 4 iterations to save memory
    )

    if train_trajectories:
        # Filter out trajectories with None values and short trajectories (< 2 steps)
        original_count = len(train_trajectories)
        train_trajectories = clean_trajectories(train_trajectories)
        filtered_count = len(train_trajectories)

        print(f"Filtered out {original_count - filtered_count} trajectories (None values or length < 2) from {original_count} total")
        print(f"Using {filtered_count} valid trajectories (length >= 2, successful) for data preparation")
    else:
        print(f"No trajectories found in {config.save_path}/train_trajectories/")

    if not train_trajectories:
        print("❌ No training trajectories found. Cannot prepare data.")
        # Clean up old config to avoid using stale settings
        config_path = os.path.join(config.save_path, "llamafactory_data", "train_config.json")
        if os.path.exists(config_path):
            os.remove(config_path)
            print(f"🗑️ Removed stale config: {config_path}")
        sys.exit(1)

    # Create replay buffer with all data (sampling happens during prepare_all_data)
    filter_same_screenshot = getattr(config.algorithm_config, 'filter_same_screenshot', True)

    print(f"\n🔧 Training Config:")
    print(f"   filter_same_screenshot: {filter_same_screenshot}")
    print()

    replay_buffer = ReplayBuffer(
        agent=agent,
        trajectories=train_trajectories,
        capacity=None,  # Use all data
        filter_successful_only=True,
        include_reward_in_sample=False,
        shuffle=False,  # Keep chronological order for recency sampling
        filter_same_screenshot=filter_same_screenshot
    )

    colorful_print(f"Loaded {len(train_trajectories)} existing trajectories for data preparation from {config.save_path}", fg='green')

    # Create data preparation manager
    data_manager = DataPreparationManager(
        agent=agent,
        save_path=config.save_path,
        algorithm_config=config.algorithm_config
    )

    # Sample successful steps with recency bias for training
    positive_samples_to_train = getattr(config.algorithm_config, 'positive_samples_to_train', None)
    recency_bias_power = getattr(config.algorithm_config, 'recency_bias_power', 1.0)
    val_split_ratio = getattr(config.algorithm_config, 'val_split_ratio', 0.0)
    do_eval = getattr(config.algorithm_config, 'do_eval', True)

    # If evaluation is disabled, force val_split_ratio to 0 (don't split data)
    if not do_eval:
        val_split_ratio = 0.0
        print(f"📊 Evaluation disabled (do_eval=False) - using all data for training (no validation split)")

    # Print sampling info
    if positive_samples_to_train:
        print(f"📊 Will sample {positive_samples_to_train} successful steps with recency_bias_power={recency_bias_power}")
    else:
        print(f"📊 Will use all available successful steps")

    if val_split_ratio > 0:
        print(f"📊 Will split {val_split_ratio*100:.1f}% of data for validation")

    train_dataset_path, val_dataset_path = data_manager.prepare_all_data(
        replay_buffer,
        positive_samples_to_train=positive_samples_to_train,
        recency_bias_power=recency_bias_power,
        val_split_ratio=val_split_ratio
    )

    # Login to WandB if key is provided via environment variable
    wandb_key_env_var = config.log_config.get("wandb_key_env_var", "WANDB_API_KEY")
    wandb_key = os.environ.get(wandb_key_env_var)
    if wandb_key:
        wandb.login(key=wandb_key)
        print(f"✅ WandB login successful (using {wandb_key_env_var})")

    # Load the training dataset to get actual sample count for max_steps calculation
    num_train_samples = None
    if train_dataset_path and os.path.exists(train_dataset_path):
        try:
            with open(train_dataset_path, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
                num_train_samples = len(train_data)
                print(f"📊 Loaded training dataset: {num_train_samples} samples")
        except Exception as e:
            print(f"⚠️  Warning: Failed to load training dataset for sample count: {e}")

    # Create LLaMA-Factory training configuration with WandB settings
    config_path = data_manager.create_llamafactory_config(train_dataset_path, val_dataset_path, config.log_config, num_train_samples)

    print(f"\n✅ Data preparation completed successfully!")
    print(f"📁 Train dataset: {train_dataset_path}")
    if val_dataset_path:
        print(f"📁 Validation dataset: {val_dataset_path}")
    print(f"⚙️ Config: {config_path}")
    print(f"\n📋 Next steps:")
    print(f"   1. Review the generated dataset and config")
    print(f"   2. Run: llamafactory-cli train {config_path}")
    print(f"   3. Trained model will be copied to model.pt automatically")

if __name__ == "__main__":
    main()
