Update Script: update_prepare.py
=================================

The ``scripts/update_prepare.py`` script prepares training data from collected trajectories
and generates LLaMA-Factory configuration for model fine-tuning.

Overview
--------

This script:

* Loads trajectories from the replay buffer
* Converts successful trajectories to LLaMA-Factory ShareGPT format
* Applies recency bias for sample weighting
* Creates dataset files and training configuration
* Sets up WandB logging for training runs

The script is invoked by ``run.sh`` during the update phase.

Entry Point
-----------

.. code-block:: bash

   python scripts/update_prepare.py [hydra_overrides...]

The script uses the Hydra config name ``update`` by default. In practice, ``run.sh`` always overrides this with ``--config-name update_online`` to use ``scripts/config/main/update_online.yaml``.

.. note::
   Running ``python scripts/update_prepare.py`` directly without ``--config-name update_online`` will fail because no ``update.yaml`` exists. Always use ``run.sh`` or pass ``--config-name update_online`` explicitly.

Key Components
--------------

DataPreparationManager
^^^^^^^^^^^^^^^^^^^^^^

The main class that handles data preparation:

.. code-block:: python

   data_manager = DataPreparationManager(
       agent=agent,
       save_path=config.save_path,
       algorithm_config=config.algorithm_config
   )

**Key Methods:**

* ``prepare_all_data()``: Samples and converts trajectories to training format
* ``create_llamafactory_config()``: Generates YAML config for LLaMA-Factory
* ``_convert_to_llamafactory_format()``: Converts single samples to ShareGPT format

Data Preparation Pipeline
-------------------------

1. Load Trajectories
^^^^^^^^^^^^^^^^^^^^

Trajectories are loaded from ``<save_path>/train_trajectories/``:

.. code-block:: python

   train_trajectories = load_all_trajectories(
       base_dir=config.save_path,
       split='train',
       last_n_iterations=4  # Memory optimization
   )

2. Clean Trajectories
^^^^^^^^^^^^^^^^^^^^^

Invalid trajectories are filtered out:

* Empty trajectories
* Trajectories with fewer than 2 steps
* Trajectories with None values in action/observation/response

3. Create Replay Buffer
^^^^^^^^^^^^^^^^^^^^^^^

The ReplayBuffer handles trajectory sampling:

.. code-block:: python

   replay_buffer = ReplayBuffer(
       trajectories=train_trajectories,
       agent=agent,
       filter_successful_only=True,
       filter_same_screenshot=True
   )

4. Sample Training Data
^^^^^^^^^^^^^^^^^^^^^^^

Positive samples are extracted with recency bias:

.. code-block:: python

   all_sampled = replay_buffer.sample_with_recency(
       num_samples=samples_to_train,
       recency_bias_power=recency_bias_power
   )
   positive_samples = [s for s in all_sampled if s['loss_weight'] > 0]

5. Convert to ShareGPT Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each sample is converted to LLaMA-Factory's ShareGPT format:

.. code-block:: json

   {
     "conversations": [
       {"from": "human", "value": "<image>Task: ..."},
       {"from": "gpt", "value": "Action: click(...)"}
     ],
     "system": "You are a web agent...",
     "images": ["/path/to/screenshot.png"],
     "loss_weight": 1.0
   }

LLaMA-Factory Configuration
---------------------------

The script generates a complete training configuration:

**Training Hyperparameters:**

.. code-block:: yaml

   stage: sft
   do_train: true
   finetuning_type: full
   mask_history: true  # Only train on last turn
   cutoff_len: 16384
   per_device_train_batch_size: 3
   gradient_accumulation_steps: 4
   learning_rate: 1e-6
   num_train_epochs: 2

**Checkpoint Resume:**

The script automatically detects and resumes from checkpoints:

* Checks for ``trainer_state.json`` and optimizer states
* Updates scheduler learning rate if config changed
* Adjusts ``max_steps`` for dataset size changes between iterations

**WandB Integration:**

.. code-block:: yaml

   report_to: wandb
   run_name: webgym-<your-run-name>

Environment variables are written to ``wandb_env.sh`` for ``run.sh`` to source.

Configuration
-------------

Key configuration options in ``update_online.yaml``:

**Log Config:**

.. code-block:: yaml

   log_config:
     run_name: 'webgym-<your-run-name>'
     wandb_key_env_var: "WANDB_API_KEY"
     entity_name: "<your-wandb-entity-name>"

**Algorithm Config:**

.. code-block:: yaml

   algorithm_config:
     model_output_name: "model.pt"
     samples_to_train: 16384
     recency_bias_power: 2
     val_split_ratio: 0.05

     # Training hyperparameters
     cutoff_len: 16384
     per_device_train_batch_size: 3
     per_device_eval_batch_size: 3
     gradient_accumulation_steps: 4
     learning_rate: 1e-6
     max_grad_norm: 1.0
     weight_decay: 0.01
     num_train_epochs: 2
     warmup_steps: 30
     lr_scheduler_type: "constant_with_warmup"
     logging_steps: 1
     bf16: True

     # Evaluation
     do_eval: false
     eval_strategy: "epoch"

     # Save settings
     save_strategy: "steps"
     save_steps: 999999
     save_total_limit: 1
     save_only_model: False

     # Data loading
     preprocessing_num_workers: 16
     dataloader_num_workers: 2
     dataloader_pin_memory: True
     remove_unused_columns: False
     min_token_length: 10

     # Other
     gradient_checkpointing: False
     plot_loss: False
     deepspeed_config_filename: "ds_config_b200_zero2.json"
     report_to: "wandb"

Output
------

The script generates files in ``<save_path>/llamafactory_data/``:

* ``finetune_train.json``: Training dataset in ShareGPT format
* ``finetune_val.json``: Validation dataset (if ``val_split_ratio`` > 0)
* ``dataset_info.json``: Dataset registry for LLaMA-Factory
* ``train_config.yaml``: Complete training configuration
* ``wandb_env.sh``: WandB environment variables

Next Steps
----------

After ``update_prepare.py`` completes, ``run.sh`` executes:

.. code-block:: bash

   # Source WandB environment
   source llamafactory_data/wandb_env.sh

   # Run LLaMA-Factory training
   llamafactory-cli train llamafactory_data/train_config.yaml

The trained model is first saved to ``<save_path>/checkpoints/model_<timestamp>/``, then the final checkpoint is copied to ``<save_path>/model.pt/``.
