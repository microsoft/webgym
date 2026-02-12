Configuration Reference
=======================

WebGym Configs
--------------

The system uses `Hydra <https://hydra.cc/>`_ for hierarchical configuration management.
Config files are located in ``scripts/config/main/``.

**Config File Hierarchy:**

.. code-block:: text

   default.yaml          # Base config (paths, policy, context)
       ├── rollout.yaml      # Base rollout config (env, timeouts, workers)
       │   ├── rollout_train.yaml   # Train rollout (train tasks, difficulty)
       │   └── rollout_test.yaml    # Eval rollout (test tasks, difficulty)
       └── update_online.yaml       # Training config (hyperparameters, data sampling)

Each config file uses ``defaults:`` to inherit from parent configs. Settings in child
configs override parent values. For example, ``rollout_train.yaml`` inherits from
``rollout.yaml`` which inherits from ``default.yaml``:

.. code-block:: yaml

   # default.yaml defines base policy settings:
   policy_config:
     temperature: 1
     max_new_tokens: 3072

   # rollout.yaml inherits default.yaml and adds env settings:
   defaults:
     - default
   env_config:
     server_size: 64

   # rollout_train.yaml inherits default.yaml and rollout.yaml, then overrides specifics:
   defaults:
     - default
     - rollout
   env_config:
     split: "train"
     train_tasks_rollout_size: 1024
   # temperature and server_size are still inherited from the parents

``save_path``
   Root directory for outputs (set via ``--log-path``)

``data_path``
   Path to read-only shared data (task files, HuggingFace cache, etc.; set via ``--data-path``)

``model_config``
   Model selection:

   * ``model_type``: Model type (``qwen3-instruct`` or ``qwen3-think``)
   * ``prompt_version``: Prompt version (``vanilla`` or ``complete``)

``policy_config``
   Model inference settings:

   * ``base_model``: HuggingFace model name (set by run.sh)
   * ``checkpoint_path``: Path to model checkpoint
   * ``max_new_tokens``: Maximum tokens to generate (default: 3072)
   * ``temperature``: Sampling temperature (default: 1)
   * ``top_p``: Top-p sampling (default: 0.99)
   * ``top_k``: Top-k sampling (default: 2)
   * ``min_p``: Min-p sampling (default: 0)

``context_config``
   Interaction mode settings:

   * ``interaction_mode``: How to represent UI elements (``coordinates`` or ``set_of_marks``)

``env_config``
   Environment and HTTP settings:

   * ``vllm_server_url``: vLLM endpoint (default: ``http://localhost:8999``)
   * ``wait_timeout``: HTTP queue timeout in seconds (default: 2400)
   * ``operation_timeout``: HTTP operation timeout (default: 120)
   * ``vllm_timeout``: vLLM request timeout (default: 240)
   * ``evaluation_workers``: Concurrent trajectory evaluation workers (default: 16)
   * ``screenshot_comparison_workers``: Concurrent screenshot comparison workers (default: 32)
   * ``completion_threshold``: Fraction of tasks to complete before killing stragglers (default: 0.95).
     For example, with 1024 tasks and ``completion_threshold=0.95``, once 973 tasks finish, the remaining
     51 slow tasks enter the grace period.
   * ``completion_grace_period``: Seconds to wait after the threshold is reached before force-killing
     remaining tasks (default: 30). Continuing the example above, the 51 remaining tasks get 30 more
     seconds to finish; any still running after that are terminated.
   * ``instance_lifetime_max``: Maximum browser instance lifetime in minutes (default: 50)
   * ``task_timeout_minutes``: Maximum task timeout in minutes (default: 300)
   * ``server_size``: Browser instances per batch (default: 64)
   * ``verbose``: Enable verbose logging (default: False)
   * ``use_rich_actree``: Use rich accessibility tree format (default: False)
   * ``max_retries``: Maximum HTTP retries (default: 2)
   * ``http_pools``: Connection pool sizes per operation type. Train and test configs use different pool sizes:

     .. code-block:: yaml

        # Train (rollout_train.yaml):
        http_pools:
          # Once per task
          metadata: 128          # Screen dimensions
          navigate: 128          # Initial navigation to website
          allocate: 4            # Instance allocation
          release: 4             # Instance cleanup/release
          # Once per step
          screenshot: 128        # Screenshot capture (most frequent, can be slow)
          ac_tree: 128           # Accessibility tree retrieval
          page_metadata: 128     # Page title/URL (once per step)
          execute: 128           # Action execution (click, type, scroll)

        # Test (rollout_test.yaml):
        http_pools:
          navigate: 32           # Initial navigation to website
          screenshot: 128        # Screenshot capture
          ac_tree: 64            # Accessibility tree retrieval
          metadata: 32           # Screen dimensions
          page_metadata: 64      # Page title/URL (once per step)
          execute: 64            # Action execution (click, type, scroll)
          allocate: 4            # Instance allocation
          release: 4             # Instance cleanup/release
   * ``max_vllm_sessions``: Concurrent vLLM requests (default: 128 train, 64 test) (experimental, not supported yet, so use ``server_size`` instead)
   * ``split``: Dataset split (``train`` or ``test``)
   * ``train_difficulty_max_steps``: Max steps per difficulty (easy: 10, medium: 20, hard: 30)
   * ``test_difficulty_max_steps``: Max steps per difficulty (easy: 30, medium: 50, hard: 70)
   * ``train_tasks_rollout_size``: Tasks per train batch (default: 1024)
   * ``test_tasks_rollout_size``: Tasks per eval batch (default: -1 for all in test config, 0 in train config)
   * ``test_tasks_repeat_times``: Repeat each test task N times (default: 1 in test config, 0 in train config)
   * ``train_tasks_sampler``: Task sampling strategy (``uniform`` or ``ratio``)
   * ``train_tasks``: Train task file (default: ``train.jsonl``)
   * ``test_tasks``: Test task file (default: ``test.jsonl``)

``openai_config``
   Evaluator API settings (supports OpenAI and Gemini). Supports per-task model configuration.

   **Default settings** (apply to all tasks unless overridden):

   * ``model``: Default evaluation model (default: ``gemini-3-flash-preview``)
   * ``openai_api_key_env_var``: Environment variable for API key (default: ``GEMINI_API_KEY``)
   * ``base_url``: Base URL for API (default: ``https://generativelanguage.googleapis.com/v1beta/openai/``)

   **Per-task overrides** (optional, each can specify ``model``, ``openai_api_key_env_var``, ``base_url``):

   * ``keypoint_detection``: For judging which screenshots to submit (N-1 calls per trajectory)
   * ``blocking_detection``: For detecting CAPTCHA/blocking pages (1 call per trajectory)
   * ``evaluation``: For criterion_a, criterion_b, and reference_answer (2 + R calls per trajectory)

   **Supported Gemini Models:**

   * ``gemini-3-flash-preview``: More capable model for complex evaluation tasks
   * ``gemini-2.5-flash-lite``: Faster, cheaper model for high-volume operations (keypoint detection, blocking detection)

   **Example: Using Gemini with per-task configuration (default):**

   .. code-block:: yaml

      openai_config:
        # Default configuration for evaluation
        model: "gemini-3-flash-preview"
        openai_api_key_env_var: "GEMINI_API_KEY"
        base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"

        # Use lighter model for high-volume operations
        keypoint_detection:
          model: "gemini-2.5-flash-lite"
        blocking_detection:
          model: "gemini-2.5-flash-lite"
        evaluation:
          model: "gemini-3-flash-preview"

   **Example: Using different providers per task:**

   .. code-block:: yaml

      openai_config:
        # Default: OpenAI for evaluation
        model: "gpt-4o-mini"
        openai_api_key_env_var: "OPENAI_API_KEY"

        # Use Gemini for high-volume keypoint detection (cheaper)
        keypoint_detection:
          model: "gemini-3-flash-preview"
          openai_api_key_env_var: "GEMINI_API_KEY"
          base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"

   Set your API keys: ``export GEMINI_API_KEY="your-key"`` and/or ``export OPENAI_API_KEY="your-key"``

``log_config``
   Logging settings:

   * ``run_name``: WandB run name prefix
   * ``wandb_key_env_var``: Environment variable name containing WandB API key (default: ``WANDB_API_KEY``)
   * ``entity_name``: WandB entity/team name

``algorithm_config``
   Training hyperparameters:

   * ``positive_samples_to_train``: Positive samples per training iteration (default: 1800)
   * ``recency_bias_power``: Bias toward recent trajectories (default: 2)
   * ``cutoff_len``: Maximum sequence length (default: 16384)
   * ``per_device_train_batch_size``: Batch size per GPU (default: 3)
   * ``gradient_accumulation_steps``: Gradient accumulation (default: 4)
   * ``learning_rate``: Learning rate (default: 1e-6)
   * ``num_train_epochs``: Epochs per iteration (default: 2)
   * ``warmup_steps``: LR warmup steps (default: 30)
   * ``lr_scheduler_type``: Scheduler type (default: ``constant_with_warmup``)
   * ``val_split_ratio``: Validation data fraction (default: 0.05)
   * ``deepspeed_config_filename``: DeepSpeed config file (default: ``ds_config_b200_zero1.json``)
   * ``report_to``: Logging backend (default: ``wandb``)

DeepSpeed Configs
-----------------

DeepSpeed config files are located in ``scripts/config/deepspeed/``. These control
distributed training memory optimization and offloading strategies.

**Available Configurations:**

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - File
     - ZeRO Stage
     - Use Case
   * - ``ds_config_h100_zero1.json``
     - 1
     - H100 GPUs, optimizer state partitioning only
   * - ``ds_config_h100_zero2.json``
     - 2
     - H100 GPUs, optimizer + gradient partitioning
   * - ``ds_config_h100_zero3.json``
     - 3
     - H100 GPUs, full parameter partitioning + CPU offload
   * - ``ds_config_b200_zero1.json``
     - 1
     - B200 GPUs, optimizer state partitioning only (default)
   * - ``ds_config_b200_zero2.json``
     - 2
     - B200 GPUs, optimizer + gradient partitioning
   * - ``ds_config_b200_zero3.json``
     - 3
     - B200 GPUs, full partitioning

**ZeRO Stages:**

* **Stage 1** - Optimizer state partitioning. Lowest memory savings, highest speed.
* **Stage 2** - Optimizer + gradient partitioning. Moderate memory savings.
* **Stage 3** - Full parameter partitioning. Maximum memory savings. CPU offload is enabled in the H100 config but not the B200 config.

**Key Settings:**

``bf16.enabled``
   Use bfloat16 precision (recommended for H100/B200)

``activation_checkpointing``
   Recompute activations during backward pass to save memory

``zero_optimization.stage``
   ZeRO optimization level (1, 2, or 3)

``zero_optimization.offload_param``
   Offload model parameters to CPU (ZeRO-3 only)

``zero_optimization.offload_optimizer``
   Offload optimizer states to CPU

``overlap_comm``
   Overlap communication with computation for better throughput

To use a different config, modify ``update_online.yaml``:

.. code-block:: yaml

   algorithm_config:
     deepspeed_config_filename: "ds_config_h100_zero2.json"  # Use ZeRO-2 instead
