Rollout Script: rollout.py
==========================

The ``scripts/rollout.py`` script handles trajectory collection from web environments using a WebAgent.
It is the core data collection component of the RL pipeline.

Overview
--------

This script:

* Loads tasks from JSONL files and samples them for rollout
* Creates a WebAgent with vLLM inference backend
* Runs asynchronous trajectory collection via AsyncWebGym
* Saves trajectories incrementally to avoid memory issues
* Calculates and logs rollout metrics (success rate, crash rate, etc.)

The script is invoked by ``run.sh`` during the rollout phase and uses Hydra for configuration management.

Entry Point
-----------

.. code-block:: bash

   python scripts/rollout.py [hydra_overrides...]

The script uses the ``rollout`` config from ``scripts/config/main/rollout.yaml`` as the base config. In practice, ``run.sh`` always overrides this with ``--config-name rollout_train`` or ``--config-name rollout_test``.

Key Components
--------------

main_logic()
^^^^^^^^^^^^

The main execution function that:

1. Sets a random seed (timestamp-based for variety between iterations)
2. Initializes the WebAgent with vLLM server connection
3. Loads and filters tasks (blocked domains are excluded for training)
4. Samples tasks using the configured sampler (uniform or ratio-based)
5. Creates AsyncWebGym environment with retry policies
6. Runs trajectory collection with checkpoint callbacks
7. Saves trajectories incrementally

WebAgent Initialization
^^^^^^^^^^^^^^^^^^^^^^^

The WebAgent is created with:

* Policy configuration (model, temperature, max tokens)
* Context configuration (interaction mode: coordinates or set-of-marks)
* Model configuration (model type: qwen3-instruct or qwen3-think)
* OpenAI configuration (for evaluation/judging)
* vLLM server URL and timeout settings

.. code-block:: python

   agent = WebAgent(
       policy_config=config.policy_config,
       context_config=config.context_config,
       model_config=model_config,
       save_path=config.save_path,
       vllm_server_url=vllm_server_url,
       openai_config=config.openai_config,
       ...
   )

Task Sampling
^^^^^^^^^^^^^

Tasks are loaded from JSONL files and sampled based on the configured strategy:

* **uniform**: Random uniform sampling from all tasks
* **ratio**: Ratio-based sampling (configurable)

For training, tasks from blocked domains are filtered out. Test tasks are not filtered.

.. code-block:: python

   sampled_tasks = rollout_sampler.sample_tasks(
       tasks,
       config.env_config.split,
       ...
   )

Trajectory Collection
^^^^^^^^^^^^^^^^^^^^^

Trajectories are collected using AsyncWebGym with:

* Configurable number of parallel workers (``server_size``)
* HTTP connection pools for different operations
* Task timeout and completion thresholds
* Checkpoint callbacks for incremental saving

.. code-block:: python

   new_trajectories = env.run_automation_with_fairness(
       agent,
       progress_callback=None,
       checkpoint_callback=checkpoint_callback,
       checkpoint_interval=0.25,
       ...
   )

Multi-Node Support
^^^^^^^^^^^^^^^^^^

The script supports multi-node distributed rollout:

* Resources are divided proportionally based on ``rank_load_weight``
* Each node gets a slice of the server pool and HTTP connections
* Trajectory files are saved with rank suffixes (e.g., ``iteration5_rank_0.pt``)

Metrics Calculation
-------------------

The ``calculate_rollout_metrics()`` function computes:

* **success_rate**: Fraction of successful trajectories (non-crashed, non-blocked)
* **avg_steps**: Average number of steps per trajectory
* **avg_chars_per_response**: Average response length
* **goback_rate**: Fraction of steps using the "goback" action
* **crashed_rate**: Fraction of trajectories that crashed
* **blocked_rate**: Fraction of trajectories blocked by CAPTCHA/etc.

Crash Detection
^^^^^^^^^^^^^^^

A trajectory is considered "crashed" if:

1. It's empty or has no steps
2. First action is ``invalid_url``
3. Contains dummy actions (None values)
4. Has fewer than 10 steps AND doesn't end with an answer action AND is not blocked

Configuration
-------------

Key configuration options in ``rollout.yaml``:

**Environment Config:**

.. code-block:: yaml

   env_config:
     split: "train"  # or "test"
     server_size: 64  # parallel browser instances
     train_tasks_rollout_size: 1024  # tasks per rollout
     train_tasks_sampler: "uniform"  # or "ratio"
     save_traj_progress: true  # checkpoint saving
     save_every_percent: 25  # checkpoint interval (percentage)

**Policy Config:**

.. code-block:: yaml

   policy_config:
     base_model: null  # set by run.sh
     max_new_tokens: 3072
     temperature: 1
     top_p: 0.99

**HTTP Pool Configuration:**

.. code-block:: yaml

   env_config:
     http_pools:
       metadata: 128          # Screen dimensions (once per task)
       navigate: 128          # Initial navigation
       allocate: 4            # Instance allocation
       release: 4             # Instance cleanup
       screenshot: 128        # Screenshot capture (per step)
       ac_tree: 128           # Accessibility tree (per step)
       page_metadata: 128     # Page title/URL (per step)
       execute: 128           # Action execution (per step)

Output
------

Trajectories are saved to:

* ``<save_path>/train_trajectories/`` for training data
* ``<save_path>/test_trajectories/`` for evaluation data

Files are saved incrementally with iteration numbers:

* ``iteration0.pt``, ``iteration1.pt``, etc.
* Multi-node: ``iteration0_rank_0.pt``, ``iteration0_rank_1.pt``, etc.

Each trajectory file contains:

* List of trajectories (each trajectory is a list of steps)
* Metadata (model info, dataset info, split, etc.)
