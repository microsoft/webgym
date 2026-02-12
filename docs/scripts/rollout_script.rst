Rollout Script: rollout.py (Optional Read)
==========================

The ``scripts/rollout.py`` script handles trajectory collection from web environments using a WebAgent.
It is the core data collection component of the RL pipeline.

Overview
--------

This script:

* Loads tasks from JSONL files and samples them for rollout
* Creates a WebAgent with vLLM inference backend
* Runs asynchronous trajectory collection via AsyncWebGym
* Saves trajectories incrementally with checkpoint callbacks
* Calculates and logs rollout metrics

The script is invoked by ``run.sh`` and uses Hydra for configuration.

Entry Point
-----------

.. code-block:: bash

   python scripts/rollout.py [hydra_overrides...]

In practice, ``run.sh`` always overrides with ``--config-name rollout_train`` or ``--config-name rollout_test``:

.. code-block:: bash

   python scripts/rollout.py --config-name rollout_train save_path=/data/exp1 data_path=/data/shared
   python scripts/rollout.py --config-name rollout_test  save_path=/data/exp1 data_path=/data/shared

Key Components
--------------

**WebAgent Initialization:**

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

**Task Sampling:**

* ``uniform``: Random uniform sampling
* ``ratio``: Ratio-based sampling
* Training tasks from blocked domains are filtered out; test tasks are not filtered

**Multi-Node Support:**

Resources are divided proportionally based on ``rank_load_weight``. Each node gets a slice of the server pool. Trajectory files are saved with rank suffixes (e.g., ``iteration5_rank_0.pt``).

.. code-block:: text

   Example: server_size=64, rank_weights="2,1,1" across 3 nodes
   Node 0 (weight 2): 32 instances, 50% of tasks
   Node 1 (weight 1): 16 instances, 25% of tasks
   Node 2 (weight 1): 16 instances, 25% of tasks

Metrics
-------

``calculate_rollout_metrics()`` computes: success_rate, avg_steps, avg_chars_per_response, goback_rate, crashed_rate, blocked_rate.

**Crash Detection** — a trajectory is "crashed" if:

1. Empty or has no steps
2. First action is ``invalid_url``
3. Contains dummy actions (None values)
4. Has fewer than 10 steps AND doesn't end with an answer AND is not blocked

Output
------

Trajectories are saved to ``<save_path>/train_trajectories/`` or ``<save_path>/test_trajectories/``:

.. code-block:: text

   <save_path>/
   ├── train_trajectories/
   │   ├── train_trajectories.pt.iteration0      # Single-node
   │   ├── train_trajectories.pt.iteration1
   │   └── train_trajectories.pt.iteration2
   └── test_trajectories/
       ├── test_trajectories.pt.iteration1_rank_0   # Multi-node: per-rank files
       ├── test_trajectories.pt.iteration1_rank_1
       └── test_trajectories.pt.iteration1           # Aggregated by master
