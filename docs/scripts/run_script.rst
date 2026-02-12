Entry Script: run.sh
====================

.. note::

   It is highly recommended to read through this documentation before launching your first RL run.

The ``scripts/run.sh`` script orchestrates distributed RL training with vLLM inference and LLaMA-Factory training.

Quick Start
-----------

.. code-block:: bash

   bash scripts/run.sh --data-path <path> --log-path <path> --rl-phase <phase> [options] [config_overrides...]

The script uses absolute paths internally and can be run from any directory.

**The three core commands:**

.. code-block:: bash

   # 1. Complete RL loop (rollout + training), eval every 6 iterations
   bash scripts/run.sh --data-path /home/user --log-path /home/user/exp1 --rl-phase both --eval-interval 6

   # 2. Rollout only - collect train trajectories (no training)
   bash scripts/run.sh --data-path /home/user --log-path /home/user/exp1 --rl-phase rollout --rollout-split train-only

   # 3. Training only (train on existing trajectories)
   bash scripts/run.sh --data-path /home/user --log-path /home/user/exp1 --rl-phase update

**Customize with config overrides** (any ``key=value`` argument is passed to Hydra):

.. code-block:: bash

   bash scripts/run.sh --data-path /home/user --log-path /home/user/exp1 --rl-phase rollout \
       env_config.train_tasks=my_tasks.jsonl \
       env_config.train_tasks_rollout_size=100 \
       openai_config.model=gpt-4o

All phases run in an infinite loop until manually stopped (Ctrl+C) or a failure occurs.

**Core Contribution:** The asynchronous rollout engine is the central component of this work.
It is designed to be portable and can be integrated with other RL frameworks beyond the
LLaMA-Factory training pipeline used here.

Required Arguments
------------------

``--data-path <path>``
   Read-only shared data directory (absolute path). HuggingFace cache stored at ``<data-path>/.cache/huggingface/hub/``.

``--log-path <path>``
   Experiment-specific logs directory (absolute path). Contains trajectories, checkpoints, and model weights.

``--rl-phase <phase>``
   One of:

   * ``rollout`` - Data collection only (requires ``--rollout-split``)
   * ``update`` - Training only
   * ``both`` - Complete RL loop (requires ``--eval-interval``)

RL Phases
---------

**rollout:** Starts vLLM, then loops: collect train or eval trajectories (per ``--rollout-split``).

**update:** Loops: wait for GPU memory clear, prepare data, run LLaMA-Factory training, clear GPU.

**both:** Starts vLLM, then loops: collect train trajectories, optionally collect eval (per ``--eval-interval``), stop vLLM, train, restart vLLM with updated model.

Options
-------

``--eval-interval <N>``
   Required for ``--rl-phase both``. Runs evaluation when ``iteration % N == 1`` (e.g., ``--eval-interval 6`` evaluates on iterations 1, 7, 13, ...).

``--rollout-split <split>``
   Required for ``--rl-phase rollout``. Either ``train-only`` or ``eval-only``.

``--debug-mode``
   Skips vLLM lifecycle management. Assumes vLLM is already running externally. Start vLLM manually:

   .. code-block:: bash

      vllm serve /home/user/exp1/model.pt \
          --host 0.0.0.0 --port 8999 --max-num-seqs 512 \
          --gpu-memory-utilization 0.95 --max-model-len 32768 \
          --tensor-parallel-size 1 --data-parallel-size ${NUM_GPUS} \
          --limit-mm-per-prompt '{"video": 0}' \
          --allowed-local-media-path /home/user/exp1

   If no checkpoint exists, use the HuggingFace model (e.g., ``Qwen/Qwen3-VL-8B-Instruct``) instead.

``--resume``
   Resume rollout from an existing checkpoint file. Requires ``--rollout-split``.
   **Single-node only** — multi-node resume is not supported.

   * Loads completed trajectories from ``.checkpoint`` file and resumes collection
   * If no checkpoint exists, exits with an error
   * Without ``--resume``, the script starts fresh and **overwrites** any existing checkpoint


Argument Combinations
---------------------

Both ``--log-path`` and ``--data-path`` are always required. Config overrides are always optional.

.. list-table::
   :header-rows: 1
   :widths: 12 12 12 12 12 12 12 12

   * - --rl-phase
     - --eval-interval
     - --rollout-split
     - --num-nodes
     - --rank-weights
     - --master/--worker
     - --debug-mode
     - --resume
   * - rollout
     - N/A
     - **Required**
     - Optional
     - If multi-node
     - If multi-node
     - Optional
     - Optional (single-node)
   * - update
     - N/A
     - N/A
     - Optional
     - If multi-node
     - If multi-node
     - Optional
     - N/A
   * - both
     - **Required**
     - N/A
     - Optional
     - If multi-node
     - If multi-node
     - Optional
     - N/A

Config Overrides
----------------

Any ``key=value`` arguments are passed to Hydra:

.. code-block:: bash

   # Common overrides
   env_config.train_tasks=my_tasks.jsonl
   env_config.train_tasks_rollout_size=50
   env_config.server_size=30
   openai_config.model=gpt-4o
   model_config.model_type=qwen3-think
   policy_config.temperature=0.8

Multi-Node
----------

See :doc:`run_script_multinode` for multi-node setup, coordination protocol, and fault tolerance.

Built-in Configuration
-----------------------

Model and vLLM settings are hardcoded in ``run.sh`` and ``scripts/shell_functions/vllm_utils.sh``.

**Model type** (set in ``run.sh``):

.. code-block:: bash

   MODEL_TYPE="qwen-instruct"    # Options: "qwen-instruct", "qwen-think"

Maps to: ``qwen-instruct`` → ``Qwen/Qwen3-VL-8B-Instruct``, ``qwen-think`` → ``Qwen/Qwen3-VL-8B-Thinking``

.. note::
   Shell scripts use ``qwen-instruct`` / ``qwen-think`` (without "3"). Hydra config uses
   ``qwen3-instruct`` / ``qwen3-think`` (with "3"). These should match.

**vLLM** (set in ``vllm_utils.sh``):

.. code-block:: bash

   vllm serve "${MODEL_TO_SERVE}" \
       --host 0.0.0.0 --port "${VLLM_PORT}" \
       --max-num-seqs 512 --gpu-memory-utilization 0.95 \
       --max-model-len 32768 --tensor-parallel-size 1 \
       --data-parallel-size ${NUM_GPUS} \
       --limit-mm-per-prompt '{"video": 0}' \
       --allowed-local-media-path "${HOST_DATA_PATH}"

Helper Functions
----------------

Located in ``scripts/shell_functions/``:

* ``vllm_utils.sh`` — ``start_vllm``, ``stop_vllm``, ``ensure_vllm_running``
* ``gpu_utils.sh`` — ``wait_for_gpu_memory_clear``, ``cleanup_deepspeed_processes``
* ``training_utils.sh`` — ``run_llamafactory_training``, ``update_atomic``
* ``rollout_utils.sh`` — ``rollout_atomic_train``, ``rollout_atomic_test`` (both support ``--resume`` via ``RESUME_CHECKPOINT_PATH``)
* ``common_utils.sh`` — ``resolve_model_to_serve``, ``detect_num_gpus``, ``find_last_checkpoint``
* ``multinode_sync.sh`` — ``create_phase_flag``/``wait_for_phase``, ``aggregate_trajectories``

Environment Variables
---------------------

* ``HF_HOME`` — HuggingFace cache directory (set in ``run.sh``)
* ``DISABLE_VERSION_CHECK=1`` — Disables LLaMA-Factory version check (set in ``run.sh``)
* ``VLLM_USE_TRITON_FLASH_ATTN=1`` — Flash attention (set in ``vllm_utils.sh``)
* ``DEEPSPEED_LOG_LEVEL=WARNING`` — Suppresses verbose messages (set in ``training_utils.sh``)
* ``WANDB_MODE=online`` — Real-time WandB syncing (set in ``training_utils.sh``)
* ``HYDRA_OVERRIDES`` — Config overrides from CLI

Stopping the Script
-------------------

Once rollout has started, Ctrl+C may not fully terminate subprocesses. To fully stop:

.. code-block:: bash

   pkill -9 python   # Kill rollout subprocesses first, use sudo if necessary
   # Then Ctrl+C the main bash script

Troubleshooting
---------------

.. _vllm-server-errors:

vLLM Server Errors (HTTP 500)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you see ``vLLM server returned status 500`` errors, the server is overloaded.

**Solution:** Reduce ``server_size`` (note that ``max_vllm_sessions`` is experimental and not supported yet, so use ``server_size`` instead).
