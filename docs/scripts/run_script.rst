Entry Script: run.sh
====================

.. note::

   It is highly recommended to read through this documentation before launching your first RL run.

The ``scripts/run.sh`` script is the main orchestration script for running distributed reinforcement learning training with vLLM inference and LLaMA-Factory training.

Overview
--------

This script handles:

* RL phases: rollout (data collection), update (training), or both
* Multi-node distributed execution with automatic coordination
* GPU memory management and cleanup
* vLLM server lifecycle management
* Trajectory collection and aggregation

**Core Contribution:** The asynchronous rollout engine is the central component of this work.
It is designed to be portable and can be integrated with other RL frameworks beyond the
LLaMA-Factory training pipeline used here.

Usage
-----

.. code-block:: bash

   bash scripts/run.sh --data-path <path> --log-path <path> --rl-phase <phase> [options] [config_overrides...]

The script can be run from any directory - it uses absolute paths internally to locate
helper scripts and configuration files.

Any additional arguments containing ``=`` are passed as Hydra config overrides
(see `Config Overrides`_ section below).

**Note:** All phases run in an infinite loop until manually stopped (Ctrl+C) or a failure occurs.
There is no maximum iteration limit.

Quick Start Examples
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Complete RL loop (rollout + training), eval every 6 iterations
   bash scripts/run.sh --data-path /home/user --log-path /home/user/exp1 --rl-phase both --eval-interval 6

   # Rollout only - train trajectories (no training)
   bash scripts/run.sh --data-path /home/user --log-path /home/user/exp1 --rl-phase rollout --rollout-split train-only

   # Training only (train on existing trajectories)
   bash scripts/run.sh --data-path /home/user --log-path /home/user/exp1 --rl-phase update

   # Multi-node (2 nodes) complete RL loop - on master node
   bash scripts/run.sh --data-path /home/user --log-path /home/user/exp1 --rl-phase both --eval-interval 6 --num-nodes 2 --rank-weights "1,1" --master

   # Multi-node (2 nodes) complete RL loop - on worker node (master IP auto-discovered from shared filesystem)
   bash scripts/run.sh --data-path /home/user --log-path /home/user/exp1 --rl-phase both --eval-interval 6 --num-nodes 2 --rank-weights "1,1" --worker 1

   # Override config values (custom dataset, larger rollout, GPT-4o eval)
   bash scripts/run.sh --data-path /home/user --log-path /home/user/exp1 --rl-phase rollout \
       env_config.train_tasks=my_tasks.jsonl \
       env_config.train_tasks_rollout_size=100 \
       openai_config.model=gpt-4o

Required Arguments
------------------

``--data-path <path>``
   Path to shared data directory. This is read-only and shared across experiments.

   * HuggingFace cache will be in ``<data-path>/.cache/huggingface/hub/``

   **Note:** This argument is always required for all phases. The path must be an **absolute path** (e.g., ``/data/shared``, not ``./data/shared``).

``--log-path <path>``
   Path to experiment logs directory. This is experiment-specific and can differ per run.
   The path must be an **absolute path** (e.g., ``/data/exp1``, not ``./exp1``).

   * Trajectories stored in ``<log-path>/train_trajectories/`` and ``<log-path>/test_trajectories/``
   * Checkpoints stored in ``<log-path>/checkpoints/``
   * Model weights stored in ``<log-path>/model.pt``

``--rl-phase <phase>``
   The RL phase to execute. Must be one of:

   * ``rollout`` - Data collection only (train and/or eval trajectories)
   * ``update`` - Training updates only (no data collection)
   * ``both`` - Complete RL loop: train collect → eval collect → update → repeat

RL Phases
---------

rollout
^^^^^^^

Runs trajectory collection in an infinite loop.

**Requires** ``--rollout-split`` to specify either train-only or eval-only collection.

Workflow:

1. Starts vLLM server
2. Infinite loop:

   * If ``--rollout-split train-only``: Collect train trajectories only
   * If ``--rollout-split eval-only``: Collect eval trajectories only
   * Repeat

update
^^^^^^

Runs training updates in an infinite loop.

Workflow:

1. Infinite loop:

   * Wait for GPU memory to clear
   * Prepare data and run LLaMA-Factory training
   * Clear GPU memory

both
^^^^

Runs the complete RL loop: data collection followed by training.

**Requires** ``--eval-interval`` to specify evaluation frequency.

Workflow:

1. Starts vLLM server
2. Infinite loop:

   * Collect train trajectories
   * Collect eval trajectories (every N iterations based on ``--eval-interval``)
   * Stop vLLM, run training update
   * Restart vLLM with updated model

Options
-------

``--eval-interval <N>``
   **Required when** ``--rl-phase both``.

   Specifies how often to run evaluation rollouts. Evaluation runs when
   ``iteration % N == 1``.

   Example: ``--eval-interval 6`` runs evaluation on iterations 1, 7, 13, 19, ...

``--rollout-split <split>``
   **Required when** ``--rl-phase rollout``.

   Specifies which type of trajectories to collect:

   * ``train-only`` - Only collect train trajectories (train→train→train)
   * ``eval-only`` - Only collect eval trajectories (eval→eval→eval)

``--num-nodes <N>``
   Number of nodes for distributed execution. Default: 1 (single-node).

   When N > 1:

   * First node becomes master (rank 0)
   * Remaining nodes are workers (ranks 1 to N-1)
   * All rollouts and training are distributed across nodes
   * Automatic coordination via sync barriers

   .. important::

      Multi-node mode **requires a shared filesystem** (e.g., NFS, Lustre, Azure Blob
      FUSE mount) accessible at the same ``--log-path`` on all nodes. The entire
      coordination protocol depends on this: phase-based sync barriers, trajectory
      aggregation, checkpoint sharing, and master IP discovery all operate through
      files written to ``--log-path``. Without a shared filesystem, multi-node mode
      will not work.

   Works with **any** ``--rl-phase`` option.

``--rank-weights <w1,w2,...>``
   **Required when** ``--num-nodes`` > 1.

   Comma-separated load weights for each node, controlling rollout task distribution proportionally.
   Must have exactly N values (one per node).

   Examples:

   * ``"1,1,1"`` - 3 nodes with equal load (each gets 33% of tasks)
   * ``"2,1,1"`` - Node 0 gets 50%, nodes 1-2 get 25% each
   * ``"1,2,1"`` - Node 1 gets 50%, nodes 0 and 2 get 25% each

``--master``
   Designate this node as the master node (rank 0).

   This is a convenience flag that automatically:

   * Sets the node rank to 0
   * Auto-detects the node's IP and writes it to the shared filesystem for workers to discover
   * Alternative to setting ``RANK=0`` environment variable

   **Note:** Only use on the master node. Worker nodes should use ``--worker`` instead.

``--worker <rank>``
   Designate this node as a worker with the specified rank (1, 2, 3, ...).

   This is a convenience flag that sets the node rank explicitly.
   Alternative to setting ``RANK=<rank>`` environment variable.

   Worker nodes automatically discover the master node's IP from the shared filesystem
   (written by the master at startup).

   Example: ``--worker 1`` for the first worker, ``--worker 2`` for the second worker, etc.

``--debug-mode``
   Debug mode flag. When present:

   * Does NOT start/stop/restart vLLM server
   * Assumes user has already launched vLLM externally
   * Useful for debugging rollout/training without vLLM lifecycle overhead

``--resume``
   Resume rollout from an existing checkpoint file.

   **Requires** ``--rollout-split`` to be specified (either ``train-only`` or ``eval-only``).

   **Limitation:** ``--resume`` is **only supported for single-node mode** (``--num-nodes 1``).
   Multi-node resume is not supported for two reasons:

   1. **Technical:** Checkpoint files are rank-specific (one per node), but the resume logic
      expects a unified checkpoint file
   2. **Efficiency:** It doesn't make sense to have only one machine resuming work while the
      other nodes wait idle. All nodes must work together synchronously in multi-node mode.

   When used:

   * Checks for an existing ``.checkpoint`` file for the specified split
   * If found, loads completed trajectories and resumes collection from where it left off
   * If no checkpoint exists, exits immediately with an error

   **Important:** If ``--resume`` is **NOT** specified, the script will start fresh from that
   iteration and **OVERWRITE** any existing checkpoint file. This means any partially collected
   trajectories will be lost.

   Example workflow for resuming an interrupted eval rollout:

   .. code-block:: bash

      # Check if checkpoint exists (optional, script will check for you)
      ls /data/exp1/test_trajectories/*.checkpoint

      # Resume from checkpoint
      bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 \
          --rl-phase rollout --rollout-split eval-only --resume

   The resume feature is particularly useful when:

   * A rollout was interrupted (crash, timeout, manual stop)
   * You want to continue collecting the remaining trajectories
   * You have partial progress saved in checkpoint files

Config Overrides
----------------

Any additional arguments containing ``=`` are passed directly to Hydra as config overrides.
This allows you to modify any configuration value from the command line without editing YAML files.

Syntax
^^^^^^

Hydra overrides follow this pattern:

.. code-block:: bash

   # Override a config value
   bash scripts/run.sh ... config.path=value

   # Add a new config value (use + prefix)
   bash scripts/run.sh ... +config.path=value

   # Remove a config value (use ~ prefix)
   bash scripts/run.sh ... ~config.path

Common Overrides
^^^^^^^^^^^^^^^^

**Dataset Configuration:**

.. code-block:: bash

   env_config.train_tasks=my_tasks.jsonl           # Training task file
   env_config.test_tasks=my_test_tasks.jsonl       # Test task file

**Rollout Size:**

.. code-block:: bash

   env_config.train_tasks_rollout_size=50          # Tasks per training rollout
   env_config.test_tasks_rollout_size=100          # Tasks per test rollout
   env_config.test_tasks_repeat_times=3            # Repeat test tasks N times

**Evaluation Judge:**

.. code-block:: bash

   openai_config.model=gpt-4o                      # Use GPT-4o for evaluation
   openai_config.model=gemini-3-flash-preview      # Use Gemini (default)

**Model Configuration:**

.. code-block:: bash

   model_config.model_type=qwen3-instruct          # Standard instruct model
   model_config.model_type=qwen3-think             # Thinking model variant

**Sampling Strategy:**

.. code-block:: bash

   env_config.train_tasks_sampler=uniform          # Uniform sampling (default)
   env_config.train_tasks_sampler=ratio            # Ratio-based sampling

**Other Useful Overrides:**

.. code-block:: bash

   env_config.server_size=50                       # Browser server pool size
   env_config.max_vllm_sessions=256                # Max concurrent vLLM requests
   policy_config.temperature=0.8                   # Sampling temperature
   policy_config.max_new_tokens=2048               # Max generation length

Override Examples
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Use a different dataset with larger rollout size
   bash scripts/run.sh \
       --data-path /data/shared \
       --log-path /data/exp1 \
       --rl-phase rollout \
       --rollout-split train-only \
       env_config.train_tasks=custom_tasks.jsonl \
       env_config.train_tasks_rollout_size=100

   # Use GPT-4o for evaluation with more test repeats
   bash scripts/run.sh \
       --data-path /data/shared \
       --log-path /data/exp1 \
       --rl-phase both \
       --eval-interval 6 \
       openai_config.model=gpt-4o \
       env_config.test_tasks_repeat_times=5

   # Multiple overrides in one command
   bash scripts/run.sh \
       --data-path /data/shared \
       --log-path /data/exp1 \
       --rl-phase rollout \
       --rollout-split train-only \
       env_config.train_tasks=my_tasks.jsonl \
       env_config.train_tasks_rollout_size=50 \
       env_config.server_size=30 \
       policy_config.temperature=0.9

Examples
--------

Single-node complete RL loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run train→eval→update loop, evaluate every 6 iterations
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase both --eval-interval 6

Single-node rollout only
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Collect only train trajectories
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase rollout --rollout-split train-only

   # Collect only eval trajectories
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase rollout --rollout-split eval-only

Single-node training only
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run training updates only (on existing trajectories)
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase update

Multi-node complete RL loop (3 nodes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Method 1: Using --master and --worker flags (Recommended)**

On the master node (rank 0):

.. code-block:: bash

   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase both --eval-interval 6 --num-nodes 3 --rank-weights "1,1,1" --master

On worker node 1:

.. code-block:: bash

   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase both --eval-interval 6 --num-nodes 3 --rank-weights "1,1,1" --worker 1

On worker node 2:

.. code-block:: bash

   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase both --eval-interval 6 --num-nodes 3 --rank-weights "1,1,1" --worker 2

**Method 2: Using RANK environment variable**

Run the same command on all nodes with different ``RANK`` values:

.. code-block:: bash

   # On node 0 (master)
   export RANK=0
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase both --eval-interval 6 --num-nodes 3 --rank-weights "1,1,1"

   # On node 1 (worker)
   export RANK=1
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase both --eval-interval 6 --num-nodes 3 --rank-weights "1,1,1"

   # On node 2 (worker)
   export RANK=2
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase both --eval-interval 6 --num-nodes 3 --rank-weights "1,1,1"

Multi-node rollout only
^^^^^^^^^^^^^^^^^^^^^^^

Distributed train-only rollout across 3 nodes:

.. code-block:: bash

   # On master node
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase rollout --rollout-split train-only --num-nodes 3 --rank-weights "1,1,1" --master

   # On worker nodes (master IP auto-discovered from shared filesystem)
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase rollout --rollout-split train-only --num-nodes 3 --rank-weights "1,1,1" --worker 1
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase rollout --rollout-split train-only --num-nodes 3 --rank-weights "1,1,1" --worker 2

Distributed eval-only rollout:

.. code-block:: bash

   # On master node
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase rollout --rollout-split eval-only --num-nodes 3 --rank-weights "1,1,1" --master

   # On worker nodes
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase rollout --rollout-split eval-only --num-nodes 3 --rank-weights "1,1,1" --worker 1
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase rollout --rollout-split eval-only --num-nodes 3 --rank-weights "1,1,1" --worker 2

Multi-node training only
^^^^^^^^^^^^^^^^^^^^^^^^

Distributed training across 3 nodes:

.. code-block:: bash

   # On master node
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase update --num-nodes 3 --rank-weights "1,1,1" --master

   # On worker nodes
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase update --num-nodes 3 --rank-weights "1,1,1" --worker 1
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase update --num-nodes 3 --rank-weights "1,1,1" --worker 2

Resume from checkpoint
^^^^^^^^^^^^^^^^^^^^^^

**Single-node mode only:**

.. code-block:: bash

   # Resume an interrupted eval-only rollout
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase rollout --rollout-split eval-only --resume

   # Resume an interrupted train-only rollout
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase rollout --rollout-split train-only --resume

**Important Notes:**

* ``--resume`` is **only supported for single-node mode**. Multi-node resume is not supported
  because it doesn't make sense to have one node resuming work while other nodes wait idle.
  All nodes must work together synchronously.
* In **multi-node mode**, the script always starts fresh for each iteration, regardless of whether
  checkpoint files exist. All nodes synchronize and begin the iteration from scratch simultaneously.
* If ``--resume`` is not specified in single-node mode, the script starts fresh and overwrites any
  existing checkpoint. Always use ``--resume`` when you want to continue from partial progress.

**Multi-node interruption recovery:**

If a multi-node rollout is interrupted, you have two options:

1. **Restart the entire iteration** - All nodes restart from the beginning of that iteration
2. **Manual aggregation** - Manually aggregate the available rank-specific checkpoint files (see fault tolerance section)

Debug mode
^^^^^^^^^^

.. code-block:: bash

   # Start vLLM manually first, then run rollout without vLLM management
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase rollout --rollout-split train-only --debug-mode

   # Debug multi-node training (vLLM already running on all nodes)
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase both --eval-interval 6 --num-nodes 3 --rank-weights "1,1,1" --debug-mode

Standalone vLLM server
^^^^^^^^^^^^^^^^^^^^^^

When using ``--debug-mode``, you need to start vLLM manually. Here's the command:

.. code-block:: bash

   # Start vLLM server (runs in foreground, Ctrl+C to stop)
   vllm serve /home/user/exp1/model.pt \
       --host 0.0.0.0 \
       --port 8999 \
       --max-num-seqs 512 \
       --gpu-memory-utilization 0.95 \
       --max-model-len 32768 \
       --tensor-parallel-size 1 \
       --data-parallel-size 8 \
       --limit-mm-per-prompt '{"video": 0}' \
       --allowed-local-media-path /home/user/exp1 \
       2>&1 | tee /home/user/exp1/vllm.log

If no checkpoint exists yet, use the HuggingFace model directly:

.. code-block:: bash

   # Start vLLM with HuggingFace model (first run, no checkpoint)
   vllm serve Qwen/Qwen3-VL-8B-Instruct \
       --host 0.0.0.0 \
       --port 8999 \
       --max-num-seqs 512 \
       --gpu-memory-utilization 0.95 \
       --max-model-len 32768 \
       --tensor-parallel-size 1 \
       --data-parallel-size 8 \
       --limit-mm-per-prompt '{"video": 0}' \
       --allowed-local-media-path /home/user/exp1 \
       2>&1 | tee /home/user/exp1/vllm.log

Argument Combinations
---------------------

Both ``--log-path`` and ``--data-path`` are always required.
Config overrides (``key=value``) are always optional and can be used with any phase.

.. list-table::
   :header-rows: 1
   :widths: 8 8 8 8 10 8 8 8 8 20

   * - --rl-phase
     - --eval-interval
     - --rollout-split
     - --num-nodes
     - --rank-weights
     - --master/--worker
     - --debug-mode
     - --resume
     - config overrides
   * - rollout
     - N/A
     - **Required**
     - Optional
     - Required if multi-node
     - Optional (for multi-node)
     - Optional
     - Optional (single-node only)
     - Optional (any key=value)
   * - update
     - N/A
     - N/A
     - Optional
     - Required if multi-node
     - Optional (for multi-node)
     - Optional
     - N/A
     - Optional (any key=value)
   * - both
     - **Required**
     - N/A
     - Optional
     - Required if multi-node
     - Optional (for multi-node)
     - Optional
     - N/A
     - Optional (any key=value)

**Notes:**

* Worker nodes automatically discover the master node's IP from the shared filesystem.
* ``--resume`` is only supported for single-node mode. In multi-node mode, all nodes always
  start fresh for each iteration, regardless of whether checkpoint files exist.

Node Rank Determination
-----------------------

When running multi-node jobs (``--num-nodes`` > 1), each node needs to know its rank (0 for master, 1+ for workers).
The script determines the rank using the following priority order:

1. **Command-line flags** (Recommended): ``--master`` or ``--worker <rank>``
2. **Environment variables**: ``RANK`` or ``NODE_RANK``

Examples:

.. code-block:: bash

   # Method 1: Using command-line flags (recommended for clarity)
   # Master node
   bash scripts/run.sh ... --num-nodes 2 --rank-weights "1,1" --master

   # Worker node
   bash scripts/run.sh ... --num-nodes 2 --rank-weights "1,1" --worker 1

   # Method 2: Using environment variables
   export RANK=0  # or RANK=1, RANK=2, etc.
   bash scripts/run.sh ... --num-nodes 2 --rank-weights "1,1"

**Master IP Discovery:**

The master node's IP is auto-detected via ``hostname -I`` and written to the shared
filesystem at startup. Worker nodes discover the master IP by reading this file.
No manual IP configuration is needed — this is handled entirely through the shared
filesystem.

.. tip::

   You can check the master node's IP with:

   .. code-block:: bash

      # Internal/cluster IP (used by default)
      hostname -I | awk '{print $1}'

      # Public IP (for reference/logging only)
      curl -s --max-time 2 https://ipinfo.io/ip

Multi-Node Coordination
-----------------------

When ``--num-nodes`` > 1, the script uses a phase-based coordination protocol:

For ``--rl-phase both``:

.. list-table::
   :header-rows: 1
   :widths: 10 45 45

   * - Phase
     - Master (rank 0)
     - Workers (rank 1+)
   * - 0
     - Sync all nodes at iteration start
     - Sync with all nodes
   * - 1-2
     - Train rollout, signal complete
     - Wait, then train rollout
   * - 3-4
     - Aggregate train trajectories
     - Signal files ready
   * - 5-6
     - Decide eval, run if needed, aggregate
     - Read decision, run if needed
   * - 7-9
     - Stop vLLM, prepare training data
     - Stop vLLM, wait for config
   * - 10
     - Multi-node training
     - Multi-node training
   * - 11-12
     - Copy checkpoint, restart vLLM
     - Wait, restart vLLM
   * - 13
     - Final sync before next iteration
     - Final sync

For ``--rl-phase rollout``:

* Only phases 0-6 are executed (data collection phases)
* No training phases

For ``--rl-phase update``:

* Only phases 7-13 are executed (training phases)
* No rollout phases

Multi-Node Fault Tolerance
---------------------------

Behavior During Node Failures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Important:** The multi-node system uses **strict barrier synchronization** without automatic
failure detection or recovery.

If any node fails or is interrupted:

* **All surviving nodes will hang indefinitely** at the next synchronization barrier
* No timeout mechanism exists - nodes wait forever for the failed node
* The iteration cannot proceed until all nodes are operational

**No automatic resume:** Multi-node mode does not support ``--resume``. When nodes restart after
an interruption, they always start fresh from the beginning of the iteration, regardless of
whether checkpoint files exist from a previous attempt.

This design is intentional: it doesn't make sense to have one node resuming from a checkpoint
while the other nodes wait idle. Multi-node coordination requires all nodes to work together
synchronously, so partial resume would waste resources and provide no benefit.

Checkpoint File State After Interruption
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When some nodes complete rollout while others fail:

Example with 4 nodes (master + 3 workers), node 2 crashes mid-rollout:

.. code-block:: text

   /logs/train_trajectories/
   ├── train_trajectories.pt.iteration5_rank_0.checkpoint  ✅ Complete (100 trajs)
   ├── train_trajectories.pt.iteration5_rank_1.checkpoint  ✅ Complete (100 trajs)
   ├── train_trajectories.pt.iteration5_rank_2.checkpoint  ⚠️  Partial/missing (37 trajs)
   └── train_trajectories.pt.iteration5_rank_3.checkpoint  ✅ Complete (100 trajs)

   # Aggregated file NOT created (aggregation never happened)
   # train_trajectories.pt.iteration5  ❌ DOES NOT EXIST

**Key behaviors:**

* Rank-specific checkpoint files are preserved on successful nodes
* Master never reaches aggregation phase (blocked at barrier)
* No unified trajectory file is created
* Partial data remains in individual rank checkpoint files

Recovery Options
^^^^^^^^^^^^^^^^

**Option 1: Restart Entire Iteration (Recommended)**

The safest approach - all nodes start the iteration from scratch:

.. code-block:: bash

   # 1. Kill all hung processes on all nodes
   pkill -9 -f "run.sh"
   pkill -9 -f "rollout.py"

   # 2. Clean up incomplete iteration files (on shared storage)
   rm /logs/train_trajectories/*iteration5_rank_*.checkpoint
   rm /logs/multinode_flags/*

   # 3. Restart all nodes with the same commands
   # This will start fresh from iteration 5

**Option 2: Manual Aggregation of Partial Data**

If significant work was completed, manually aggregate available files:

.. code-block:: bash

   cd /logs/train_trajectories

   # Python script to manually aggregate partial results
   python3 << 'EOF'
   import torch
   import glob
   import datetime

   iteration = 5
   traj_type = "train"

   # Find all available rank checkpoint files
   pattern = f"{traj_type}_trajectories.pt.iteration{iteration}_rank_*.checkpoint"
   rank_files = sorted(glob.glob(pattern))

   print(f"Found {len(rank_files)} rank files")
   all_trajectories = []
   metadata = {}

   for filepath in rank_files:
       try:
           data = torch.load(filepath, weights_only=False)
           trajs = data['trajectories']
           # Get metadata from first file
           if not metadata and 'metadata' in data:
               metadata = data['metadata'].copy()
           print(f"  {filepath}: {len(trajs)} trajectories")
           all_trajectories.extend(trajs)
       except Exception as e:
           print(f"  {filepath}: FAILED - {e}")

   # Save aggregated (even if incomplete) in dict format
   output = f"{traj_type}_trajectories.pt.iteration{iteration}"
   metadata['iteration'] = iteration
   metadata['timestamp'] = metadata.get('timestamp', datetime.datetime.now().isoformat())
   metadata['num_ranks_aggregated'] = len(rank_files)
   save_data = {
       'trajectories': all_trajectories,
       'metadata': metadata
   }
   torch.save(save_data, output)
   print(f"\n✅ Saved {len(all_trajectories)} trajectories to {output}")
   print(f"⚠️  WARNING: Incomplete data - missing some nodes")
   EOF

**Result:** You get partial data (e.g., 337/400 trajectories = 84%) and can proceed with training,
though results may be affected by incomplete sampling.

Best Practices for Production
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To minimize the impact of node failures:

1. **Monitor node health:**

   * Implement watchdog timers
   * Log heartbeats from each node
   * Alert on missing synchronization flags after timeout

2. **Use checkpoint backups:**

   .. code-block:: bash

      # After each iteration, backup rank-specific checkpoints
      if [ "${MULTINODE_RANK}" -eq 0 ]; then
          mkdir -p /logs/checkpoint_backups/iteration_${iteration}
          cp /logs/train_trajectories/*iteration${iteration}_rank_*.checkpoint \
             /logs/checkpoint_backups/iteration_${iteration}/
      fi

3. **Implement graceful degradation:**

   * Consider allowing partial aggregation in your fork
   * Log warnings when nodes are missing
   * Adjust training batch sizes for incomplete data

4. **Use reliable infrastructure:**

   * Shared filesystem with good reliability (NFS, Lustre, PVC)
   * Network with low latency and high availability
   * Avoid preemptible/spot instances for multi-node jobs

Configuration
-------------

Model Configuration
^^^^^^^^^^^^^^^^^^^

The script supports Qwen3 models:

* Model type options: ``qwen3-instruct`` (standard) or ``qwen3-think`` (reasoning)
* Default HuggingFace model: ``Qwen/Qwen3-VL-8B-Instruct`` (instruct) or ``Qwen/Qwen3-VL-8B-Thinking`` (thinking)

.. note::
   The shell scripts (``run.sh``, ``common_utils.sh``) use ``qwen-instruct`` / ``qwen-think`` (without "3")
   as the internal ``MODEL_TYPE`` variable. The Hydra config uses ``qwen3-instruct`` / ``qwen3-think``
   (with "3") for ``model_config.model_type``. These are separate variables that should match.

vLLM Configuration
^^^^^^^^^^^^^^^^^^

* Port: 8999
* Max sequences: 512
* GPU memory utilization: 0.95
* Max model length: 32768
* Tensor parallel size: 1
* Data parallel size: Number of GPUs

Key Helper Functions
--------------------

Located in ``scripts/shell_functions/``:

``vllm_utils.sh``
   * ``start_vllm`` - Starts vLLM server with retries (up to 3 attempts)
   * ``stop_vllm`` - Comprehensive vLLM shutdown
   * ``ensure_vllm_running`` - Start if not running, reuse if healthy

``gpu_utils.sh``
   * ``wait_for_gpu_memory_clear`` - Waits for GPU memory < 10GB
   * ``cleanup_deepspeed_processes`` - Kills training processes, clears shared memory

``training_utils.sh``
   * ``run_llamafactory_training`` - Runs training with 1-hour timeout
   * ``update_atomic`` - Prepares data, trains, copies checkpoint

``rollout_utils.sh``
   * ``rollout_atomic_train`` - Collects train trajectories
   * ``rollout_atomic_test`` - Collects eval trajectories

   Both functions support ``--resume`` mode via the ``RESUME_CHECKPOINT_PATH`` environment variable.

``common_utils.sh``
   * ``resolve_model_to_serve`` - Determines model path
   * ``detect_num_gpus`` - Counts available GPUs
   * ``find_last_checkpoint`` - Finds latest checkpoint

``multinode_sync.sh``
   * ``create_phase_flag`` / ``wait_for_phase`` - Barrier synchronization
   * ``aggregate_trajectories`` - Merges trajectories from all ranks

Environment Variables
---------------------

The script and its helper functions set:

* ``HF_HOME``: HuggingFace cache directory (set in ``run.sh``)
* ``DISABLE_VERSION_CHECK=1``: Disables LLaMA-Factory version check (set in ``run.sh``)
* ``VLLM_USE_TRITON_FLASH_ATTN=1``: Enables flash attention (set in ``vllm_utils.sh`` when starting vLLM)
* ``DEEPSPEED_LOG_LEVEL=WARNING``: Suppresses verbose DeepSpeed messages (set in ``training_utils.sh`` during training)
* ``WANDB_MODE=online``: Enables real-time WandB syncing (set in ``training_utils.sh`` when wandb env file exists)
* ``HYDRA_OVERRIDES``: Config overrides passed from CLI (used by rollout utilities)

Stopping the Script
-------------------

**Important:** Once the rollout phase has started, pressing Ctrl+C may not fully terminate
the program. The rollout spawns Python subprocesses for trajectory collection that don't
receive the interrupt signal.

To fully stop the script after rollout has started:

1. First, kill the Python subprocesses:

   .. code-block:: bash

      # Kill all Python processes (required to stop rollout subprocesses)
      pkill -9 python

      # Or more targeted: kill only rollout-related processes
      pkill -9 -f rollout.py

2. Then, press **Ctrl+C** in the terminal to exit the main bash script.

This two-step process is necessary because:

1. The rollout phase launches Python subprocesses via ``subprocess`` or ``multiprocessing``
2. These child processes run independently and don't inherit the parent's signal handlers
3. Ctrl+C only sends SIGINT to the foreground process group, which may not include all workers

**Note:** This behavior will be improved in a future release to support graceful shutdown with Ctrl+C.

Cleanup and Signal Handling
---------------------------

The script registers a cleanup handler on EXIT that:

1. Stops the vLLM server (unless ``--debug-mode``)
2. Removes the sync flag file
3. Removes LLaMA-Factory data directory
4. Cleans up multinode flags for the current rank

Troubleshooting
---------------

.. _vllm-server-errors:

vLLM Server Errors (HTTP 500)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom:**

.. code-block:: text

   vLLM request attempt 1 failed: vLLM server returned status 500:
   {"error":{"message":"","type":"Internal Server Error","param":null,"code":500}}

**Root Cause:**

The vLLM server is overloaded when ``server_size`` is too large, even if ``max_vllm_sessions`` is configured.

**Key Insight:** ``server_size`` is more reliable than ``max_vllm_sessions`` for controlling vLLM load.
The vLLM session limit may not work effectively when the browser server pool size is large,
causing the server to become blocked by excessive concurrent requests.

**Solution:**

Reduce ``server_size`` to a value the vLLM server can handle reliably:

.. code-block:: bash

   # If you're seeing 500 errors with server_size=112 and max_vllm_sessions=32:
   bash scripts/run.sh \
       --data-path /data/shared \
       --log-path /data/exp1 \
       --rl-phase rollout \
       env_config.server_size=64    # Reduce from 112 to 64

   # Or try a middle ground:
   env_config.server_size=96        # Reduce from 112 to 96

**Performance Impact:**

Reducing ``server_size`` typically improves or maintains rollout speed because:

* The system is no longer blocked waiting for overloaded vLLM responses
* Lower concurrency reduces queueing delays
* More reliable throughput with fewer retries and failures

**Recommended Values:**

* **For most setups:** ``server_size=64`` with ``max_vllm_sessions=32``
* **For high-end GPUs:** Start with ``server_size=96`` and monitor for errors
* **If still seeing errors:** Further reduce to ``server_size=48`` or ``32``

The optimal value depends on:

* GPU memory and compute capacity
* Model size (8B vs 72B parameters)
* vLLM data parallelism configuration
* Network latency between rollout and vLLM servers

**Monitoring:**

Watch for these signs of vLLM overload:

* HTTP 500 errors in rollout logs
* Increasing request retry attempts
* vLLM server logs showing queue saturation or OOM warnings
* Rollout throughput degradation over time
