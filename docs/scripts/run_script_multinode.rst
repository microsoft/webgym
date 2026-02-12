Multi-Node Mode: run.sh
=================

Distributes rollout and training across nodes using a shared filesystem for coordination.

.. important::

   Multi-node mode **requires a shared filesystem** (NFS, Lustre, Azure Blob FUSE) accessible
   at the same ``--log-path`` on all nodes.

Setup
-----

``--num-nodes <N>``
   Number of nodes (default: 1). Node 0 is master, 1+ are workers.

``--rank-weights <w1,w2,...>``
   Required for multi-node. Comma-separated load weights (one per node).
   E.g., ``"2,1,1"`` gives node 0 50% of tasks, nodes 1-2 get 25% each.

``--master``
   Designate this node as master (rank 0). Auto-detects IP and writes to shared filesystem.

``--worker <rank>``
   Designate this node as a worker with the given rank. Discovers master IP from shared filesystem.

Ranks can also be set via ``RANK`` or ``NODE_RANK`` environment variables.

Examples
--------

.. code-block:: bash

   # Complete RL loop (3 nodes)
   # Master:
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase both --eval-interval 6 --num-nodes 3 --rank-weights "1,1,1" --master
   # Workers:
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase both --eval-interval 6 --num-nodes 3 --rank-weights "1,1,1" --worker 1
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase both --eval-interval 6 --num-nodes 3 --rank-weights "1,1,1" --worker 2

   # Rollout only (3 nodes)
   # Master:
   bash scripts/run.sh --data-path /data/shared --log-path /data/exp1 --rl-phase rollout --rollout-split train-only --num-nodes 3 --rank-weights "1,1,1" --master
   # Workers:
   bash scripts/run.sh ... --worker 1
   bash scripts/run.sh ... --worker 2

Coordination Protocol
---------------------

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

``--rl-phase rollout`` uses phases 0-6 only. ``--rl-phase update`` uses phases 7-13 only.

Fault Tolerance
---------------

The multi-node system uses **strict barrier synchronization** without automatic failure detection.
If any node fails, **all surviving nodes hang indefinitely** at the next barrier.

**Recovery:** Kill all processes on all nodes, clean up incomplete files, and restart:

.. code-block:: bash

   # On all nodes:
   pkill -9 -f "run.sh|rollout.py"

   # On shared storage:
   rm /logs/train_trajectories/*iteration${N}_rank_*.checkpoint
   rm /logs/multinode_flags/*

   # Restart all nodes with the same commands

For partial data recovery, manually aggregate available rank checkpoint files using ``torch.load``
and concatenate trajectories.
