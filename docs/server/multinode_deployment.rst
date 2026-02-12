Multi-node Deployment
=====================

Deploy OmniBoxes across **multiple machines** using Redis-based automatic node discovery.

Architecture Overview
---------------------

.. code-block:: text

   MASTER NODE (<MASTER_IP>)
   ├── Redis Server (DB 0: local pools, DB 1: service registry)
   ├── Master Server (port 7000, API + load balancer)
   └── Worker Program (Node :8080, Instances :9000+)
           │
           │ Discovers workers via Redis DB 1
           │ Health-checks via /info every 10s
           │
   ┌───────┴───────┐
   WORKER NODE 1    WORKER NODE N
   ├── Local Redis   ├── Local Redis
   └── Worker Program └── Worker Program
       (Node :8080)      (Node :8080)
       Registers with master Redis DB 1 (heartbeat every 30s, TTL 120s)

**Key points:**

- Master node runs master program AND one worker program
- Each worker node runs one worker program
- Each machine runs its own **local Redis DB 0** for instance pools
- Only the master's **Redis DB 1** is shared for node discovery

Prerequisites
-------------

Complete :doc:`../environment/environment_omnibox` on **every machine**.

On the **master node**, configure Redis for network access:

.. code-block:: bash

   sudo sed -i 's/^bind 127.0.0.1.*/bind 0.0.0.0/' /etc/redis/redis.conf
   sudo sed -i 's/^protected-mode yes/protected-mode no/' /etc/redis/redis.conf
   sudo systemctl restart redis

.. warning::
   This makes Redis accessible from any IP. Use firewall rules to restrict access in production.

**Required ports:** 6379 (Redis on master), 7000 (Master API), 8080 (Node server on each worker)

Quick Start
-----------

Example: 2 machines with 112 browser instances each (224 total).

**Step 1: Start Master Node** (on ``<MASTER_IP>``):

.. code-block:: bash

   conda activate webgym && cd omniboxes/deploy && mkdir -p redis-data
   python deploy_multinode.py 112 --mode both --master-redis-host <MASTER_IP>

**Step 2: Start Worker Node** (on ``<WORKER_IP>``):

.. code-block:: bash

   conda activate webgym && cd omniboxes/deploy && mkdir -p redis-data
   python deploy_multinode.py 112 --mode worker \
       --master-redis-host <MASTER_IP> --advertise-ip <WORKER_IP>

.. note::
   ``--advertise-ip`` is required when auto-detected IP differs from what the master can reach (common in cloud environments).

**Step 3: Verify:**

.. code-block:: bash

   curl http://<MASTER_IP>:7000/info -H "x-api-key: default_key"

Deployment Modes
----------------

**both** (recommended for master): Runs master program + one worker program.

.. code-block:: bash

   python deploy_multinode.py <N> --mode both --master-redis-host <MASTER_IP>

**worker**: Runs one worker that registers with master.

.. code-block:: bash

   python deploy_multinode.py <N> --mode worker --master-redis-host <MASTER_IP> --advertise-ip <WORKER_IP>

**master**: Coordinator only, no browser instances.

.. code-block:: bash

   python deploy_multinode.py --mode master --master-redis-host <MASTER_IP>

Configuration Options
---------------------

.. code-block:: text

   deploy_multinode.py [NUM_INSTANCES] [OPTIONS]

   Mode:       --mode {master,worker,both}    (default: both)
   Discovery:  --master-redis-host IP         (default: localhost)
               --advertise-ip IP              (when auto-detect fails)
   Ports:      --master-redis-port PORT       (default: 6379)
               --master-port PORT             (default: 7000)
               --node-port PORT               (default: 8080)
               --instance-start-port PORT     (default: 9000)
               --redis-port PORT              (default: 6379)
   Tuning:     --max-parallel N               (default: auto)
               --disable-recovery

Service Discovery Flow
----------------------

1. **Master starts**: Launches Redis, instance servers, node server, master API. Adds ``localhost:8080`` as static node.
2. **Workers start**: Launch local Redis, instance servers, node server. Register with master's Redis DB 1 (``omnibox:nodes:{hostname} = http://{ip}:8080``, TTL 120s). Heartbeat every 30s.
3. **Master discovers**: Polls Redis DB 1 every 10s, merges with static nodes, removes expired entries.
4. **Worker failure**: Heartbeats stop → key expires after 120s → master removes worker on next poll.

Stopping & Restarting
---------------------

.. code-block:: bash

   # Stop all processes on a machine
   pkill -9 -f "deploy_multinode|instance_server|omniboxes\."

   # Clean restart (master)
   redis-cli -p 6379 -n 1 FLUSHDB
   redis-cli -p 6379 -n 0 DEL available in_use
   python deploy_multinode.py 112 --mode both --master-redis-host <MASTER_IP>

   # Clean restart (worker)
   redis-cli -p 6379 -n 0 DEL available in_use
   python deploy_multinode.py 112 --mode worker --master-redis-host <MASTER_IP> --advertise-ip <WORKER_IP>

After master restart, workers re-register automatically within 30s.

Troubleshooting
---------------

**Worker unhealthy** (``"healthy": false``): Master can't reach worker's registered IP on port 8080. Use ``--advertise-ip`` with correct public IP.

**Worker not appearing**: Check ``redis-cli -h <MASTER_IP> -p 6379 ping`` and ``redis-cli -h <MASTER_IP> -p 6379 -n 1 KEYS "omnibox:nodes:*"``. Verify master's Redis has ``bind 0.0.0.0`` and ``protected-mode no``.

**Stale nodes after restart**: Flush registry: ``redis-cli -p 6379 -n 1 FLUSHDB``

Scaling Examples
----------------

.. code-block:: bash

   # 2 Machines (224 instances)
   python deploy_multinode.py 112 --mode both --master-redis-host <MASTER_IP>
   python deploy_multinode.py 112 --mode worker --master-redis-host <MASTER_IP> --advertise-ip <WORKER_IP>

   # 5 Machines (500 instances) on same private network (no --advertise-ip needed)
   python deploy_multinode.py 100 --mode both --master-redis-host 10.0.0.1
   python deploy_multinode.py 100 --mode worker --master-redis-host 10.0.0.1  # on machines 2-5

Next Steps
----------

- See :doc:`quickstart_server` for single-machine deployment
- See :doc:`rollout_server` for OmniBoxes API reference
- See :doc:`../rl_pipeline/rl_rollout` for integration with WebGym environments
