Multi-node Deployment
=====================

This guide explains how to deploy OmniBoxes across **multiple machines** using Redis-based automatic node discovery.

Architecture Overview
---------------------

The multi-node system uses a master-worker architecture:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────┐
   │  MASTER NODE (e.g. <MASTER_IP>)                                  │
   │                                                                     │
   │  ┌──────────────┐  ┌────────────────┐  ┌────────────────────────┐  │
   │  │ Redis Server │  │ Master Server  │  │ Worker Program         │  │
   │  │ Port 6379    │  │ Port 7000      │  │ Node :8080             │  │
   │  │ DB 0: local  │  │ (API + LB)     │  │ Instances :9000-9111   │  │
   │  │ DB 1: registry│  │               │  │ Local Redis DB 0       │  │
   │  └──────────────┘  └───────┬────────┘  └────────────────────────┘  │
   │                            │                                        │
   └────────────────────────────┼────────────────────────────────────────┘
                                │ Discovers workers via Redis DB 1
                                │ Health-checks via /info every 10s
                                │
                  ┌─────────────┴─────────────┐
                  │                           │
                  ▼                           ▼
   ┌──────────────────────────┐  ┌──────────────────────────┐
   │  WORKER NODE 1           │  │  WORKER NODE N           │
   │  (e.g. <WORKER_IP>)   │  │  (e.g. 10.0.0.X)        │
   │                          │  │                          │
   │  ┌────────────────────┐  │  │  ┌────────────────────┐  │
   │  │ Worker Program     │  │  │  │ Worker Program     │  │
   │  │ Node :8080         │  │  │  │ Node :8080         │  │
   │  │ Instances :9000+   │  │  │  │ Instances :9000+   │  │
   │  │ Local Redis DB 0   │  │  │  │ Local Redis DB 0   │  │
   │  └────────────────────┘  │  │  └────────────────────┘  │
   │                          │  │                          │
   │  Registers with master   │  │  Registers with master   │
   │  Redis DB 1 (heartbeat   │  │  Redis DB 1 (heartbeat   │
   │  every 30s, TTL 120s)    │  │  every 30s, TTL 120s)    │
   └──────────────────────────┘  └──────────────────────────┘

**Key points:**

- The **master node** runs the master program (API + load balancer) AND one worker program.
- Each **worker node** runs one worker program.
- The master program discovers and health-checks all worker programs (local + remote).
- Every machine (master and workers) runs its own **local Redis DB 0** for instance pool management.
- Only the master's **Redis DB 1** is shared across machines for node registration/discovery.

Prerequisites
-------------

Complete the :doc:`../environment/environment_omnibox` setup on **every machine** (master and workers).

Additionally, on the **master node**, configure Redis for network access:

   Edit ``/etc/redis/redis.conf``:

   .. code-block:: bash

      # Change bind from 127.0.0.1 to 0.0.0.0
      sudo sed -i 's/^bind 127.0.0.1.*/bind 0.0.0.0/' /etc/redis/redis.conf

      # Disable protected mode
      sudo sed -i 's/^protected-mode yes/protected-mode no/' /etc/redis/redis.conf

      # Restart Redis
      sudo systemctl restart redis

   .. warning::
      This makes Redis accessible from any IP. In production, use firewall rules
      to restrict access to your worker IPs only.

**Verify ports are open** between machines:

   - **6379** (Redis on master, workers must reach this)
   - **7000** (Master API, clients must reach this)
   - **8080** (Node server on each worker, master must reach this)

Quick Start
-----------

Example: 2 machines with 112 browser instances each (224 total).

- **Master**: ``<MASTER_IP>``
- **Worker**: ``<WORKER_IP>``

Step 1: Start the Master Node
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On the master machine (``<MASTER_IP>``):

.. code-block:: bash

   conda activate webgym
   cd omniboxes/deploy
   mkdir -p redis-data
   python deploy_multinode.py 112 \
       --mode both \
       --master-redis-host <MASTER_IP>

This starts:

- Redis server on port 6379 (service registry, DB 1)
- 112 browser instance servers (ports 9000-9111)
- 1 node server on port 8080 (manages local instances)
- Master API server on port 7000 (coordinates all workers)

Step 2: Start Worker Nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On each worker machine (e.g. ``<WORKER_IP>``):

.. code-block:: bash

   conda activate webgym
   cd omniboxes/deploy
   mkdir -p redis-data
   python deploy_multinode.py 112 \
       --mode worker \
       --master-redis-host <MASTER_IP> \
       --advertise-ip <WORKER_IP>

This starts:

- Local Redis on port 6379 (instance pool management, DB 0)
- 112 browser instance servers (ports 9000-9111)
- 1 node server on port 8080
- Registers with master's Redis (DB 1) using the advertise IP
- Sends heartbeat every 30 seconds

.. note::
   ``--advertise-ip`` is required when the worker's auto-detected IP (internal/private)
   differs from the IP the master can reach. This is common in cloud environments
   where machines have both private and public IPs.

Step 3: Verify Deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^

From any machine:

.. code-block:: bash

   curl http://<MASTER_IP>:7000/info -H "x-api-key: default_key"

Expected output (2 healthy nodes, 224 total capacity):

.. code-block:: json

   {
     "nodes": [
       {
         "url": "http://localhost:8080",
         "hash": "abc123",
         "healthy": true,
         "capacity": 112,
         "available": 112,
         "instances": []
       },
       {
         "url": "http://<WORKER_IP>:8080",
         "hash": "def456",
         "healthy": true,
         "capacity": 112,
         "available": 112,
         "instances": []
       }
     ]
   }

Deployment Modes
----------------

Mode: both (Recommended for Master)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Runs the master program **and** one worker program on the same machine.

.. code-block:: bash

   python deploy_multinode.py <NUM_INSTANCES> \
       --mode both \
       --master-redis-host <MASTER_PUBLIC_IP>

**Starts**: Redis, master API, instance servers, node server.

**When to use**: Almost always for the master node. This lets the master machine
also contribute browser instances to the pool.

Mode: worker
^^^^^^^^^^^^

Runs one worker program that registers with the master.

.. code-block:: bash

   python deploy_multinode.py <NUM_INSTANCES> \
       --mode worker \
       --master-redis-host <MASTER_PUBLIC_IP> \
       --advertise-ip <THIS_MACHINES_PUBLIC_IP>

**Starts**: Local Redis (DB 0), instance servers, node server.

**When to use**: For all non-master machines in the cluster.

Mode: master
^^^^^^^^^^^^

Runs only the master program (coordinator) with no browser instances.

.. code-block:: bash

   python deploy_multinode.py --mode master --master-redis-host <MASTER_PUBLIC_IP>

**Starts**: Redis, master API.

**When to use**: When the master machine is dedicated to coordination only
(e.g., a small VM that just load-balances across many large workers).

Configuration Options
---------------------

.. code-block:: text

   deploy_multinode.py [NUM_INSTANCES] [OPTIONS]

   Positional:
     NUM_INSTANCES              Number of browser instances (required for worker/both, default: 0)

   Mode and Discovery:
     --mode {master,worker,both}  Deployment mode (default: both)
     --master-redis-host IP       Master node's IP address (default: localhost)

   Networking:
     --advertise-ip IP          Public IP for worker registration (when auto-detect fails)
     --master-redis-port PORT   Master's Redis port (default: 6379)
     --master-port PORT         Master API port (default: 7000)
     --node-port PORT           Node server port (default: 8080)
     --instance-start-port PORT Starting port for instances (default: 9000)
     --redis-port PORT          Local Redis server port (default: 6379)

   Tuning:
     --max-parallel N           Max parallel instance startups (default: auto)
     --disable-recovery         Disable automatic process recovery

How It Works
------------

Redis Architecture
^^^^^^^^^^^^^^^^^^

Each machine runs its own Redis. Two databases are used:

- **DB 0 (local)**: Instance pool management. Each machine tracks its own ``available``
  and ``in_use`` instance sets locally. This is never shared across machines.
- **DB 1 (shared, master only)**: Service registry. Workers connect to the **master's**
  Redis DB 1 to register themselves. The master reads DB 1 to discover workers.

.. code-block:: text

   # Master's Redis DB 1 (registry - shared)
   omnibox:nodes:worker-hostname-1 = "http://<WORKER_IP>:8080"  [TTL: 120s]
   omnibox:nodes:worker-hostname-2 = "http://<WORKER_2_IP>:8080"      [TTL: 120s]

   # Each machine's Redis DB 0 (local - not shared)
   available = {9000, 9001, 9002, ...}    # ports of idle instances
   in_use = {instance_id_1, ...}          # ports of active instances

Service Discovery Flow
^^^^^^^^^^^^^^^^^^^^^^

1. **Master starts** with ``--mode both``:

   - Launches local Redis
   - Starts instance servers and node server (local worker)
   - Starts master API server with Redis discovery enabled
   - Master adds ``localhost:8080`` as a static node (always present)

2. **Workers start** with ``--mode worker``:

   - Launch their own local Redis (for DB 0 instance pool)
   - Start instance servers and node server
   - Connect to master's Redis (DB 1)
   - Register: ``omnibox:nodes:{hostname} = http://{advertise_ip}:8080`` (TTL: 120s)
   - Start heartbeat thread (refreshes registration every 30s)

3. **Master discovers workers**:

   - Polls Redis DB 1 every 10 seconds
   - Merges discovered nodes with static nodes (localhost)
   - Removes expired nodes (TTL expired = worker dead)
   - Health-checks all nodes via ``GET /info`` every 10 seconds

4. **Worker failure**:

   - Worker stops sending heartbeats
   - Redis key expires after 120s
   - Master removes worker from active node list on next poll
   - Worker remains in ``/info`` as ``"healthy": false`` (so clients with existing
     references can still route to it and get a clear error)

Stopping & Restarting
---------------------

Stopping All Processes
^^^^^^^^^^^^^^^^^^^^^^^

To stop all OmniBoxes processes on a machine:

.. code-block:: bash

   # Kill all OmniBoxes processes
   pkill -9 -f "deploy_multinode|instance_server|omniboxes\."

Full Restart (Master)
^^^^^^^^^^^^^^^^^^^^^^

To cleanly restart the master node:

.. code-block:: bash

   # 1. Kill all processes
   pkill -9 -f "deploy_multinode|instance_server|omniboxes\."

   # 2. Clean Redis state
   redis-cli -p 6379 -n 1 FLUSHDB          # Clear service registry
   redis-cli -p 6379 -n 0 DEL available in_use   # Clear instance pool

   # 3. Relaunch
   conda activate webgym
   cd omniboxes/deploy
   mkdir -p redis-data
   python deploy_multinode.py 112 --mode both --master-redis-host <MASTER_IP>

.. note::
   After restarting the master, worker nodes will automatically re-register
   via their heartbeat (within 30 seconds). If workers were also restarted,
   they need to be relaunched manually.

Full Restart (Worker)
^^^^^^^^^^^^^^^^^^^^^^

To cleanly restart a worker node:

.. code-block:: bash

   # 1. Kill all processes
   pkill -9 -f "deploy_multinode|instance_server|omniboxes\."

   # 2. Clean local Redis
   redis-cli -p 6379 -n 0 DEL available in_use

   # 3. Relaunch
   conda activate webgym
   cd omniboxes/deploy
   mkdir -p redis-data
   python deploy_multinode.py 112 --mode worker \
       --master-redis-host <MASTER_IP> \
       --advertise-ip <WORKER_PUBLIC_IP>

Troubleshooting
---------------

Worker Shows as Unhealthy
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom**: Worker appears in ``/info`` but ``"healthy": false``

**Cause**: Master cannot reach the worker's registered IP on port 8080.

**Solution**:

.. code-block:: bash

   # From the master machine, test connectivity to the worker:
   curl http://<WORKER_IP>:8080/info -H "x-api-key: default_key"

   # If this fails, the worker registered with a wrong IP.
   # Use --advertise-ip to specify the correct public IP:
   python deploy_multinode.py 112 --mode worker \
       --master-redis-host <MASTER_IP> \
       --advertise-ip <WORKER_PUBLIC_IP>

Worker Not Appearing at All
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom**: ``/info`` shows no worker node.

**Solution**:

.. code-block:: bash

   # 1. Check if worker can reach master's Redis:
   redis-cli -h <MASTER_IP> -p 6379 ping
   # Should return: PONG

   # 2. Check if worker registered in Redis:
   redis-cli -h <MASTER_IP> -p 6379 -n 1 KEYS "omnibox:nodes:*"

   # 3. If Redis is unreachable, check master's Redis config:
   #    /etc/redis/redis.conf must have: bind 0.0.0.0
   #    and: protected-mode no

Worker Shows with Wrong IP
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom**: Worker appears as ``http://10.x.x.x:8080`` (private IP) instead of public IP.

**Cause**: Auto-detected IP is the machine's internal/private network interface.

**Solution**: Use ``--advertise-ip`` to specify the public IP that the master can reach.

Stale Nodes After Restart
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom**: Old/dead nodes still appear in ``/info`` briefly after a restart.

**Cause**: Redis registration has a 120s TTL. Old entries persist until they expire.

**Solution**: Flush the registry before restarting:

.. code-block:: bash

   # On master machine:
   redis-cli -p 6379 -n 1 FLUSHDB
   redis-cli -p 6379 -n 0 DEL available in_use

Scaling Examples
----------------

2 Machines (224 instances)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Machine 1 (master + worker): <MASTER_IP>
   python deploy_multinode.py 112 --mode both --master-redis-host <MASTER_IP>

   # Machine 2 (worker): <WORKER_IP>
   python deploy_multinode.py 112 --mode worker \
       --master-redis-host <MASTER_IP> --advertise-ip <WORKER_IP>

5 Machines (500 instances)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Machine 1 (master + worker): 10.0.0.1
   python deploy_multinode.py 100 --mode both --master-redis-host 10.0.0.1

   # Machines 2-5 (workers):
   python deploy_multinode.py 100 --mode worker --master-redis-host 10.0.0.1

.. note::
   When all machines are on the same private network (e.g. ``10.0.0.x``),
   ``--advertise-ip`` is not needed since the auto-detected IP is already reachable.

Next Steps
----------

- See :doc:`../environment/environment_omnibox` for installation details
- See :doc:`quickstart_server` for single-machine deployment
- See :doc:`rollout_server` for OmniBoxes API reference
- See :doc:`../rl_pipeline/rl_rollout` for integration with WebGym environments
