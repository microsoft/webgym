Deployment
==========

This guide covers how to deploy the OmniBoxes browser automation infrastructure.

For installation prerequisites (Python, Redis, Playwright, etc.), see :doc:`../environment/environment_omnibox`.

Quick Start
-----------

Local Development
^^^^^^^^^^^^^^^^^

For local development and testing:

.. code-block:: bash

   cd omniboxes/deploy
   python deploy.py 10

Access at: ``http://localhost:7000``

Production (External API Server)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For machines that need to provide API access to other machines, expose the master port directly:

.. code-block:: bash

   cd omniboxes/deploy
   python deploy.py 10 --master-port 7000

Access at: ``http://your-server-ip:7000``

Deployment Options
------------------

Usage
^^^^^

.. code-block:: bash

   python deploy.py [instances] [options]

Arguments
"""""""""

- ``instances`` - Number of browser instances (required)

Options
"""""""

- ``--nginx`` - (Experimental, not yet supported) Setup nginx reverse proxy.
- ``--master-port PORT`` - Master server port (default: 7000)
- ``--node-port PORT`` - Node server port (default: 8080)
- ``--instance-start-port PORT`` - First instance port (default: 9000)
- ``--redis-port PORT`` - Redis port (default: 6379)
- ``--max-parallel N`` - Maximum parallel instance starts (default: auto)
- ``--disable-recovery`` - Disable automatic process recovery

Examples
""""""""

.. code-block:: bash

   # Local dev with 10 instances
   python deploy.py 10

   # Production with 20 instances on custom port
   python deploy.py 20 --master-port 7500

   # Large deployment with custom parallel workers
   python deploy.py 100 --max-parallel 20

Architecture
------------

.. code-block:: text

   ┌─────────────────────────────────────────┐
   │  Master Server (port 7000)              │
   │  Orchestrates browser sessions          │
   └─────────────────────────────────────────┘
                 ↓
   ┌─────────────────────────────────────────┐
   │  Node Server (port 8080)                │
   │  Manages browser instances               │
   └─────────────────────────────────────────┘
                 ↓
   ┌─────────────────────────────────────────┐
   │  Instance Servers (9000-9009)           │
   │  Individual Playwright browser instances │
   └─────────────────────────────────────────┘
                 ↓
   ┌─────────────────────────────────────────┐
   │  Redis (port 6379)                      │
   │  State coordination & caching            │
   └─────────────────────────────────────────┘

Starting Components Individually
---------------------------------

For more control, you can start each component separately:

**1. Start Redis (if not already running):**

.. code-block:: bash

   redis-server --port 6379

**2. Start instance servers:**

.. code-block:: bash

   # Start multiple instance servers on different ports
   for port in $(seq 9000 9049); do
       python -m omniboxes.node.instance_server --port $port &
   done

**3. Start the node server:**

.. code-block:: bash

   python -m omniboxes.node.server --port 8080 --workers 50

**4. Start the master server:**

.. code-block:: bash

   python -m omniboxes.master.server --port 7000 --nodes http://localhost:8080

Verifying the Server
--------------------

**Check server health:**

.. code-block:: bash

   curl http://localhost:7000/info

**Expected response:**

.. code-block:: json

   {
     "nodes": [
       {
         "url": "http://localhost:8080",
         "hash": "abc123",
         "healthy": true,
         "capacity": 50,
         "available": 50,
         "instances": []
       }
     ]
   }

**Test browser allocation:**

.. code-block:: bash

   # Allocate a browser instance
   curl -X POST http://localhost:7000/get

   # Expected response:
   # {"instance_id": "uuid:9000", "node": "abc123"}

API Usage
---------

All requests go to the master server directly:

.. code-block:: bash

   # Get server info
   curl -H "x-api-key: default_key" http://YOUR_IP:7000/info

   # Get a new browser instance
   curl -X POST -H "x-api-key: default_key" "http://YOUR_IP:7000/get?lifetime_mins=60"

   # Get a screenshot
   curl -H "x-api-key: default_key" \
     "http://YOUR_IP:7000/screenshot?instance_id=UUID:9000&node=NODE_HASH" \
     > screenshot.png

   # Execute a command
   curl -X POST -H "x-api-key: default_key" \
     -H "Content-Type: application/json" \
     -d '{"instance_id":"UUID:9000","node":"NODE_HASH","visit_page":{"url":"https://example.com"}}' \
     http://YOUR_IP:7000/execute

   # Reset an instance
   curl -X POST -H "x-api-key: default_key" \
     "http://YOUR_IP:7000/reset?instance_id=UUID:9000&node=NODE_HASH"

Configuration Options
---------------------

**API key authentication:**

The API key is hardcoded as ``default_key`` in the server code. All requests must include the API key header:

.. code-block:: bash

   # Requests must include the API key header
   curl -H "x-api-key: default_key" http://localhost:7000/info

Firewall Configuration
----------------------

GCP Firewall Rules
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   gcloud compute firewall-rules create allow-omniboxes \
       --allow tcp:7000 \
       --source-ranges 0.0.0.0/0 \
       --description "Allow OmniBoxes master server"

Security Notes
--------------

- The API requires an API key (``x-api-key`` header) for all requests
- Default API key is ``default_key`` - **change this in production**
- Consider using environment variables for secrets management

Troubleshooting
---------------

Port Conflicts
^^^^^^^^^^^^^^

If you get "address already in use" errors:

.. code-block:: bash

   # Check what's using the port
   sudo lsof -i :7000

   # Kill the process
   sudo kill -9 <PID>

Redis Issues
^^^^^^^^^^^^

.. code-block:: bash

   # Check if Redis is running
   redis-cli -p 6379 ping

   # Kill Redis if needed
   redis-cli -p 6379 shutdown

Production Deployment Checklist
--------------------------------

For production, consider:

1. **Change the default API key** by setting an environment variable or using a secrets manager

2. **Set up monitoring** for the OmniBox processes

3. **Configure log rotation** for application logs

4. **Run as a systemd service** for automatic restart on boot

Deployment Files
----------------

- **deploy.py** - Unified deployment script (all-in-one)
- **process_manager.py** - Core launcher class (imported by deploy.py)
- **redis.conf** - Redis server configuration

Next Steps
----------

- See :doc:`rollout_server` for detailed architecture and API documentation
- See :doc:`../rl_pipeline/rl_rollout` for integration with WebGym environments
