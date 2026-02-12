Deployment
==========

This guide covers how to deploy the OmniBoxes browser automation infrastructure.

For installation prerequisites (Python, Redis, Playwright, etc.), see :doc:`../environment/environment_omnibox`.

Quick Start
-----------

Local Development
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd omniboxes/deploy
   python deploy.py 10

Access at: ``http://localhost:7000``

Production (External API Server)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd omniboxes/deploy
   python deploy.py 10 --master-port 7000

Access at: ``http://your-server-ip:7000``

Deployment Options
------------------

.. code-block:: bash

   python deploy.py [instances] [options]

**Arguments:**

- ``instances`` - Number of browser instances (required)

**Options:**

- ``--nginx`` - Setup nginx reverse proxy (requires ``sudo`` and ``nginx``)
- ``--master-port PORT`` - Master server port (default: 7000)
- ``--node-port PORT`` - Node server port (default: 8080)
- ``--instance-start-port PORT`` - First instance port (default: 9000)
- ``--redis-port PORT`` - Redis port (default: 6379)
- ``--max-parallel N`` - Maximum parallel instance starts (default: auto)
- ``--disable-recovery`` - Disable automatic process recovery

Architecture
------------

.. code-block:: text

   Master Server (port 7000)  →  Node Server (port 8080)  →  Instance Servers (9000+)
   Orchestrates sessions          Manages instances            Individual Playwright browsers
                                                                        ↓
                                                               Redis (port 6379)
                                                               State coordination

Starting Components Individually
---------------------------------

.. code-block:: bash

   # 1. Redis
   redis-server --port 6379

   # 2. Instance servers
   for port in $(seq 9000 9049); do
       python -m omniboxes.node.instance_server --port $port &
   done

   # 3. Node server
   python -m omniboxes.node.server --port 8080 --workers 50

   # 4. Master server
   python -m omniboxes.master.server --port 7000 --nodes http://localhost:8080

Verifying the Server
--------------------

.. code-block:: bash

   # Health check
   curl -H "x-api-key: default_key" http://localhost:7000/info

   # Allocate a browser instance
   curl -X POST -H "x-api-key: default_key" http://localhost:7000/get

API Usage
---------

All requests require the ``x-api-key: default_key`` header and go to the master server.

.. code-block:: bash

   # Get server info
   curl -H "x-api-key: default_key" http://YOUR_IP:7000/info

   # Get a new browser instance
   curl -X POST -H "x-api-key: default_key" "http://YOUR_IP:7000/get?lifetime_mins=60"

   # Screenshot
   curl -H "x-api-key: default_key" \
     "http://YOUR_IP:7000/screenshot?instance_id=UUID:9000&node=NODE_HASH" > screenshot.png

   # Execute a command
   curl -X POST -H "x-api-key: default_key" \
     -H "Content-Type: application/json" \
     -d '{"instance_id":"UUID:9000","node":"NODE_HASH","visit_page":{"url":"https://example.com"}}' \
     http://YOUR_IP:7000/execute

   # Reset an instance
   curl -X POST -H "x-api-key: default_key" \
     "http://YOUR_IP:7000/reset?instance_id=UUID:9000&node=NODE_HASH"

Security
--------

- Default API key is ``default_key`` — **change this in production**
- Consider using environment variables for secrets management

Troubleshooting
---------------

.. code-block:: bash

   # Port conflicts
   sudo lsof -i :7000 && sudo kill -9 <PID>

   # Redis issues
   redis-cli -p 6379 ping
   redis-cli -p 6379 shutdown

Deployment Files
----------------

- **deploy.py** - Single-node deployment
- **deploy_multinode.py** - Multi-node deployment
- **process_manager.py** - Core launcher class
- **multinode_manager.py** - Multi-node launcher class
- **nginx_manager.py** - Nginx reverse proxy configuration

Next Steps
----------

- See :doc:`rollout_server` for detailed architecture and API documentation
- See :doc:`../rl_pipeline/rl_rollout` for integration with WebGym environments
