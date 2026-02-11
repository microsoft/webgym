Code Details
============

The rollout server (``omniboxes/``) provides a distributed browser automation infrastructure for collecting web agent trajectories at scale.

Architecture Overview
---------------------

OmniBoxes uses a three-tier architecture:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                        Master Server                            │
   │                     (Port 7000 by default)                      │
   │  • Routes requests to nodes                                     │
   │  • Load balancing across nodes                                  │
   │  • API gateway for all browser operations                       │
   └───────────────────────────┬─────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
   │ Node Server │      │ Node Server │      │ Node Server │
   │  (Port 8080)│      │  (Port 8080)│      │  (Port 8080)│
   │  Machine 1  │      │  Machine 2  │      │  Machine N  │
   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
          │                    │                    │
    ┌─────┴─────┐        ┌─────┴─────┐        ┌─────┴─────┐
    │ Instance  │        │ Instance  │        │ Instance  │
    │ Servers   │        │ Servers   │        │ Servers   │
    │ (9000+)   │        │ (9000+)   │        │ (9000+)   │
    └───────────┘        └───────────┘        └───────────┘

Module Structure
----------------

.. code-block:: text

   omniboxes/
   ├── __init__.py
   ├── master/                    # Master server (load balancer)
   │   ├── server.py             # FastAPI master server
   │   ├── node_manager.py       # Node registration and health
   │   └── logging_utils.py
   ├── node/                      # Node server (instance pool manager)
   │   ├── server.py             # FastAPI node server
   │   ├── instance_server.py    # Single instance server
   │   ├── logging_utils.py      # Node-level logger configuration
   │   ├── instances/            # Browser instance implementations
   │   │   ├── base.py           # Base instance class
   │   │   ├── playwright_instance.py  # Playwright browser
   │   │   ├── _playwright_controller.py  # Browser controller
   │   │   ├── _set_of_marks.py  # Visual annotation system
   │   │   ├── _types.py         # Type definitions (FunctionCall, etc.)
   │   │   └── _page_script.js   # JS page script for UI element interaction
   │   └── utils.py
   ├── deploy/                    # Deployment utilities
   │   ├── deploy.py             # CLI entry point
   │   ├── process_manager.py    # Server lifecycle management
   │   └── nginx_manager.py      # Nginx reverse proxy (experimental, not yet supported)
   └── common/
       └── redis_registry.py       # Redis-based service registry for multi-node

Components
----------

Master Server
^^^^^^^^^^^^^

The master server (``omniboxes/master/server.py``) acts as the API gateway and load balancer.

**Responsibilities:**

- Registers and monitors worker nodes
- Routes requests to the least-loaded node
- Provides unified API for all browser operations

**Starting the master:**

.. code-block:: bash

   python -m omniboxes.master.server --port 7000 --nodes http://node1:8080 http://node2:8080

**API Endpoints:**

``POST /get``
   Allocate a new browser instance from the pool.

   Returns: ``{"instance_id": "uuid:port", "node": "node_hash"}``

``POST /reset``
   Reset and release a browser instance.

   Parameters: ``instance_id``, ``node``

``GET /screenshot``
   Capture screenshot from browser instance.

   Parameters: ``instance_id``, ``node``, ``interaction_mode`` (coordinates/set_of_marks)

``POST /execute``
   Execute browser command (click, type, scroll, etc.).

   Body: ``{"node": "...", "instance_id": "...", "<command_name>": {<command_args>}}``

   Example: ``{"node": "abc", "instance_id": "uuid:9000", "visit_page": {"url": "https://example.com"}}``

``GET /info``
   Get cluster status including node health and capacity.

``GET /probe``
   Check if a browser instance is still alive.

   Parameters: ``instance_id``, ``node``

``GET /metadata``
   Get page metadata (title, URL) from a browser instance.

   Parameters: ``instance_id``, ``node``

Node Server
^^^^^^^^^^^

The node server (``omniboxes/node/server.py``) manages a pool of browser instances on a single machine.

**Responsibilities:**

- Manages instance server pool via Redis
- Tracks available vs in-use instances
- Routes requests to correct instance server

**Starting the node:**

.. code-block:: bash

   python -m omniboxes.node.server --port 8080 --workers 32

**Instance Pool Management:**

Uses Redis sets to track instance state:

- ``available``: Ports of idle instance servers
- ``in_use``: Instance IDs currently allocated

Instance Server
^^^^^^^^^^^^^^^

The instance server (``omniboxes/node/instance_server.py``) wraps a single browser instance.

**Responsibilities:**

- Manages one Playwright browser instance
- Handles screenshot capture with annotations
- Executes browser commands

**Starting an instance:**

.. code-block:: bash

   python -m omniboxes.node.instance_server --port 9000

PlaywrightInstance
^^^^^^^^^^^^^^^^^^

The browser implementation (``omniboxes/node/instances/playwright_instance.py``) provides browser automation.

**Supported Commands:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - ``visit_page``
     - Navigate to URL: ``{"url": "https://..."}``
   * - ``click_coords``
     - Click at coordinates: ``{"x": 500, "y": 300}``
   * - ``click_id``
     - Click element by ID: ``{"id": "15"}``
   * - ``fill_coords``
     - Type at coordinates: ``{"x": 500, "y": 300, "value": "text"}``
   * - ``fill_id``
     - Type in element: ``{"id": "15", "value": "text", "press_enter": true}``
   * - ``page_down``
     - Scroll down: ``{"amount": 200}``
   * - ``page_up``
     - Scroll up: ``{"amount": 200}``
   * - ``scroll_id``
     - Scroll element: ``{"id": "15", "direction": "down"}``
   * - ``hover_coords``
     - Hover at coordinates: ``{"x": 500, "y": 300}``
   * - ``hover_id``
     - Hover over element: ``{"id": "15"}``
   * - ``keypress``
     - Press keys: ``{"keys": ["ctrl", "a"]}``
   * - ``back``
     - Navigate back: ``{}``
   * - ``select_option``
     - Select dropdown option: ``{"id": "15"}``
   * - ``hover_and_scroll_coords``
     - Hover and scroll at coordinates: ``{"x": 100, "y": 200, "direction": "down"}``
   * - ``sleep``
     - Wait for duration: ``{"duration": 2.0}``
   * - ``tab_and_enter``
     - Press Tab then Enter: ``{}``
   * - ``get_page_metadata``
     - Get page title and URL
   * - ``get_webpage_text``
     - Get page text content: ``{"n_lines": 100}``
   * - ``get_interactive_rects``
     - Get interactive element rectangles
   * - ``get_screenshot_info``
     - Get last screenshot info
   * - ``screenshot``
     - Capture screenshot: ``{"use_sequential_ids": true}``

**Interaction Modes:**

``set_of_marks`` (default)
   Screenshots include numbered annotations on interactive elements.
   Agent references elements by ID (e.g., ``click([15])``).

``coordinates``
   Plain screenshots without annotations.
   Agent uses pixel coordinates (e.g., ``click(500, 300)``).

Launcher
--------

The launcher (``omniboxes/deploy/deploy.py``) starts all components for local development.

**Usage:**

.. code-block:: bash

   cd omniboxes/deploy && python deploy.py 100

**What it starts:**

1. Redis server (port 6379)
2. N instance servers (ports 9000+)
3. Node server (port 8080)
4. Master server (port 7000)

**Features:**

- Parallel instance startup for fast scaling
- Automatic process recovery
- Graceful shutdown on Ctrl+C

**Configuration:**

.. code-block:: python

   launcher = OmniboxesLauncher(num_instances=100)
   launcher.redis_port = 6379
   launcher.instance_start_port = 9000
   launcher.node_port = 8080
   launcher.master_port = 7000
   launcher.run()

API Usage Examples
------------------

**Allocate and use a browser instance:**

.. code-block:: python

   import requests

   API_URL = "http://localhost:7000"
   HEADERS = {"x-api-key": "your-api-key"}

   # Allocate instance
   response = requests.post(f"{API_URL}/get", headers=HEADERS)
   data = response.json()
   instance_id = data["instance_id"]
   node = data["node"]

   # Navigate to page
   requests.post(f"{API_URL}/execute", headers=HEADERS, json={
       "node": node,
       "instance_id": instance_id,
       "visit_page": {"url": "https://example.com"}
   })

   # Take screenshot
   response = requests.get(f"{API_URL}/screenshot", headers=HEADERS, params={
       "instance_id": instance_id,
       "node": node,
       "interaction_mode": "set_of_marks"
   })
   screenshot_bytes = response.content

   # Click element
   requests.post(f"{API_URL}/execute", headers=HEADERS, json={
       "node": node,
       "instance_id": instance_id,
       "click_id": {"id": "15"}
   })

   # Release instance
   requests.post(f"{API_URL}/reset", headers=HEADERS, params={
       "instance_id": instance_id,
       "node": node
   })

Integration with WebGym
-----------------------

The ``AsyncWebGym`` environment (see :doc:`../rl_pipeline/rl_rollout`) communicates with OmniBoxes via HTTP:

.. code-block:: python

   from webgym.environment.async_webgym import AsyncWebGym

   env = AsyncWebGym(
       master_port=7000,          # OmniBoxes master port
       host_ip="localhost",       # OmniBoxes host
       cpu_cluster_token=token,   # API key
       sampled_tasks=tasks,
       save_path="/path/to/save",
       num_workers=20,            # Concurrent browser instances
       verbose=False,
       retry_policy={"max_retries": 2},
       task_timeout_minutes=20,
       split="train"
   )

Deployment
----------

**Single Machine:**

.. code-block:: bash

   # Start all services
   cd omniboxes/deploy && python deploy.py 50

**Multi-Node Cluster:**

1. Start instance servers and node server on each worker machine:

.. code-block:: bash

   # On each worker node
   cd omniboxes/deploy && python deploy.py 50

2. Start master server with all node URLs:

.. code-block:: bash

   # On master machine
   python -m omniboxes.master.server \
       --port 7000 \
       --nodes http://worker1:8080 http://worker2:8080 http://worker3:8080

Troubleshooting
---------------

**Common Issues:**

``No available nodes with capacity``
   All browser instances are in use. Increase ``--instances`` or add more nodes.

``Instance not in use or already released``
   The instance was already reset. Check for race conditions in your code.

``Redis connection refused``
   Ensure Redis is running: ``redis-cli ping``

**Monitoring:**

Check cluster health:

.. code-block:: bash

   curl -H "x-api-key: your-key" http://localhost:7000/info

Check Redis instance pool:

.. code-block:: bash

   redis-cli smembers available
   redis-cli smembers in_use
