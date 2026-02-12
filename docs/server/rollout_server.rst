Rollout Server Code Details (Optional Read)
============

The rollout server (``omniboxes/``) provides distributed browser automation for collecting web agent trajectories at scale.

Architecture Overview
---------------------

OmniBoxes uses a three-tier architecture:

.. code-block:: text

   Master Server (port 7000)   ← API gateway + load balancer
       ├── Node Server (Machine 1, port 8080)
       │       └── Instance Servers (9000+)
       ├── Node Server (Machine 2, port 8080)
       │       └── Instance Servers (9000+)
       └── Node Server (Machine N, port 8080)
               └── Instance Servers (9000+)

Module Structure
----------------

.. code-block:: text

   omniboxes/
   ├── master/                    # Master server (load balancer)
   │   ├── server.py             # FastAPI master server
   │   ├── node_manager.py       # Node registration and health
   │   └── logging_utils.py
   ├── node/                      # Node server (instance pool manager)
   │   ├── server.py             # FastAPI node server
   │   ├── instance_server.py    # Single instance server
   │   ├── logging_utils.py
   │   ├── instances/            # Browser instance implementations
   │   │   ├── base.py
   │   │   ├── playwright_instance.py
   │   │   ├── _playwright_controller.py
   │   │   ├── _set_of_marks.py
   │   │   ├── _types.py
   │   │   └── _page_script.js
   │   └── utils.py
   ├── deploy/                    # Deployment utilities
   │   ├── deploy.py             # Single-node CLI
   │   ├── process_manager.py    # Server lifecycle management
   │   ├── deploy_multinode.py   # Multi-node CLI
   │   ├── multinode_manager.py  # Multi-node launcher
   │   └── nginx_manager.py
   └── common/
       └── redis_registry.py     # Redis-based service registry

Components
----------

Master Server
^^^^^^^^^^^^^

The master server (``omniboxes/master/server.py``) routes requests to the least-loaded node.

**API Endpoints:**

``POST /get``
   Allocate a browser instance. Returns: ``{"instance_id": "uuid:port", "node": "node_hash"}``

``POST /reset``
   Reset and release a browser instance. Params: ``instance_id``, ``node``

``GET /screenshot``
   Capture screenshot. Params: ``instance_id``, ``node``, ``interaction_mode`` (coordinates/set_of_marks)

``POST /execute``
   Execute browser command. Body: ``{"node": "...", "instance_id": "...", "<command>": {<args>}}``

``GET /info``
   Cluster status (node health and capacity).

``GET /probe``
   Check if instance is alive. Params: ``instance_id``, ``node``

``GET /metadata``
   Page metadata (title, URL). Params: ``instance_id``, ``node``

Node Server
^^^^^^^^^^^

The node server (``omniboxes/node/server.py``) manages a pool of browser instances on a single machine using Redis sets (``available`` and ``in_use``).

Instance Server
^^^^^^^^^^^^^^^

The instance server (``omniboxes/node/instance_server.py``) wraps a single Playwright browser instance.

Supported Commands
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - ``visit_page``
     - Navigate to URL: ``{"url": "https://..."}``
   * - ``click_coords`` / ``click_id``
     - Click at coordinates or by element ID
   * - ``fill_coords`` / ``fill_id``
     - Type text at coordinates or in element: ``{"value": "text", "press_enter": true, "delete_existing": false}``
   * - ``page_down`` / ``page_up``
     - Scroll: ``{"amount": 200, "full_page": false}``
   * - ``scroll_id``
     - Scroll element: ``{"id": "15", "direction": "down"}``
   * - ``hover_coords`` / ``hover_id``
     - Hover at coordinates or over element
   * - ``keypress``
     - Press keys: ``{"keys": ["ctrl", "a"]}``
   * - ``back``
     - Navigate back
   * - ``select_option``
     - Select dropdown: ``{"id": "15"}``
   * - ``hover_and_scroll_coords``
     - Hover and scroll: ``{"x": 100, "y": 200, "direction": "down"}``
   * - ``sleep``
     - Wait: ``{"duration": 2.0}``
   * - ``tab_and_enter``
     - Press Tab then Enter
   * - ``get_page_metadata``
     - Get page title and URL
   * - ``get_webpage_text``
     - Get text content: ``{"n_lines": 100}``
   * - ``get_interactive_rects``
     - Get interactive element rectangles
   * - ``screenshot``
     - Capture screenshot: ``{"interaction_mode": "set_of_marks"}``

**Interaction Modes:**

- ``set_of_marks`` (default): Screenshots include numbered annotations. Agent references by ID (``click([15])``).
- ``coordinates``: Plain screenshots. Agent uses pixel coordinates (``click(500, 300)``).

Integration with WebGym
-----------------------

.. code-block:: python

   from webgym.environment.async_webgym import AsyncWebGym

   env = AsyncWebGym(
       master_port=7000,
       host_ip="localhost",
       cpu_cluster_token=token,
       sampled_tasks=tasks,
       save_path="/path/to/save",
       num_workers=20,
       split="train"
   )

Multi-Node Deployment
---------------------

Use ``deploy_multinode.py`` for multi-node deployments with Redis-based service discovery.
See :doc:`multinode_deployment` for detailed setup.

Troubleshooting
---------------

``No available nodes with capacity``
   All browser instances in use. Increase instances or add more nodes.

``Instance not in use or already released``
   Instance was already reset. Check for race conditions.

``Redis connection refused``
   Ensure Redis is running: ``redis-cli ping``

**Monitoring:**

.. code-block:: bash

   curl -H "x-api-key: your-key" http://localhost:7000/info
   redis-cli smembers available
   redis-cli smembers in_use
