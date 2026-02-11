Rollout Collection
==================

The rollout collection module (``webgym/environment/``) handles trajectory collection through browser interactions.

Module Structure
----------------

.. code-block:: text

   webgym/environment/
   ├── __init__.py
   ├── async_webgym.py           # Main async rollout environment
   ├── client.py                 # HTTP client for browser server
   ├── task_monitor.py           # Task progress monitoring
   ├── process_isolator.py       # Process isolation for stability
   ├── pickleable_http_functions.py  # Serializable HTTP operations
   ├── actions.py                # Action definitions
   └── foundry_endpoints_models.py   # API endpoint models

AsyncWebGym
-----------

The ``AsyncWebGym`` class (``async_webgym.py``) is the main environment for collecting trajectories:

**Key Features:**

- Async/parallel trajectory collection
- HTTP-based communication with browser servers
- Automatic task timeout and completion handling
- Support for multi-node distributed rollouts

**Initialization:**

.. code-block:: python

   from webgym.environment.async_webgym import AsyncWebGym

   env = AsyncWebGym(
       master_port=7000,
       host_ip="localhost",
       cpu_cluster_token=token,
       sampled_tasks=tasks,
       save_path="/path/to/save",
       num_workers=20,
       verbose=True,
       retry_policy=retry_config,
       task_timeout_minutes=20,
       completion_threshold=0.98,
       completion_grace_period=120,
       blocklist_manager=None,
       skip_instance_cleanup=False,
       multinode_rank_suffix='',
       split='train',
       env_config=None,
       interaction_mode='coordinates'
   )

**Key Parameters:**

``num_workers``
   Number of concurrent browser instances

``task_timeout_minutes``
   Maximum time per task before timeout

``completion_threshold``
   Percentage of tasks to complete before killing stragglers (e.g., 0.98 = 98%)

``completion_grace_period``
   Seconds to wait before killing remaining tasks

Task Monitor
------------

The ``TaskMonitor`` class (``task_monitor.py``) tracks rollout progress:

- Real-time task status updates
- Success/failure rate tracking
- Timeout detection and handling
- Progress visualization

HTTP Client
-----------

The ``client.py`` module provides HTTP communication with browser servers:

- Screenshot capture
- Action execution (click, type, scroll)
- Page navigation
- Browser instance management

Evaluation Integration
----------------------

AsyncWebGym integrates with the ``Evaluator`` class (see :doc:`rl_models`) for reward computation:

.. code-block:: python

   # Access evaluator through the agent
   reward_value, evaluation, is_blocked = agent.evaluator.get_verifiable_reward(trajectory)

   # Check for blocking when agent doesn't answer
   is_blocked = agent.evaluator.check_if_blocked(trajectory)

The evaluator handles:

- Multi-criteria reward verification (Criterion A and B)
- Blocking detection (CAPTCHA, anti-bot pages)
- Image relevance judgment for evaluation submission
