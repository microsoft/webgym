Rollout Collection
==================

The rollout collection module (``webgym/environment/``) handles trajectory collection through browser interactions.

Module Structure
----------------

.. code-block:: text

   webgym/environment/
   ├── async_webgym.py           # Main async rollout environment
   ├── client.py                 # HTTP client for browser server
   ├── task_monitor.py           # Task progress monitoring
   ├── process_isolator.py       # Process isolation for stability
   ├── pickleable_http_functions.py  # Serializable HTTP operations
   ├── actions.py                # Action definitions
   └── foundry_endpoints_models.py   # API endpoint models

AsyncWebGym
-----------

The ``AsyncWebGym`` class (``async_webgym.py``) is the main environment for collecting trajectories.

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
       split='train',
       interaction_mode='coordinates'
   )

**Key Parameters:**

- ``num_workers``: Concurrent browser instances
- ``task_timeout_minutes``: Max time per task before timeout
- ``completion_threshold``: Fraction of tasks to complete before killing stragglers (e.g., 0.98)
- ``completion_grace_period``: Seconds to wait before killing remaining tasks

Evaluation Integration
----------------------

AsyncWebGym integrates with the ``Evaluator`` class (see :doc:`rl_models`) for reward computation:

.. code-block:: python

   reward_value, evaluation, is_blocked = agent.evaluator.get_verifiable_reward(trajectory)
   is_blocked = agent.evaluator.check_if_blocked(trajectory)
