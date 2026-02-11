WandB Logger
============

The logging module (``webgym/logging/``) provides WandB integration for experiment tracking.

Module Structure
----------------

.. code-block:: text

   webgym/logging/
   ├── __init__.py
   └── wandb_manager.py       # WandB management utilities

WandB Manager
-------------

The ``wandb_manager.py`` module provides utilities for WandB experiment tracking:

**Key Functions:**

``get_latest_wandb_run_info(project_name, entity_name, run_name)``
   Finds the latest WandB run with matching name and returns its ID and step count.

   .. code-block:: python

      from webgym.logging.wandb_manager import get_latest_wandb_run_info

      run_id, max_step, is_new_run = get_latest_wandb_run_info(
          project_name="rl",
          entity_name="my-team",
          run_name="experiment-1"
      )

   Returns:
      - ``run_id``: ID of the existing run (or None for new run)
      - ``max_step``: Maximum step logged so far
      - ``is_new_run``: Whether this is a new run

``check_different_run_names(project_name, entity_name, current_run_name)``
   Check if there are runs with different names in the project.

``initialize_wandb_run(project_name, entity_name, run_name, config=None, wandb_key=None)``
   Initializes a WandB run, resuming an existing run if found.

   .. code-block:: python

      from webgym.logging import initialize_wandb_run

      wandb_run, initial_step = initialize_wandb_run(
          project_name="rl",
          entity_name="my-team",
          run_name="experiment-1"
      )

   Returns:
      - ``wandb_run``: The WandB run object (or None if initialization fails)
      - ``initial_step``: The step to resume logging from

**Key Classes:**

``WandBStepManager(initial_step=0)``
   Manages monotonically increasing step numbers across training restarts.

   .. code-block:: python

      from webgym.logging import WandBStepManager

      step_manager = WandBStepManager(initial_step=100)
      next_step = step_manager.get_next_step(local_step=5)

Features
--------

**Run Resumption:**
   Automatically detects existing runs and resumes logging from the last step.

**Step Synchronization:**
   Ensures monotonically increasing step numbers across training restarts.

**Dynamic Step Management:**
   Handles step tracking for continuous RL training loops.

Configuration
-------------

WandB settings are configured in ``update_online.yaml``:

.. code-block:: yaml

   log_config:
     run_name: "webgym-<your-run-name>"
     wandb_key_env_var: "WANDB_API_KEY"
     entity_name: "<your-wandb-entity-name>"

   algorithm_config:
     report_to: "wandb"

Environment Variables
---------------------

``WANDB_API_KEY``
   Your WandB API key for authentication. This is the only WandB environment variable used by the codebase.
   Project and entity names are configured via ``log_config`` in ``update_online.yaml``.
