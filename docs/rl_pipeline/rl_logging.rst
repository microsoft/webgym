WandB Logger
============

The logging module (``webgym/logging/``) provides WandB integration for experiment tracking.

Module Structure
----------------

.. code-block:: text

   webgym/logging/
   ├── __init__.py
   └── wandb_manager.py

Key Functions
-------------

``initialize_wandb_run(project_name, entity_name, run_name, config=None, wandb_key=None)``
   Initializes a WandB run, resuming an existing run if found. Returns ``(wandb_run, initial_step)``.

``get_latest_wandb_run_info(project_name, entity_name, run_name)``
   Finds the latest WandB run with matching name. Returns ``(run_id, max_step, is_new_run)``.

``check_different_run_names(project_name, entity_name, current_run_name)``
   Checks if there are runs with different names in the project.

``WandBStepManager(initial_step=0)``
   Manages monotonically increasing step numbers across training restarts.

   .. code-block:: python

      from webgym.logging import WandBStepManager
      step_manager = WandBStepManager(initial_step=100)
      next_step = step_manager.get_next_step(local_step=5)

Configuration
-------------

.. code-block:: yaml

   log_config:
     run_name: "webgym-<your-run-name>"
     wandb_key_env_var: "WANDB_API_KEY"
     entity_name: "<your-wandb-entity-name>"

   algorithm_config:
     report_to: "wandb"

The only required environment variable is ``WANDB_API_KEY``. Project and entity names are configured via ``log_config`` in ``update_online.yaml``.
