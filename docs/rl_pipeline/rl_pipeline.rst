RL Pipeline Overview
====================

The WebGym RL pipeline consists of several interconnected components located in ``webgym/``.

.. image:: ../../figures/rl_arch.png
   :alt: RL Pipeline Overview
   :align: center

Component Summary
-----------------

**Context Management** (``webgym/context/``)
   Handles conversation building and response parsing for different model types and interaction modes.

**Replay Buffer** (``webgym/data/``)
   Manages trajectory storage, sampling, and filtering for training.

**Rollout Collection** (``webgym/environment/``)
   Orchestrates parallel browser interactions and trajectory collection.

**Policy Configuration** (``webgym/models/``)
   Defines the WebAgent and model interfaces for action generation.

**Utilities** (``webgym/utils/``)
   Shared utilities including blocklist management, image processing, task sampling, task history tracking, and trajectory storage.

   .. code-block:: text

      webgym/utils/
      ├── __init__.py
      ├── blocklist_manager.py      # Blocked website management
      ├── image_utils.py            # Image encoding/decoding for vision models
      ├── rollout_sampler.py        # Task selection strategies for rollouts
      ├── task_history_manager.py   # Task attempt history tracking
      └── trajectory_storage.py     # Incremental trajectory file storage

**WandB Logger** (``webgym/logging/``)
   Provides experiment tracking and logging integration.

Data Flow
---------

.. code-block:: text

   ┌─────────────────┐
   │  Task Sampler   │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐      ┌─────────────────┐
   │  AsyncWebGym    │◄────►│   WebAgent      │
   │  (environment)  │      │   (policy)      │
   └────────┬────────┘      └────────┬────────┘
            │                        │
            │                        ▼
            │               ┌─────────────────┐
            │               │  vLLM Server    │
            │               └─────────────────┘
            ▼
   ┌─────────────────┐
   │  Replay Buffer  │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  LLaMA-Factory  │
   │  Training       │
   └─────────────────┘
