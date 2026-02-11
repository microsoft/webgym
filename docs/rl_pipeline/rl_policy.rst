Policy Configuration
====================

The policy module (``webgym/models/``) defines the web agent and model interfaces.

Module Structure
----------------

.. code-block:: text

   webgym/models/
   ├── __init__.py
   ├── web_agent.py          # Main WebAgent class
   ├── model_factory.py      # Model interface factory
   ├── base/                  # Base classes and utilities
   │   ├── __init__.py
   │   ├── model_interface.py
   │   ├── conversation_builder.py
   │   ├── evaluation_prompt.py
   │   └── prompt_processing.py
   └── qwen/                  # Qwen-specific implementations
       └── ...

WebAgent
--------

The ``WebAgent`` class (``web_agent.py``) is the main policy interface:

**Initialization:**

.. code-block:: python

   from webgym.models.web_agent import WebAgent

   agent = WebAgent(
       policy_config=policy_cfg,
       context_config=context_cfg,
       model_config={'model_type': 'qwen3-instruct'},
       save_path="/path/to/save",
       vllm_server_url="http://localhost:8999",
       openai_config=openai_cfg,
       vllm_timeout=120,
       max_vllm_sessions=32
   )

**Key Features:**

- vLLM integration for fast inference
- Context management for conversation building
- OpenAI API integration for reward evaluation
- Concurrent request handling with semaphores

**Key Parameters:**

``vllm_server_url``
   URL of the vLLM server for model inference

``vllm_timeout``
   Timeout for vLLM requests in seconds

``max_vllm_sessions``
   Maximum concurrent vLLM requests

``openai_config``
   Configuration for OpenAI-based reward evaluation

Model Factory
-------------

The ``model_factory.py`` creates model-specific interfaces:

.. code-block:: python

   from webgym.models.model_factory import create_model_interface

   interface = create_model_interface({
       'model_type': 'qwen3-instruct'  # or 'qwen3-think'
   })

**Supported Models:**

- ``qwen3-instruct``: Qwen/Qwen3-VL-8B-Instruct (standard)
- ``qwen3-think``: Qwen/Qwen3-VL-8B-Thinking (with reasoning)

Base Classes
------------

``ModelInterface`` (``base/model_interface.py``)
   Abstract interface for model-specific operations

``ConversationBuilder`` (``base/conversation_builder.py``)
   Builds conversations in model-specific formats

``prompt_processing.py``
   Utilities for preparing prompts for vLLM format (``batch_get_vllm_prompts``) and HuggingFace format (``batch_get_hf_prompts``)
