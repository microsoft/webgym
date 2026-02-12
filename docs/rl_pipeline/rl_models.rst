Models
======

The models module (``webgym/models/``) provides the core agent and evaluation components for web automation.

Module Structure
----------------

.. code-block:: text

   webgym/models/
   ├── __init__.py
   ├── web_agent.py         # WebAgent class for action generation
   ├── evaluator.py         # Evaluator class for reward computation
   ├── model_factory.py     # Model interface factory
   ├── base/
   │   ├── model_interface.py       # Abstract model interface
   │   ├── conversation_builder.py  # Conversation building
   │   ├── evaluation_prompt.py     # Evaluation prompt templates
   │   └── prompt_processing.py     # Prompt processing utilities
   └── qwen/
       ├── qwen_interface.py        # Qwen3-VL model interface
       └── conversation_builder.py  # Qwen3-VL conversation builder

WebAgent
--------

The ``WebAgent`` class (``web_agent.py``) generates browser actions using vLLM for inference.

.. code-block:: python

   from webgym.models import WebAgent

   agent = WebAgent(
       policy_config=policy_config,
       context_config=context_config,
       model_config={'model_type': 'qwen3-instruct'},
       save_path='/path/to/checkpoints',
       vllm_server_url='http://localhost:8999',
       openai_config=openai_config,  # Creates internal Evaluator
       operation_timeout=120,
       vllm_timeout=120,
       max_retries=1,
       max_vllm_sessions=32,
       verbose=True
   )

**Key Methods:**

``get_action_and_observation_sync(trajectory, screenshot_path, page_metadata, step_data)``
   Returns ``(Action, Response)`` for the next step based on current state and trajectory history.

``parse_action_to_browser_command(action)``
   Converts an Action object to a browser-executable command.

When ``openai_config`` is provided, the evaluator is accessible via ``agent.evaluator``.

Evaluator
---------

The ``Evaluator`` class (``evaluator.py``) handles trajectory evaluation using vision models (OpenAI/Gemini).

**Key Methods:**

``get_verifiable_reward(trajectory)``
   Returns ``(reward, evaluation_texts, is_blocked)``. Uses multi-criteria verification:

   1. ``judge_submission_images()``: Select relevant screenshots
   2. **Criterion B** (anti-hallucination): Check agent's response against screenshots
   3. **Criterion A** (fact verification): Check each rubric/fact against screenshots
   4. **Reference Answer** (Step 4): If reference exists and all Criterion A passed, verify answer match

   Final reward: ``1`` if all checks pass, ``0`` otherwise.

``check_if_blocked(trajectory)``
   Samples up to 20 screenshots to detect CAPTCHA/blocking pages.

``check_single_screenshot_for_blocking(screenshot_path, task_name, step_number)``
   Real-time blocking detection during rollout.

Configuration
-------------

.. code-block:: yaml

   openai_config:
     model: "gemini-3-flash-preview"
     openai_api_key_env_var: "GEMINI_API_KEY"
     base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"

   policy_config:
     base_model: "Qwen/Qwen3-VL-8B-Instruct"
     temperature: 1
     top_p: 0.99
     top_k: 2
     max_new_tokens: 3072
