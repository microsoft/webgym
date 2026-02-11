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
   │   ├── __init__.py
   │   ├── model_interface.py       # Abstract model interface
   │   ├── conversation_builder.py  # Conversation building
   │   ├── evaluation_prompt.py     # Evaluation prompt templates
   │   └── prompt_processing.py     # Prompt processing utilities
   └── qwen/
       ├── __init__.py
       ├── qwen_interface.py        # Qwen3-VL model interface implementation
       └── conversation_builder.py  # Qwen3-VL conversation builder

Architecture Overview
---------------------

The module separates concerns into two main classes:

- **WebAgent**: Handles action generation via vLLM inference
- **Evaluator**: Handles reward evaluation via OpenAI/Gemini API

This separation allows:

- Independent scaling of inference and evaluation
- Cleaner code organization
- Flexibility to use different evaluation backends

WebAgent
--------

The ``WebAgent`` class (``web_agent.py``) generates browser actions using vLLM for model inference.

**Key Responsibilities:**

- vLLM HTTP client management and connection pooling
- Conversation building via ContextManager
- Action generation and parsing
- Browser command translation

**Initialization:**

.. code-block:: python

   from webgym.models import WebAgent

   agent = WebAgent(
       policy_config=policy_config,
       context_config=context_config,
       model_config={'model_type': 'qwen3-instruct'},
       save_path='/path/to/checkpoints',
       vllm_server_url='http://localhost:8999',
       openai_config=openai_config,  # Optional: creates internal Evaluator
       operation_timeout=120,        # General operation timeout in seconds
       vllm_timeout=120,             # vLLM request timeout in seconds
       max_retries=1,                # Maximum HTTP retries
       max_vllm_sessions=32,         # Maximum concurrent vLLM sessions
       verbose=True                  # Enable verbose logging
   )

**Key Methods:**

``get_action_and_observation_sync(trajectory, screenshot_path, page_metadata, step_data)``
   Returns the next action based on current state and trajectory history.

``parse_action_to_browser_command(action)``
   Converts an Action object to a browser-executable command.

**Accessing the Evaluator:**

When ``openai_config`` is provided, the agent creates an internal ``Evaluator`` accessible via:

.. code-block:: python

   # Direct evaluator access
   reward, evaluation, is_blocked = agent.evaluator.get_verifiable_reward(trajectory)

Evaluator
---------

The ``Evaluator`` class (``evaluator.py``) handles trajectory evaluation using vision models (supports OpenAI and Gemini backends).

**Key Responsibilities:**

- Reward computation based on task completion
- Blocking detection (CAPTCHA, anti-bot measures)
- Image relevance judgment for evaluation
- Multi-criteria verification (Criterion A and B)

**Initialization:**

.. code-block:: python

   from webgym.models import Evaluator

   evaluator = Evaluator(
       openai_config={
           'model': 'gemini-3-flash-preview',
           'openai_api_key_env_var': 'GEMINI_API_KEY',
           'base_url': 'https://generativelanguage.googleapis.com/v1beta/openai/'
       },
       conversation_builder=agent.prompt_constructor,
       max_retries=1,
       verbose=True
   )

**Key Methods:**

``get_verifiable_reward(trajectory)``
   Evaluates a completed trajectory and returns ``(reward, evaluation_text, is_blocked)``.
   Note: ``evaluation_text`` may be a string or a list of evaluation response strings.

   - Internally calls ``judge_submission_images()`` to select relevant screenshots
   - Runs Criterion B (anti-hallucination) check
   - Runs Criterion A (fact verification) for each rubric
   - Returns binary reward (0 or 1)

``check_if_blocked(trajectory)``
   Analyzes trajectory screenshots to detect website blocking.
   Samples up to 20 screenshots and checks for CAPTCHA, access denied pages, etc.

``judge_submission_images(trajectory)``
   Processes trajectory images in parallel to determine which should be submitted for evaluation.
   Sets ``submit=True/False`` on each step's reward object.

``check_single_screenshot_for_blocking(screenshot_path, task_name, step_number)``
   Checks a single screenshot for blocking indicators.
   Used for real-time blocking detection during rollout.

Evaluation Criteria
-------------------

The evaluator uses a two-criteria system:

**Criterion B (Anti-Hallucination):**
   Verified ONCE per task. Checks if the agent's response is supported by the submitted screenshots.

**Criterion A (Fact Verification):**
   Verified for EACH rubric/fact in the task. Checks if screenshots contain evidence for each required fact.

**Final Reward Calculation:**

.. code-block:: text

   reward = 1 if (Criterion B passes) AND (ALL Criterion A checks pass)
   reward = 0 otherwise

Usage in AsyncWebGym
--------------------

The ``AsyncWebGym`` environment accesses the evaluator through the agent:

.. code-block:: python

   # In async_webgym.py
   reward_value, evaluation, is_blocked = agent.evaluator.get_verifiable_reward(trajectory)

   # For blocking detection when agent doesn't answer
   is_blocked = agent.evaluator.check_if_blocked(trajectory)

Configuration
-------------

**OpenAI Config:**

.. code-block:: yaml

   openai_config:
     model: "gemini-3-flash-preview"
     openai_api_key_env_var: "GEMINI_API_KEY"
     base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"

**Policy Config (for WebAgent):**

.. code-block:: yaml

   policy_config:
     base_model: "Qwen/Qwen3-VL-8B-Instruct"
     temperature: 1
     top_p: 0.99
     top_k: 2
     max_new_tokens: 3072
