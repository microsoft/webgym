Context Management
==================

The context management module (``webgym/context/``) handles conversation building and response parsing for the web agent.

Module Structure
----------------

.. code-block:: text

   webgym/context/
   ├── __init__.py
   ├── context_manager.py      # Main context manager class
   ├── universal_prompt.py     # Prompt templates
   └── parsers/                # Action parsers
       ├── __init__.py
       ├── base_parser.py          # Abstract parser interface
       ├── coordinates_parser.py   # Coordinates mode parser
       └── set_of_marks_parser.py  # Set-of-marks mode parser

ContextManager
--------------

The ``ContextManager`` class (``context_manager.py``) is the central component that:

- Manages conversation building for different model types
- Handles response parsing
- Supports different interaction modes (coordinates vs set-of-marks)

**Initialization:**

.. code-block:: python

   from webgym.context import ContextManager

   context_manager = ContextManager(
       context_config={'interaction_mode': 'coordinates'},  # default is 'set_of_marks'
       model_config={'model_type': 'qwen3-instruct'},
       verbose=True
   )

**Key Methods:**

``build_conversation(task, trajectory, current_observation, **kwargs)``
   Builds a model-specific conversation from the current state.

``parse_response(raw_response)``
   Parses the model's raw response into structured action data.

``get_interaction_mode()``
   Returns the current interaction mode (``'coordinates'`` or ``'set_of_marks'``).

.. note::
   The ``interaction_mode`` value must use underscores: ``set_of_marks`` (not ``set-of-marks`` with hyphens).

Interaction Modes
-----------------

**Coordinates Mode:**
   Actions are specified using pixel coordinates (x, y).
   Example: ``click(500, 300)``

**Set-of-Marks Mode:**
   Actions reference numbered UI elements.
   Example: ``click([15])`` (clicks element #15)

Parsers
-------

Located in ``webgym/context/parsers/``, parsers convert raw model responses into executable actions:

- Extract action type (click, type, scroll, etc.)
- Parse action parameters
- Validate action format
