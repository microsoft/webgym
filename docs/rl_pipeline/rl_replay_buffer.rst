Replay Buffer
=============

The replay buffer module (``webgym/data/``) manages training data storage and sampling for RL training.

Module Structure
----------------

.. code-block:: text

   webgym/data/
   ├── __init__.py
   ├── replay_buffer.py        # Main replay buffer class
   ├── components.py           # Data structures (Task, Action, Reward, etc.)
   └── response_decomposer.py  # Response parsing utilities

ReplayBuffer
------------

The ``ReplayBuffer`` class (``replay_buffer.py``) extends PyTorch's ``Dataset`` and provides:

- Trajectory storage and management
- Filtering for successful/unsuccessful trajectories
- Same-screenshot step filtering
- Support for distributed training

**Initialization:**

.. code-block:: python

   from webgym.data import ReplayBuffer

   replay_buffer = ReplayBuffer(
       trajectories=trajectory_list,
       agent=web_agent,
       capacity=None,  # None = unlimited
       filter_successful_only=False,
       include_reward_in_sample=True,
       shuffle=False,
       filter_same_screenshot=True
   )

**Key Parameters:**

``trajectories``
   List of trajectory data to process

``agent``
   WebAgent instance for context management

``capacity``
   Maximum number of samples to store (optional)

``filter_successful_only``
   If True, only samples from successful trajectories are accessible

``include_reward_in_sample``
   Include reward information in each sample (default: True)

``shuffle``
   Shuffle samples (default: False)

``filter_same_screenshot``
   Filter out steps where screenshots haven't changed (default: True)

Data Components
---------------

``webgym/data/components.py`` defines core data structures:

- ``Task``: Task description and metadata (task_name, domain, subdomain, website, difficulty, evaluator_reference, reference_answer, attempt_level, task_id, max_steps, trajectory_index)
- ``Observation``: Screenshot and page state (task, image_path, ac_tree, page_metadata)
- ``Action``: Agent action (action, action_string)
- ``Response``: Model response (raw_response, answering_tokens, raw_prompt)
- ``Reward``: Task completion reward (reward, evaluation, is_blocked, submit, submission_judgment)

**Key Methods:**

``get_training_samples(num_samples=None, recency_bias_power=1.0)``
   Returns training-eligible samples with recency-weighted sampling. Samples from successful
   trajectory steps. If ``recency_bias_power`` is 1.0, uses uniform random sampling.
