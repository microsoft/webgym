"""WebGym reward function for AReaL RL training.

Evaluates whether the web agent successfully completed its task.
Supports two modes:

1. **Simple mode** (default): Checks if the agent produced an ``answer``
   action in its final completion. Returns 1.0 if an answer tool call
   is found, 0.0 otherwise.

2. **Evaluator mode**: When a WebGym ``Evaluator`` instance is available
   (passed via ``evaluator`` kwarg), uses the full AI-judge pipeline
   to verify task completion against rubrics.
"""

import json
import re

from areal.utils import logging

logger = logging.getLogger("WebGymReward")


def _extract_answer_from_completion(completions: str) -> str | None:
    """Extract the answer content from a model completion.

    Looks for an ``answer`` tool call in the completion text.
    """
    # Try to find tool_call with answer action
    tool_call_match = re.search(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>", completions, re.DOTALL
    )
    if tool_call_match:
        try:
            call = json.loads(tool_call_match.group(1))
            args = call.get("arguments", {})
            action = args.get("action", "")
            if action == "answer":
                return args.get("text", "")
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback: look for answer action pattern in plain text
    answer_match = re.search(
        r'["\']?action["\']?\s*:\s*["\']?answer["\']?', completions, re.IGNORECASE
    )
    if answer_match:
        text_match = re.search(
            r'["\']?text["\']?\s*:\s*["\']([^"\']*)["\']', completions
        )
        if text_match:
            return text_match.group(1)
        return ""

    return None


def webgym_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    **kwargs,
) -> float:
    """Compute reward for a WebGym task completion.

    In simple mode (no evaluator), returns 1.0 if the agent produced an
    answer action, 0.0 otherwise. This provides a basic learning signal
    for the agent to learn to produce answers.

    Parameters
    ----------
    prompt : str
        The prompt text.
    completions : str
        The model's completion text (may span multiple turns).
    prompt_ids : list[int]
        Token IDs of the prompt.
    completion_ids : list[int]
        Token IDs of the completion.
    **kwargs
        Additional fields from the dataset sample. May include
        ``trajectory`` (list of step dicts) and ``evaluator``
        for full evaluation mode.

    Returns
    -------
    float
        Reward value (0.0 or 1.0).
    """
    trajectory = kwargs.get("trajectory")
    evaluator = kwargs.get("evaluator")

    # Full evaluator mode: use AI judge
    if evaluator is not None and trajectory is not None:
        try:
            reward_value, evaluation, is_blocked = evaluator.get_verifiable_reward(
                trajectory
            )
            if is_blocked:
                logger.info("Task blocked by website, reward=0")
                return 0.0
            return float(reward_value)
        except Exception:
            logger.warning("Evaluator failed, falling back to simple mode", exc_info=True)

    # Simple mode: check if agent produced an answer
    answer = _extract_answer_from_completion(completions)
    if answer is not None:
        return 1.0

    return 0.0
