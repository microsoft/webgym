import re
from typing import Dict, Any
from .components import Response

def decompose_raw_response(raw_response: str, model_interface=None, raw_prompt: str = "") -> Response:
    """
    Decompose raw model response into structured Response object

    Args:
        raw_response: Raw string response from the model
        model_interface: ModelInterface instance for parsing (required)
        raw_prompt: Raw string prompt sent to the model (optional)

    Returns:
        Response object with parsed answering_tokens

    Raises:
        ValueError: If model_interface is not provided
    """
    if model_interface is None:
        raise ValueError("model_interface is required for response parsing")

    answering_tokens = model_interface.parse_response(raw_response)

    return Response(
        raw_response=raw_response,  # Keep for debugging/logging
        answering_tokens=answering_tokens,
        raw_prompt=raw_prompt  # Keep for debugging/logging
    )

def get_action_string(response: Response) -> str:
    """Extract action string from Response"""
    return response.answering_tokens.get("action", "")
