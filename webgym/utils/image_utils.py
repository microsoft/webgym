"""
Image utilities for context management and vision model integration.
"""
import base64
import io
import os
import json
from pathlib import Path
from typing import List, Union, Dict, Any
from PIL import Image


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 data URL

    NOTE: This function is only for OpenAI API evaluation requests.
    For vLLM inference, use encode_image_to_file_url() instead.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        ext = Path(image_path).suffix.lower()
        if ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif ext == ".png":
            mime_type = "image/png"
        else:
            mime_type = "image/jpeg"
        return f"data:{mime_type};base64,{encoded_string}"


def encode_image_to_file_url(image_path: str) -> str:
    """Convert image file path to file:// URL for vLLM

    This is more efficient than base64 encoding as vLLM can read the file directly.

    Args:
        image_path: Path to image file (absolute or relative)

    Returns:
        file:// URL string
    """
    abs_path = os.path.abspath(image_path)
    return f"file://{abs_path}"


def convert_messages_to_path_format(messages: Union[List[Dict], Dict], trajectory: List[Dict] = None, current_observation: Dict = None) -> str:
    """
    Convert messages to string format for storage in trajectory files.

    Since all messages now use file:// URLs (not base64), this function simply
    serializes the messages to JSON for compact storage.

    Args:
        messages: Message(s) with file:// image URLs
        trajectory: Not used (kept for backward compatibility)
        current_observation: Not used (kept for backward compatibility)

    Returns:
        JSON string representation of messages

    Raises:
        ValueError: If messages contain base64-encoded images (not supported)
    """
    # Handle both single message dict and list of messages
    if isinstance(messages, dict):
        messages = [messages]

    # Validate that all images use file:// URLs (no base64)
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:image"):
                        raise ValueError(
                            "Base64-encoded images are not supported. "
                            "All images must use file:// URLs. "
                            f"Found: {image_url[:50]}..."
                        )
                    if not image_url.startswith("file://"):
                        raise ValueError(
                            f"Invalid image URL format. Expected file:// URL, got: {image_url[:50]}..."
                        )

    # Return JSON string for compact storage
    return json.dumps(messages, indent=2, ensure_ascii=False)
