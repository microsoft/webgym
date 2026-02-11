# webgym/models/model_factory.py
from typing import Dict, Any
from .base.model_interface import ModelInterface

def create_model_interface(model_config: Dict[str, Any]) -> ModelInterface:
    """
    Factory function to create appropriate model interface

    Args:
        model_config: Config dict with:
            - model_type: 'qwen3-instruct' (uses 'instruct' variant) | 'qwen3-think' (uses 'thinking' variant)
            - interaction_mode: 'coordinates' | 'set_of_marks'
            - prompt_version: 'vanilla' | 'complete' (optional, defaults to 'vanilla')

    Returns:
        ModelInterface instance

    Raises:
        ValueError: If model_type is not supported

    Example:
        >>> config = {
        ...     'model_type': 'qwen3-instruct',
        ...     'interaction_mode': 'coordinates'
        ... }
        >>> interface = create_model_interface(config)
    """
    model_type = model_config.get('model_type', 'qwen3-instruct').lower()
    interaction_mode = model_config.get('interaction_mode', 'set_of_marks')

    if model_type in ['qwen3-instruct', 'qwen3-think']:
        from .qwen.qwen_interface import Qwen3VLInterface

        # Derive variant from model_type
        variant = 'thinking' if model_type == 'qwen3-think' else 'instruct'

        prompt_version = model_config.get('prompt_version', 'vanilla')
        return Qwen3VLInterface(
            interaction_mode=interaction_mode,
            variant=variant,
            prompt_version=prompt_version
        )

    else:
        raise ValueError(
            f"Unsupported model_type: {model_type}. "
            f"Supported types are: 'qwen3-instruct', 'qwen3-think'"
        )


def get_supported_models():
    """
    Get list of supported model types

    Returns:
        List of supported model type strings
    """
    return ['qwen3-instruct', 'qwen3-think']
