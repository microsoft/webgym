# webgym/models/base/model_interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union
from .conversation_builder import ConversationBuilder

class ModelInterface(ABC):
    """Abstract interface for different VLM models"""

    def __init__(self):
        self.conversation_builder: ConversationBuilder = None

    def build_conversation(self, task: str, trajectory: List[Dict], current_observation: Dict, **kwargs) -> Union[List[Dict], Dict]:
        """
        Build model-specific conversation format

        Args:
            task: Task description
            trajectory: List of trajectory steps
            current_observation: Current observation with image and metadata
            **kwargs: Additional parameters

        Returns:
            Conversation in model-specific format
        """
        if self.conversation_builder is None:
            raise NotImplementedError("conversation_builder not initialized")

        return self.conversation_builder.build_conversation(
            task, trajectory, current_observation, **kwargs
        )

    def is_multi_turn(self) -> bool:
        """Check if model uses multi-turn conversation"""
        if self.conversation_builder is None:
            return False
        return self.conversation_builder.get_conversation_type() == "multi-turn"

    @abstractmethod
    def parse_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Parse model-specific response format

        Args:
            raw_response: Raw string response from the model

        Returns:
            Dict with parsed fields (thought, action, memory, etc.)
        """
        pass

    @abstractmethod
    def extract_action(self, parsed_response: Dict) -> Dict[str, Any]:
        """
        Extract action in standardized format

        Args:
            parsed_response: Parsed response dict

        Returns:
            Dict with 'key' (action type) and 'arguments' (action args)
        """
        pass

    @abstractmethod
    def supports_native_thinking(self) -> bool:
        """
        Whether model has native <think> token support

        Returns:
            True if model supports thinking tokens (like Qwen-Thinking)
        """
        pass
