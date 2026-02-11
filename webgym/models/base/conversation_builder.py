# webgym/models/base/conversation_builder.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union

class ConversationBuilder(ABC):
    """Abstract base for building model-specific conversation formats"""

    @abstractmethod
    def build_conversation(self,
                          task: str,
                          trajectory: List[Dict],
                          current_observation: Dict,
                          **kwargs) -> Union[List[Dict], Dict]:
        """
        Build conversation in model-specific format

        Args:
            task: Task description/goal
            trajectory: List of trajectory steps (observation, response, action)
            current_observation: Current observation dict with image_path, ac_tree, etc.
            **kwargs: Additional model-specific parameters

        Returns:
            List of message dicts for multi-turn conversation
        """
        pass

    @abstractmethod
    def get_conversation_type(self) -> str:
        """
        Return conversation type

        Returns:
            'single-turn' or 'multi-turn'
        """
        pass
