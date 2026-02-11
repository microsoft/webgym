# webgym/models/qwen/qwen_interface.py
import re
import json
from typing import Dict, Any
from ..base.model_interface import ModelInterface
from .conversation_builder import QwenConversationBuilder

class Qwen3VLInterface(ModelInterface):
    """Qwen3-VL specific implementation"""

    def __init__(self, interaction_mode: str, variant: str = "instruct", prompt_version: str = "vanilla"):
        """
        Initialize Qwen3-VL interface

        Args:
            interaction_mode: 'coordinates' or 'set_of_marks'
            variant: 'instruct' or 'thinking'
            prompt_version: 'vanilla' or 'complete'
        """
        super().__init__()
        self.interaction_mode = interaction_mode
        self.variant = variant.lower()
        self.prompt_version = prompt_version.lower()
        self.conversation_builder = QwenConversationBuilder(interaction_mode, variant, prompt_version)

    def parse_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Parse Qwen3-VL response format

        Args:
            raw_response: Raw string response from Qwen

        Returns:
            Dict with thought, action_description, tool_call, thinking, observation, memory fields
        """
        # Extract thinking content and create response for parsing Action/tool_call
        thinking_content = ""
        response_for_parsing = raw_response

        if self.variant == 'thinking':
            # Try to match both opening and closing tags
            think_match = re.search(r'<think>(.*?)</think>', raw_response, re.DOTALL)
            if think_match:
                thinking_content = think_match.group(1).strip()
                # Remove thinking tags for parsing Action/tool_call
                response_for_parsing = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
            else:
                # If no opening tag, try to match content before closing tag
                # This handles models that auto-start thinking without outputting <think>
                think_end_match = re.search(r'^(.*?)</think>', raw_response, re.DOTALL)
                if think_end_match:
                    thinking_content = think_end_match.group(1).strip()
                    # Remove everything up to and including </think>
                    response_for_parsing = re.sub(r'^.*?</think>\s*', '', raw_response, flags=re.DOTALL)

        # Extract Thought (if present)
        thought_match = re.search(r'Thought:\s*(.*?)(?=Action:|<tool_call>|$)', response_for_parsing, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else ""

        # If thinking variant and no explicit Thought, use first line of thinking
        if self.variant == 'thinking' and not thought and thinking_content:
            thought = thinking_content.split('\n')[0][:200]  # First 200 chars

        # Extract Action description (if present)
        action_desc_match = re.search(r'Action:\s*(.*?)(?=<tool_call>|$)', response_for_parsing, re.DOTALL | re.IGNORECASE)
        action_desc = action_desc_match.group(1).strip() if action_desc_match else ""

        # Extract tool call
        tool_call_match = re.search(r'<tool_call>\s*({.*?})\s*</tool_call>', response_for_parsing, re.DOTALL)
        tool_call = {}
        if tool_call_match:
            try:
                tool_call = json.loads(tool_call_match.group(1))
            except json.JSONDecodeError:
                tool_call = {}

        # Extract Memory (if present) - supports both "Memory:" and "Memory_Updated:" formats
        memory_match = re.search(r'Memory(?:_Updated)?:\s*(.*?)(?=Action:|<tool_call>|$)', response_for_parsing, re.DOTALL | re.IGNORECASE)
        memory = memory_match.group(1).strip() if memory_match else ""

        return {
            "thought": thought,
            "action": action_desc,  # Keep as 'action' for compatibility
            "tool_call": tool_call,
            "thinking": thinking_content if self.variant == 'thinking' else "",
            "observation": "",  # Set by environment
            "memory": memory
        }

    def extract_action(self, parsed_response: Dict) -> Dict[str, Any]:
        """
        Extract action from tool call

        Args:
            parsed_response: Parsed response dict

        Returns:
            Dict with 'key' and 'arguments'
        """
        tool_call = parsed_response.get("tool_call", {})

        if not tool_call:
            return {"key": "wait", "arguments": {}}

        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})
        action_type = arguments.get("action", "wait")

        # Map Qwen action format to webgym action format
        if self.interaction_mode == "coordinates":
            return self._extract_coordinates_action(action_type, arguments)
        else:
            return self._extract_set_of_marks_action(action_type, arguments)

    def _extract_coordinates_action(self, action_type: str, arguments: Dict) -> Dict[str, Any]:
        """Extract action for coordinates mode"""
        if action_type == "left_click":
            return {
                "key": "click",
                "arguments": {
                    "coordinates": arguments.get("coordinate", [500, 500])
                }
            }
        elif action_type == "type":
            return {
                "key": "type",
                "arguments": {
                    "coordinates": arguments.get("coordinate", [500, 500]),
                    "text": arguments.get("text", "")
                }
            }
        elif action_type == "scroll":
            direction = arguments.get("direction", "down")
            return {
                "key": "scroll",
                "arguments": {
                    "direction": direction
                }
            }
        elif action_type == "answer":
            return {
                "key": "answer",
                "arguments": {
                    "content": arguments.get("text", "")
                }
            }
        elif action_type == "wait":
            return {
                "key": "wait",
                "arguments": {
                    "time": arguments.get("time", 2.0)  # Default to 2 seconds if not provided
                }
            }
        elif action_type == "go_back":
            # Always go back to last page (homepage option removed - agent can use navigate for that)
            return {
                "key": "goback",
                "arguments": {
                    "goback_to": "last_page"
                }
            }
        elif action_type == "navigate":
            url = arguments.get("url", "")
            # Ensure URL starts with https://
            if url and not url.startswith("https://"):
                url = "https://" + url
            return {
                "key": "navigate",
                "arguments": {
                    "url": url
                }
            }

        return {"key": "wait", "arguments": {}}

    def _extract_set_of_marks_action(self, action_type: str, arguments: Dict) -> Dict[str, Any]:
        """Extract action for set_of_marks mode"""
        if action_type == "click":
            return {
                "key": "click",
                "arguments": {
                    "element_id": arguments.get("element_id", "1")
                }
            }
        elif action_type == "hover":
            return {
                "key": "hover",
                "arguments": {
                    "element_id": arguments.get("element_id", "1")
                }
            }
        elif action_type == "type":
            return {
                "key": "type",
                "arguments": {
                    "element_id": arguments.get("element_id", "1"),
                    "content": arguments.get("text", "")
                }
            }
        elif action_type == "scroll":
            direction = arguments.get("direction", "down")
            return {
                "key": "scroll",
                "arguments": {
                    "direction": direction
                }
            }
        elif action_type == "answer":
            return {
                "key": "answer",
                "arguments": {
                    "content": arguments.get("text", "")
                }
            }
        elif action_type == "wait":
            return {
                "key": "wait",
                "arguments": {
                    "time": arguments.get("time", 2.0)  # Default to 2 seconds if not provided
                }
            }
        elif action_type == "go_back":
            # Always go back to last page (homepage option removed - agent can use navigate for that)
            return {
                "key": "goback",
                "arguments": {
                    "goback_to": "last_page"
                }
            }
        elif action_type == "navigate":
            url = arguments.get("url", "")
            # Ensure URL starts with https://
            if url and not url.startswith("https://"):
                url = "https://" + url
            return {
                "key": "navigate",
                "arguments": {
                    "url": url
                }
            }

        return {"key": "wait", "arguments": {}}

    def supports_native_thinking(self) -> bool:
        """Qwen-Thinking variant has native <think> token support"""
        return self.variant == 'thinking'
