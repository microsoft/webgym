from typing import Dict, Any, Tuple, Optional, List, Union
from .parsers import create_parser

class ContextManager:
    """
    Enhanced context manager supporting VLM models
    """

    def __init__(self, context_config: Dict[str, Any], model_config: Dict[str, Any], verbose: bool = True):
        """
        Initialize ContextManager with model-specific interface

        Args:
            context_config: Dict with 'interaction_mode' ('coordinates' or 'set_of_marks')
            model_config: Dict with 'model_type', 'variant', etc.
            verbose: Whether to print initialization info
        """
        self.interaction_mode = context_config.get('interaction_mode', 'set_of_marks')

        # Validate configuration
        if self.interaction_mode not in ['coordinates', 'set_of_marks']:
            raise ValueError(f"Invalid interaction_mode: {self.interaction_mode}")

        # Create model-specific interface
        from webgym.models.model_factory import create_model_interface
        self.model_interface = create_model_interface(model_config)

        # Parser for actions (based on interaction_mode, shared across models)
        self.parser = create_parser(interaction_mode=self.interaction_mode)

        if verbose:
            conv_type = self.model_interface.conversation_builder.get_conversation_type()
            model_type = model_config.get('model_type', 'unknown')
            print(f"ContextManager initialized:")
            print(f"  - Model type: {model_type}")
            print(f"  - Conversation type: {conv_type}")
            print(f"  - Interaction mode: {self.interaction_mode}")
            print(f"  - Native thinking: {self.model_interface.supports_native_thinking()}")

    def get_interaction_mode(self) -> str:
        """Get the interaction mode"""
        return self.interaction_mode

    def get_model_interface(self):
        """Get the model interface"""
        return self.model_interface

    def get_parser(self):
        """Get the response parser"""
        return self.parser

    def build_conversation(self, task: str, trajectory: List[Dict], current_observation: Dict, **kwargs) -> Union[List[Dict], Dict]:
        """
        Build model-specific conversation (single-turn or multi-turn)

        Args:
            task: Task description
            trajectory: List of trajectory steps
            current_observation: Current observation with image and metadata
            **kwargs: Additional parameters (e.g., website URL)

        Returns:
            List of message dicts for multi-turn conversation
        """
        return self.model_interface.build_conversation(
            task=task,
            trajectory=trajectory,
            current_observation=current_observation,
            **kwargs
        )

    def parse_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Parse model-specific response

        Args:
            raw_response: Raw string response from model

        Returns:
            Parsed response dict
        """
        return self.model_interface.parse_response(raw_response)

    def extract_action(self, parsed_response: Dict) -> Dict[str, Any]:
        """
        Extract action in standardized format

        Args:
            parsed_response: Parsed response dict

        Returns:
            Dict with 'key' and 'arguments'
        """
        return self.model_interface.extract_action(parsed_response)

    def parse_action_to_browser_command(self, action, screen_dimensions: Optional[Tuple[int, int]] = None, homepage_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert action to browser command based on interaction mode

        Args:
            action: Action object with .action dict
            screen_dimensions: Tuple of (width, height) in pixels
            homepage_url: Optional homepage URL for goback action

        Returns:
            Browser command dict
        """
        if self.interaction_mode == 'set_of_marks':
            return self._parse_set_of_marks_action(action, homepage_url)
        else:  # coordinates
            return self._parse_coordinates_action(action, screen_dimensions, homepage_url)

    def _parse_set_of_marks_action(self, action, homepage_url: Optional[str] = None) -> Dict[str, Any]:
        """Convert set-of-marks action to browser command"""
        action_key = action.action['key']
        action_args = action.action['arguments']

        if action_key == "click":
            element_id = action_args.get('element_id', '1')
            return {"click_id": {"id": str(element_id)}}
        elif action_key == "hover":
            element_id = action_args.get('element_id', '1')
            return {"hover_id": {"id": str(element_id)}}
        elif action_key == "type":
            element_id = action_args.get('element_id', '1')
            content = action_args.get('content', '')
            return {"fill_id": {
                "id": str(element_id),
                "value": content,
                "press_enter": True,
                "delete_existing": True
            }}
        elif action_key == "scroll":
            direction = action_args.get('direction', 'down').lower()
            if direction == "up":
                return {"page_up": {"full_page": True}}
            else:
                return {"page_down": {"full_page": True}}
        elif action_key == "goback":
            goback_to = action_args.get('goback_to', 'last_page')
            if goback_to == 'homepage' and homepage_url:
                return {"visit_page": {"url": homepage_url}}
            else:
                return {"back": {}}
        elif action_key == "navigate":
            url = action_args.get('url', '')
            if url:
                return {"visit_page": {"url": url}}
            else:
                return {"sleep": {"duration": 0.5}}
        elif action_key == "wait":
            duration = action_args.get("time", 2.0)  # Use provided time, default to 2.0
            return {"sleep": {"duration": duration}}
        elif action_key == "tabandenter":
            return {"tab_and_enter": {}}
        elif action_key == "answer":
            return {"sleep": {"duration": 0.1}}
        else:
            return {"sleep": {"duration": 0.5}}

    def _convert_normalized_coords(self, normalized_coords, screen_dimensions: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Convert normalized coordinates (0-1000 range) to actual screen coordinates.

        Args:
            normalized_coords: List/tuple of [x, y] coordinates in 0-1000 range
            screen_dimensions: Tuple of (width, height) in pixels

        Returns:
            Tuple of (actual_x, actual_y) in pixel coordinates
        """
        if screen_dimensions is None:
            print(f"Warning: No screen dimensions available, using normalized coordinates as-is")
            # Fallback: assume 1000x1000 screen for 1:1 mapping
            return (int(normalized_coords[0]), int(normalized_coords[1]))

        screen_width, screen_height = screen_dimensions

        if isinstance(normalized_coords, (list, tuple)) and len(normalized_coords) >= 2:
            if len(normalized_coords) == 4:
                # Handle bounding box format [x1, y1, x2, y2] - convert to center point
                norm_x = (normalized_coords[0] + normalized_coords[2]) // 2
                norm_y = (normalized_coords[1] + normalized_coords[3]) // 2
            else:
                norm_x, norm_y = normalized_coords[0], normalized_coords[1]

            # Convert from normalized (0-1000) to actual pixel coordinates using rounding
            actual_x = round(screen_width * norm_x / 1000.0)
            actual_y = round(screen_height * norm_y / 1000.0)

            # Clamp to screen bounds for safety (0 to screen_dimension-1 for valid pixel indices)
            actual_x = max(0, min(actual_x, screen_width - 1))
            actual_y = max(0, min(actual_y, screen_height - 1))

            # print(f"Coordinate conversion: normalized [{norm_x}, {norm_y}] -> actual [{actual_x}, {actual_y}] (screen: {screen_width}x{screen_height})")
            return (actual_x, actual_y)
        else:
            # Fallback for invalid input
            return (500, 500)

    def _parse_coordinates_action(self, action, screen_dimensions: Optional[Tuple[int, int]] = None, homepage_url: Optional[str] = None) -> Dict[str, Any]:
        """Convert coordinates action to browser command with normalization support"""
        action_key = action.action.get('key', 'wait')
        action_args = action.action.get('arguments', {})

        if action_key == "click":
            normalized_coordinates = action_args.get('coordinates', [500, 500])
            actual_x, actual_y = self._convert_normalized_coords(normalized_coordinates, screen_dimensions)

            return {"click_coords": {"x": actual_x, "y": actual_y}}

        elif action_key == "hover":
            normalized_coordinates = action_args.get('coordinates', [500, 500])
            actual_x, actual_y = self._convert_normalized_coords(normalized_coordinates, screen_dimensions)

            return {"hover_coords": {"x": actual_x, "y": actual_y}}

        elif action_key == "hover_and_scroll":
            normalized_coordinates = action_args.get('coordinates', [500, 500])
            actual_x, actual_y = self._convert_normalized_coords(normalized_coordinates, screen_dimensions)
            direction = action_args.get('direction', 'down')

            return {"hover_and_scroll_coords": {
                "x": actual_x,
                "y": actual_y,
                "direction": direction
            }}

        elif action_key == "type":
            text = action_args.get('text', '')
            normalized_coordinates = action_args.get('coordinates', [500, 500])
            actual_x, actual_y = self._convert_normalized_coords(normalized_coordinates, screen_dimensions)

            return {"fill_coords": {
                "x": actual_x,
                "y": actual_y,
                "value": text,
                "press_enter": True,
                "delete_existing": True
            }}

        elif action_key == "scroll":
            direction = action_args.get('direction', 'down')
            # For scroll actions, we check if coordinates are provided for element-specific scrolling
            if 'coordinates' in action_args:
                normalized_coordinates = action_args.get('coordinates', [500, 500])
                actual_x, actual_y = self._convert_normalized_coords(normalized_coordinates, screen_dimensions)
                return {"scroll_coords": {
                    "x": actual_x,
                    "y": actual_y,
                    "direction": direction
                }}
            else:
                # Window-level scrolling
                if direction == 'up':
                    return {"page_up": {"full_page": True}}
                else:
                    return {"page_down": {"full_page": True}}

        elif action_key == "wait":
            duration = action_args.get("time", 2.0)  # Use provided time, default to 2.0
            return {"sleep": {"duration": duration}}

        elif action_key == "goback":
            goback_to = action_args.get('goback_to', 'last_page')
            if goback_to == 'homepage' and homepage_url:
                return {"visit_page": {"url": homepage_url}}
            else:
                return {"back": {}}

        elif action_key == "navigate":
            url = action_args.get('url', '')
            if url:
                return {"visit_page": {"url": url}}
            else:
                return {"sleep": {"duration": 0.5}}

        elif action_key == "tabandenter":
            return {"tab_and_enter": {}}

        elif action_key == "answer":
            return {"sleep": {"duration": 0.1}}

        else:
            return {"sleep": {"duration": 0.5}}
