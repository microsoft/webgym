# webgym/context/parsers/coordinates_parser.py
from .base_parser import BaseResponseParser
from typing import Dict, Any
import re

class CoordinatesResponseParser(BaseResponseParser):
    """Parser for coordinates interaction mode responses"""

    def _looks_like_action(self, line: str) -> bool:
        """Check if line looks like a coordinates action command"""
        action_patterns = [
            r'^Click\s*\[\d+,\s*\d+\]',
            r'^Type\s*\[\d+,\s*\d+\]',
            r'^Scroll\s*\[',
            r'^Hover\s*\[\d+,\s*\d+\]',
            r'^HoverAndScroll\s*\[\d+,\s*\d+\]',
            r'^Wait\s*$',
            r'^GoBack\s*$',
            r'^TabAndEnter\s*$',
            r'^ANSWER\s*\['
        ]
        return any(re.match(pattern, line, re.IGNORECASE) for pattern in action_patterns)

    def _parse_action_string(self, action_string: str) -> Dict[str, Any]:
        """Parse coordinates action string"""
        if not action_string:
            return {"key": "wait", "arguments": {}}

        action_string = action_string.strip()

        # Helper to extract [x, y] coordinates
        def _swap_groups_to_xy(m):
            # Input provides [x, y]; return [x, y]
            return [int(m.group(2)), int(m.group(1))] 

        patterns = [
            # Click [x, y]
            (r'^Click\s*\[(\d+),\s*(\d+)\]', "click", lambda m: {
                "coordinates": _swap_groups_to_xy(m)
            }),

            # Hover [x, y]
            (r'^Hover\s*\[(\d+),\s*(\d+)\]', "hover", lambda m: {
                "coordinates": _swap_groups_to_xy(m)
            }),

            # HoverAndScroll [x, y] [up/down]
            (r'^HoverAndScroll\s*\[(\d+),\s*(\d+)\]\s*\[?(up|down)\]?', "hover_and_scroll", lambda m: {
                "coordinates": _swap_groups_to_xy(m),
                "direction": m.group(3) if m.group(3) else "down"
            }),

            # Type [x, y] [content]
            (r'^Type\s*\[(\d+),\s*(\d+)\]\s*\[([^\]]*)\]', "type", lambda m: {
                "coordinates": _swap_groups_to_xy(m),
                "text": m.group(3)
            }),

            # Scroll [x, y] [up/down]
            (r'^Scroll\s*\[(\d+),\s*(\d+)\]\s*\[?(up|down)\]?', "scroll", lambda m: {
                "coordinates": _swap_groups_to_xy(m),
                "direction": m.group(3) if m.group(3) else "down"
            }),

            # Scroll [WINDOW] [up/down]
            (r'^Scroll\s*\[WINDOW\]\s*\[?(up|down)\]?', "scroll", lambda m: {
                "direction": m.group(1) if m.group(1) else "down"
            }),

            # Scroll without coordinates (just direction)
            (r'^Scroll\s*\[?(up|down)\]?', "scroll", lambda m: {
                "direction": m.group(1) if m.group(1) else "down"
            }),

            # ANSWER [content]
            (r'^ANSWER\s*\[([^\]]*)\]', "answer", lambda m: {"content": m.group(1)}),

            # Simple actions
            (r'^Wait$', "wait", lambda m: {}),
            (r'^GoBack$', "goback", lambda m: {}),
            (r'^Go\s*Back$', "goback", lambda m: {}),
            (r'^TabAndEnter$', "tabandenter", lambda m: {}),
            (r'^Tab\s*And\s*Enter$', "tabandenter", lambda m: {}),
        ]

        # Try each pattern
        for pattern, action_key, arg_extractor in patterns:
            match = re.match(pattern, action_string, re.IGNORECASE)
            if match:
                try:
                    arguments = arg_extractor(match)
                    return {"key": action_key, "arguments": arguments}
                except Exception:
                    return {"key": action_key, "arguments": {}}

        # Fallback parsing for partial matches
        if re.search(r"click", action_string, re.IGNORECASE):
            coord_match = re.search(r'\[(\d+),\s*(\d+)\]', action_string)
            if coord_match:
                # Model outputs [x, y], no swap needed
                x, y = int(coord_match.group(1)), int(coord_match.group(2))
                return {"key": "click", "arguments": {"coordinates": [x, y]}}
            else:
                return {"key": "click", "arguments": {"coordinates": [500, 500]}}

        if re.search(r"type|input|fill", action_string, re.IGNORECASE):
            coord_match = re.search(r'\[(\d+),\s*(\d+)\]', action_string)
            text_match = re.search(r'"([^"]*)"', action_string)

            if coord_match:
                # Model outputs [x, y], no swap needed
                coordinates = [int(coord_match.group(1)), int(coord_match.group(2))]
            else:
                coordinates = [500, 500]
            text = text_match.group(1) if text_match else ""

            return {"key": "type", "arguments": {"coordinates": coordinates, "text": text}}

        if re.search(r"scroll", action_string, re.IGNORECASE):
            direction = "down"
            if re.search(r"up", action_string, re.IGNORECASE):
                direction = "up"

            coord_match = re.search(r'\[(\d+),\s*(\d+)\]', action_string)
            if coord_match:
                # Model outputs [x, y], no swap needed
                coordinates = [int(coord_match.group(1)), int(coord_match.group(2))]
                return {"key": "scroll", "arguments": {"coordinates": coordinates, "direction": direction}}
            else:
                return {"key": "scroll", "arguments": {"direction": direction}}

        if re.search(r"answer", action_string, re.IGNORECASE):
            content_match = re.search(r"answer[:\s]*(.*)", action_string, re.IGNORECASE)
            content = content_match.group(1).strip() if content_match else "Task completed"
            return {"key": "answer", "arguments": {"content": content}}

        # No pattern matches - default to wait
        return {"key": "wait", "arguments": {}}
