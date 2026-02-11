# webgym/context/parsers/set_of_marks_parser.py
from .base_parser import BaseResponseParser
from typing import Dict, Any
import re

class SetOfMarksResponseParser(BaseResponseParser):
    """Parser for set-of-marks interaction mode responses"""
    
    def _looks_like_action(self, line: str) -> bool:
        """Check if line looks like a set-of-marks action command"""
        # Clean the line first to handle prefixes and markup
        cleaned_line = self._clean_action_line(line)
        
        action_patterns = [
            r'Click\s*\[\d+\]',
            r'Type\s*\[\d+\]',
            r'Scroll\s*\[',
            r'Hover\s*\[\d+\]',
            r'Wait\s*$',
            r'GoBack\s*$',
            r'TabAndEnter\s*$',
            r'ANSWER\s*\['
        ]
        
        # Use re.search instead of re.match to find pattern anywhere in cleaned line
        return any(re.search(pattern, cleaned_line, re.IGNORECASE) for pattern in action_patterns)
    
    def _clean_action_line(self, line: str) -> str:
        """Clean action line by removing prefixes and markup"""
        if not line:
            return ""
        
        cleaned = line.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            r'^Action:\s*',
            r'^Next:\s*',
            r'^Next action:\s*',
            r'^I will\s*',
            r'^I should\s*',
            r'^My action:\s*',
            r'^The action is:\s*'
        ]
        
        for prefix in prefixes_to_remove:
            cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE).strip()
        
        # Remove box markup like <| begin_of_box |> and < | end_of_box | >
        cleaned = re.sub(r'<\s*\|\s*begin_of_box\s*\|\s*>', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'<\s*\|\s*end_of_box\s*\|\s*>', '', cleaned, flags=re.IGNORECASE)
        
        # Remove any remaining angle bracket markup
        cleaned = re.sub(r'<[^>]*>', '', cleaned)
        
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned

    def _parse_action_string(self, action_string: str) -> Dict[str, Any]:
        """Parse set-of-marks action string"""
        if not action_string:
            return {"key": "wait", "arguments": {}}
        
        action_string = action_string.strip()
        
        # Additional cleaning in case anything slipped through
        action_string = self._clean_action_line(action_string)
        
        # Define patterns for set-of-marks actions
        patterns = [
            # Click [numerical_label]
            (r'Click\s*\[(\d+)\]', "click", lambda m: {"element_id": m.group(1)}),
            
            # Hover [numerical_label]
            (r'Hover\s*\[(\d+)\]', "hover", lambda m: {"element_id": m.group(1)}),
            
            # Type [numerical_label] content or Type [numerical_label] [content]
            (r'Type\s*\[(\d+)\]\s*\[([^\]]*)\]', "type", lambda m: {"element_id": m.group(1), "content": m.group(2)}),
            (r'Type\s*\[(\d+)\]\s*([^\[]+)', "type", lambda m: {"element_id": m.group(1), "content": m.group(2).strip()}),
            
            # Scroll [numerical_label/WINDOW] [up/down]
            (r'Scroll\s*\[(\d+|WINDOW)\]\s*\[?(up|down)\]?', "scroll", lambda m: {
                "target": m.group(1), 
                "direction": m.group(2) if m.group(2) else "down"
            }),
            
            # Scroll without target (just direction)
            (r'Scroll\s*\[?(up|down)\]?', "scroll", lambda m: {
                "direction": m.group(1) if m.group(1) else "down"
            }),
            
            # ANSWER [content]
            (r'ANSWER\s*\[([^\]]*)\]', "answer", lambda m: {"content": m.group(1)}),

            # Simple actions
            (r'Wait$', "wait", lambda m: {}),
            (r'GoBack$', "goback", lambda m: {}),
            (r'Go\s*Back$', "goback", lambda m: {}),
            (r'TabAndEnter$', "tabandenter", lambda m: {}),
            (r'Tab\s*And\s*Enter$', "tabandenter", lambda m: {})
        ]
        
        # Try each pattern
        for pattern, action_key, arg_extractor in patterns:
            match = re.search(pattern, action_string, re.IGNORECASE)
            if match:
                try:
                    arguments = arg_extractor(match)
                    return {"key": action_key, "arguments": arguments}
                except Exception:
                    return {"key": action_key, "arguments": {}}
        
        # Special case: if the string contains just a number, assume it's a click
        number_match = re.match(r'^(\d+)$', action_string.strip())
        if number_match:
            return {"key": "click", "arguments": {"element_id": number_match.group(1)}}
        
        # Fallback parsing for partial matches
        if re.search(r"click", action_string, re.IGNORECASE):
            num_match = re.search(r"(\d+)", action_string)
            if num_match:
                return {"key": "click", "arguments": {"element_id": num_match.group(1)}}
            return {"key": "click", "arguments": {"element_id": "1"}}
        
        if re.search(r"type|input|fill", action_string, re.IGNORECASE):
            num_match = re.search(r"(\d+)", action_string)
            text_match = re.search(r'"([^"]*)"', action_string)
            element_id = num_match.group(1) if num_match else "1"
            content = text_match.group(1) if text_match else ""
            return {"key": "type", "arguments": {"element_id": element_id, "content": content}}
        
        if re.search(r"scroll", action_string, re.IGNORECASE):
            direction = "down"
            if re.search(r"up", action_string, re.IGNORECASE):
                direction = "up"
            return {"key": "scroll", "arguments": {"direction": direction}}

        if re.search(r"answer", action_string, re.IGNORECASE):
            content_match = re.search(r"answer[:\s]*(.*)", action_string, re.IGNORECASE)
            content = content_match.group(1).strip() if content_match else "Task completed"
            return {"key": "answer", "arguments": {"content": content}}

        # No pattern matches - default to wait
        return {"key": "wait", "arguments": {}}