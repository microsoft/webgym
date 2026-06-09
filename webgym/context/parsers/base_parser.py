# webgym/context/parsers/base_parser.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import re

class BaseResponseParser(ABC):
    """Base class for response parsers"""

    def _extract_observation_section(self, response: str) -> str:
        """Extract observation section between markers"""
        pattern = r'===Observation summarization===(.*?)===Reasoning==='
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback: try to find observation content at the beginning
        lines = response.split('\n')
        observation_lines = []
        in_observation = True
        
        for line in lines:
            line_stripped = line.strip()
            if '===Reasoning===' in line_stripped or line_stripped.startswith('Thoughts:'):
                break
            if in_observation:
                observation_lines.append(line)
        
        return '\n'.join(observation_lines).strip()
    
    def _extract_reasoning_section(self, response: str) -> str:
        """Extract reasoning section between markers"""
        pattern = r'===Reasoning===(.*?)Action:'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback: look for Thoughts: section
        pattern = r'Thoughts:(.*?)Action:'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return ""
    
    def _extract_action_string(self, response: str) -> str:
        """Extract action string after Action: marker"""
        pattern = r'Action:\s*(.*)$'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        if match:
            action_text = match.group(1).strip()
            # Get the first meaningful line as action
            lines = [line.strip() for line in action_text.split('\n') if line.strip()]
            return lines[0] if lines else ""
        return ""
    
    def _extract_field_from_section(self, section: str, field_name: str) -> str:
        """Extract a specific field from observation section"""
        patterns = [
            rf"^{field_name}:\s*(.*?)(?=^(?:Task|Title|URL|Date|Effect|Notes|Submit):|\Z)",
            rf"{field_name}\s*:\s*(.*?)(?=\n\s*(?:Task|Title|URL|Date|Effect|Notes|Submit)\s*:|\Z)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, section, re.DOTALL | re.IGNORECASE | re.MULTILINE)
            if match:
                content = match.group(1).strip()
                if content:
                    return content
        
        return ""
    
    def _extract_thoughts_from_reasoning(self, reasoning_section: str) -> str:
        """Extract thoughts from reasoning section"""
        # Look for content after "Thoughts:" marker
        if "Thoughts:" in reasoning_section:
            thoughts = reasoning_section.split("Thoughts:")[1].strip()
            return thoughts
        
        # If no "Thoughts:" marker, use the entire reasoning section
        return reasoning_section.strip()
    
    @abstractmethod
    def _looks_like_action(self, line: str) -> bool:
        """Check if line looks like an action command"""
        pass
    
    @abstractmethod
    def _parse_action_string(self, action_string: str) -> Dict[str, Any]:
        """Parse action string into structured format"""
        pass