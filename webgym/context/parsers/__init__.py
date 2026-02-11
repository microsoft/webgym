# webgym/context/parsers/__init__.py
from .base_parser import BaseResponseParser
from .set_of_marks_parser import SetOfMarksResponseParser
from .coordinates_parser import CoordinatesResponseParser

def create_parser(interaction_mode: str) -> BaseResponseParser:
    """Create parser based on interaction mode"""
    if interaction_mode == 'set_of_marks':
        return SetOfMarksResponseParser()
    elif interaction_mode == 'coordinates':
        return CoordinatesResponseParser()
    else:
        raise ValueError(f"Unknown interaction mode: {interaction_mode}")

__all__ = [
    'BaseResponseParser',
    'SetOfMarksResponseParser',
    'CoordinatesResponseParser',
    'create_parser'
]