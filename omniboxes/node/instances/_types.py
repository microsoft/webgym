from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Union

from PIL import Image as PILImage


@dataclass
class FunctionCall:
    """Represents a function call from an LLM agent."""
    id: str
    arguments: str  # JSON args
    name: str  # Function to call


@dataclass
class FunctionExecutionResult:
    """Represents the result of a function execution."""
    content: str
    call_id: str


class Image:
    """Image wrapper with base64 encoding support."""

    def __init__(self, image: PILImage.Image):
        self.image: PILImage.Image = image.convert("RGB")

    @classmethod
    def from_pil(cls, pil_image: PILImage.Image) -> Image:
        return cls(pil_image)

    @classmethod
    def from_base64(cls, base64_str: str) -> Image:
        return cls(PILImage.open(BytesIO(base64.b64decode(base64_str))))

    def to_base64(self) -> str:
        buffered = BytesIO()
        self.image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @classmethod
    def from_file(cls, file_path: Path) -> Image:
        return cls(PILImage.open(file_path))

    @property
    def data_uri(self) -> str:
        base64_image = self.to_base64()
        image_data = base64.b64decode(base64_image)
        if image_data.startswith(b"\xff\xd8\xff"):
            mime_type = "image/jpeg"
        elif image_data.startswith(b"\x89PNG\r\n\x1a\n"):
            mime_type = "image/png"
        else:
            mime_type = "image/jpeg"
        return f"data:{mime_type};base64,{base64_image}"


UserContent = Union[str, List[Union[str, Image]]]
AssistantContent = Union[str, List[FunctionCall]]
FunctionExecutionContent = List[FunctionExecutionResult]
SystemContent = str


class DOMRectangle(TypedDict):
    x: Union[int, float]
    y: Union[int, float]
    width: Union[int, float]
    height: Union[int, float]
    top: Union[int, float]
    right: Union[int, float]
    bottom: Union[int, float]
    left: Union[int, float]


class VisualViewport(TypedDict):
    height: Union[int, float]
    width: Union[int, float]
    offsetLeft: Union[int, float]
    offsetTop: Union[int, float]
    pageLeft: Union[int, float]
    pageTop: Union[int, float]
    scale: Union[int, float]
    clientWidth: Union[int, float]
    clientHeight: Union[int, float]
    scrollWidth: Union[int, float]
    scrollHeight: Union[int, float]


class InteractiveRegion(TypedDict):
    tag_name: str
    role: str
    aria_name: str
    v_scrollable: bool
    rects: List[DOMRectangle]


# Helper functions for dealing with JSON. Not sure there's a better way?


def _get_str(d: Any, k: str) -> str:
    val = d[k]
    assert isinstance(val, str)
    return val


def _get_number(d: Any, k: str) -> Union[int, float]:
    val = d[k]
    assert isinstance(val, int) or isinstance(val, float)
    return val


def _get_bool(d: Any, k: str) -> bool:
    val = d[k]
    assert isinstance(val, bool)
    return val


def domrectangle_from_dict(rect: Dict[str, Any]) -> DOMRectangle:
    return DOMRectangle(
        x=_get_number(rect, "x"),
        y=_get_number(rect, "y"),
        width=_get_number(rect, "width"),
        height=_get_number(rect, "height"),
        top=_get_number(rect, "top"),
        right=_get_number(rect, "right"),
        bottom=_get_number(rect, "bottom"),
        left=_get_number(rect, "left"),
    )


def interactiveregion_from_dict(region: Dict[str, Any]) -> InteractiveRegion:
    typed_rects: List[DOMRectangle] = []
    for rect in region["rects"]:
        typed_rects.append(domrectangle_from_dict(rect))

    return InteractiveRegion(
        tag_name=_get_str(region, "tag_name"),
        role=_get_str(region, "role"),
        aria_name=_get_str(region, "aria-name"),
        v_scrollable=_get_bool(region, "v-scrollable"),
        rects=typed_rects,
    )


def visualviewport_from_dict(viewport: Dict[str, Any]) -> VisualViewport:
    return VisualViewport(
        height=_get_number(viewport, "height"),
        width=_get_number(viewport, "width"),
        offsetLeft=_get_number(viewport, "offsetLeft"),
        offsetTop=_get_number(viewport, "offsetTop"),
        pageLeft=_get_number(viewport, "pageLeft"),
        pageTop=_get_number(viewport, "pageTop"),
        scale=_get_number(viewport, "scale"),
        clientWidth=_get_number(viewport, "clientWidth"),
        clientHeight=_get_number(viewport, "clientHeight"),
        scrollWidth=_get_number(viewport, "scrollWidth"),
        scrollHeight=_get_number(viewport, "scrollHeight"),
    )