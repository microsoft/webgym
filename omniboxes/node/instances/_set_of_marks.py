import io
import random
from typing import BinaryIO, Dict, List, Tuple, cast

from PIL import Image, ImageDraw, ImageFont

from ._types import DOMRectangle, InteractiveRegion

"""
This module provides functionality to annotate screenshots with numbered markers for interactive regions.
It handles marking visible elements as well as tracking elements that are above or below the viewport.
"""

TOP_NO_LABEL_ZONE = 20  # Don't print any labels close the top of the page


def add_set_of_mark(
    screenshot: bytes | Image.Image | io.BufferedIOBase,
    ROIs: Dict[str, InteractiveRegion],
    use_sequential_ids: bool = False,
) -> Tuple[Image.Image, List[str], List[str], List[str], Dict[str, str]]:
    """
    Add numbered markers to a screenshot for each interactive region.

    Args:
        screenshot (bytes | Image.Image | io.BufferedIOBase): The screenshot image as bytes, PIL Image, or file-like object
        ROIs (Dict[str, InteractiveRegion]): Dictionary mapping element IDs to their interactive regions
        use_sequential_ids (bool): If True, assigns sequential numbers to elements instead of using original IDs

    Returns:
        Tuple containing:
        - Image.Image: Annotated image
        - List[str]: List of visible element IDs
        - List[str]: List of element IDs above viewport
        - List[str]: List of element IDs below viewport
        - Dict[str, str]: Mapping of displayed IDs to original element IDs
    """
    if isinstance(screenshot, Image.Image):
        return _add_set_of_mark(screenshot, ROIs, use_sequential_ids)

    if isinstance(screenshot, bytes):
        screenshot = io.BytesIO(screenshot)

    image = Image.open(cast(BinaryIO, screenshot))
    comp, visible_rects, rects_above, rects_below, id_mapping = _add_set_of_mark(image, ROIs, use_sequential_ids)
    image.close()
    return comp, visible_rects, rects_above, rects_below, id_mapping


def _add_set_of_mark(
    screenshot: Image.Image,
    ROIs: Dict[str, InteractiveRegion],
    use_sequential_ids: bool = False,
) -> Tuple[Image.Image, List[str], List[str], List[str], Dict[str, str]]:
    """
    Internal implementation for adding markers to the screenshot.

    Args:
        screenshot (Image.Image): PIL Image to annotate
        ROIs (Dict[str, InteractiveRegion]): Dictionary of interactive regions
        use_sequential_ids (bool): Whether to use sequential numbers instead of original IDs

    Returns:
        Same as :func:`add_set_of_mark`
    """
    visible_rects: List[str] = []
    rects_above: List[str] = []  # Scroll up to see
    rects_below: List[str] = []  # Scroll down to see
    id_mapping: Dict[str, str] = {}  # Maps new IDs to original IDs

    base = screenshot.convert("RGBA")
    
    # Calculate screen area for filtering
    screen_width, screen_height = base.size
    screen_area = screen_width * screen_height
    area_threshold = screen_area * 0.9  # 90% of screen area

    # Process all elements to categorize by viewport position
    for original_id, roi in ROIs.items():
        # Handle options separately and add to visible only (special case)
        if roi.get("tag_name") == "option" or roi.get("tag_name") == "input, type=file":
            if original_id not in visible_rects:
                visible_rects.append(original_id)
            continue

        # Check each rectangle for the element
        element_added = False  # Track if this element has been added to any category
        for rect in roi["rects"]:
            if not rect or rect["width"] * rect["height"] == 0:
                continue

            # Calculate rect area and filter out large elements
            rect_area = rect["width"] * rect["height"]
            if rect_area > area_threshold:
                # Skip this rect if it's larger than 90% of screen area
                continue

            mid = (
                (rect["right"] + rect["left"]) / 2.0,
                (rect["top"] + rect["bottom"]) / 2.0,
            )

            # Only process if x coordinate is valid
            if 0 <= mid[0] < base.size[0] and not element_added:
                # Add to exactly one list based on y coordinate
                if mid[1] < 0 and original_id not in rects_above:
                    rects_above.append(original_id)
                    element_added = True
                elif mid[1] >= base.size[1] and original_id not in rects_below:
                    rects_below.append(original_id)
                    element_added = True
                elif 0 <= mid[1] < base.size[1] and original_id not in visible_rects:
                    visible_rects.append(original_id)
                    element_added = True

    # Create ID mappings
    original_to_new: Dict[str, str] = {}  # Maps original IDs to new IDs for quick lookup

    if use_sequential_ids:
        # Only assign sequential IDs to visible elements that will actually be drawn
        next_id = 1
        new_visible_rects = []
        for original_id in visible_rects:
            new_id = str(next_id)
            id_mapping[new_id] = original_id
            original_to_new[original_id] = new_id
            new_visible_rects.append(new_id)
            next_id += 1
        
        # For non-visible elements, keep original IDs but don't assign sequential numbers
        new_rects_above = rects_above.copy()
        new_rects_below = rects_below.copy()
        
        # Add identity mappings for non-visible elements (they keep original IDs)
        for original_id in rects_above + rects_below:
            id_mapping[original_id] = original_id
            original_to_new[original_id] = original_id
    else:
        # Use original IDs for all elements
        new_visible_rects = visible_rects.copy()
        new_rects_above = rects_above.copy()
        new_rects_below = rects_below.copy()
        
        # Create identity mapping for all IDs
        for id_list in [visible_rects, rects_above, rects_below]:
            for original_id in id_list:
                id_mapping[original_id] = original_id
                original_to_new[original_id] = original_id

    # Drawing code - TWO-PHASE APPROACH for higher label priority
    fnt = ImageFont.load_default(14)
    overlay = Image.new("RGBA", base.size)
    draw = ImageDraw.Draw(overlay)

    # Collect visible elements and their drawing data
    elements_to_draw = []
    for original_id in visible_rects:
        roi = ROIs.get(original_id)
        if not roi or roi.get("tag_name") == "option":
            continue

        new_id = original_to_new.get(original_id)
        if new_id is None:
            continue  # Skip if no mapping found

        for rect in roi["rects"]:
            if not rect or rect["width"] * rect["height"] == 0:
                continue

            # Apply the same area filter for drawing
            rect_area = rect["width"] * rect["height"]
            if rect_area > area_threshold:
                continue

            mid = (
                (rect["right"] + rect["left"]) / 2.0,
                (rect["top"] + rect["bottom"]) / 2.0,
            )

            # Double-check that the element is actually visible in viewport
            if 0 <= mid[0] < base.size[0] and 0 <= mid[1] < base.size[1]:
                elements_to_draw.append((new_id, rect))

    # PHASE 1: Draw all rectangle borders first
    for new_id, rect in elements_to_draw:
        _draw_roi_border(draw, new_id, rect)

    # PHASE 2: Draw all labels on top (highest priority)
    for new_id, rect in elements_to_draw:
        _draw_roi_label(draw, new_id, fnt, rect)

    comp = Image.alpha_composite(base, overlay)
    overlay.close()

    return comp, new_visible_rects, new_rects_above, new_rects_below, id_mapping
    

def _color(identifier: int) -> Tuple[int, int, int, int]:
    """Generate a random color based on the identifier."""
    rnd = random.Random(int(identifier))
    color = [rnd.randint(0, 255), rnd.randint(125, 255), rnd.randint(0, 50)]
    rnd.shuffle(color)
    color.append(255)
    return cast(Tuple[int, int, int, int], tuple(color))


def _draw_roi_border(
    draw: ImageDraw.ImageDraw,
    idx: str | int,
    rect: DOMRectangle,
) -> None:
    """
    Draw only the border of a region of interest.

    Args:
        draw (ImageDraw.ImageDraw): PIL ImageDraw object
        idx (str | int): Index/ID for color generation
        rect (DOMRectangle): Rectangle coordinates for the region
    """
    color = _color(int(idx))
    roi = ((rect["left"], rect["top"]), (rect["right"], rect["bottom"]))
    
    # Draw rectangle with colored border and semi-transparent fill
    draw.rectangle(roi, outline=color, fill=(color[0], color[1], color[2], 48), width=2)


def _draw_roi_label(
    draw: ImageDraw.ImageDraw,
    idx: str | int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    rect: DOMRectangle,
) -> None:
    """
    Draw only the label of a region of interest.

    Args:
        draw (ImageDraw.ImageDraw): PIL ImageDraw object
        idx (str | int): Index/ID to display on the marker
        font (ImageFont.FreeTypeFont | ImageFont.ImageFont): Font to use for the marker text
        rect (DOMRectangle): Rectangle coordinates for the region
    """
    color = _color(int(idx))
    luminance = color[0] * 0.3 + color[1] * 0.59 + color[2] * 0.11
    text_color = (0, 0, 0, 255) if luminance > 90 else (255, 255, 255, 255)

    # Adjust label position if too close to top of screen
    label_location = (rect["right"], rect["top"])
    label_anchor = "rb"

    if label_location[1] <= TOP_NO_LABEL_ZONE:
        label_location = (rect["right"], rect["bottom"])
        label_anchor = "rt"

    bbox = draw.textbbox(label_location, str(idx), font=font, anchor=label_anchor, align="center")
    bbox = (bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3)
    draw.rectangle(bbox, fill=color)

    draw.text(
        label_location,
        str(idx),
        fill=text_color,
        font=font,
        anchor=label_anchor,
        align="center",
    )


def _draw_roi(
    draw: ImageDraw.ImageDraw,
    idx: str | int,  # Fix type hint to allow both string and int indices
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    rect: DOMRectangle,
) -> None:
    """
    Draw a single region of interest on the image.
    
    NOTE: This function is kept for backward compatibility but is no longer used
    in the main drawing logic. The drawing is now split into _draw_roi_border
    and _draw_roi_label for better layering control.

    Args:
        draw (ImageDraw.ImageDraw): PIL ImageDraw object
        idx (str | int): Index/ID to display on the marker
        font (ImageFont.FreeTypeFont | ImageFont.ImageFont): Font to use for the marker text
        rect (DOMRectangle): Rectangle coordinates for the region
    """
    _draw_roi_border(draw, idx, rect)
    _draw_roi_label(draw, idx, font, rect)