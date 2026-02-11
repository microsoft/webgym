"""
Miscellaneous Utility Functions
"""
import click
import warnings
from PIL import Image
import random

def colorful_print(string: str, *args, **kwargs) -> None:
    print(click.style(string, *args, **kwargs))

def is_white_image(image, white_threshold=250, white_percentage=1.0):
    """
    Check if image is entirely or predominantly white by analyzing all pixels

    Args:
        image: PIL Image object
        white_threshold: RGB threshold above which a pixel is considered "white" (default: 250)
        white_percentage: Percentage of pixels that must be white (default: 1.0 for strict 100% check)

    Returns:
        bool: True if the specified percentage of pixels are white
    """
    import numpy as np

    # Convert to numpy array for efficient processing
    img_array = np.array(image)

    # For grayscale images, add a dimension
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=2)

    # For strict 100% check, use optimized all() which can short-circuit
    if white_percentage >= 1.0:
        if img_array.shape[2] == 1:  # Grayscale
            return np.all(img_array[:, :, 0] > white_threshold)
        else:  # RGB/RGBA
            return np.all(img_array[:, :, :3] > white_threshold)

    # For non-strict checks, count white pixels
    if img_array.shape[2] == 1:  # Grayscale
        white_pixels = img_array[:, :, 0] > white_threshold
    else:  # RGB/RGBA
        white_pixels = np.all(img_array[:, :, :3] > white_threshold, axis=2)

    white_ratio = np.sum(white_pixels) / white_pixels.size

    return white_ratio >= white_percentage
