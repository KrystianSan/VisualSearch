"""
utils/image_utils.py
Image loading and display helpers.
PIL / OpenCV utilities with no direct GUI/widget dependencies.
"""

import cv2
import logging
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)

from PIL import Image, ImageTk


def load_pil_image(path) -> Image.Image | None:
    """Open an image with Pillow; return None on failure."""
    try:
        return Image.open(path)
    except Exception as exc:
        log.warning("load_pil_image: cannot open %s: %s", path, exc)
        return None


def load_cv2_image(path) -> np.ndarray | None:
    """Read an image with OpenCV (handles Unicode paths)."""
    try:
        img_bytes = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        log.warning("load_cv2_image: cannot read %s: %s", path, exc)
        return None


def fit_image_to_canvas(
    image: Image.Image,
    canvas_width: int,
    canvas_height: int,
) -> tuple[ImageTk.PhotoImage, int, int]:
    """
    Resize *image* to fit within (canvas_width × canvas_height) while
    preserving aspect ratio.

    Returns
    -------
    (photo_image, x_offset, y_offset)
        Ready to be placed at (x_offset, y_offset) with anchor=NW.
    """
    orig_w, orig_h = image.size
    if orig_w == 0 or orig_h == 0:
        raise ValueError("Image has zero dimension.")

    ratio = min(canvas_width / orig_w, canvas_height / orig_h)
    new_w = max(1, int(orig_w * ratio))
    new_h = max(1, int(orig_h * ratio))

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(resized)

    x = (canvas_width - new_w) // 2
    y = (canvas_height - new_h) // 2
    return photo, x, y


def get_image_dimensions(path) -> tuple[int, int] | None:
    """Return (width, height) of an image without fully decoding it."""
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return None
