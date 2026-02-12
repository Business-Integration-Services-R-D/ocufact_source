from typing import List, Tuple
import cv2
import numpy as np


def bbox_intersection(box1: List[int], box2: List[int]) -> float:
    """Calculate intersection area between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection coordinates
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    # Check if there's no intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    return (x_right - x_left) * (y_bottom - y_top)


def bbox_area(box: List[int]) -> float:
    """Calculate area of a bounding box."""
    x_min, y_min, x_max, y_max = box
    return (x_max - x_min) * (y_max - y_min)


def bbox_overlap_ratio(box1: List[int], box2: List[int], threshold: float = 0.1) -> float:
    """Calculate overlap ratio between two bounding boxes."""
    intersection = bbox_intersection(box1, box2)
    if intersection == 0:
        return 0.0
    
    area1 = bbox_area(box1)
    area2 = bbox_area(box2)
    min_area = min(area1, area2)
    
    if min_area == 0:
        return 0.0
    
    return intersection / min_area


def boxes_overlap(box1: List[int], box2: List[int], threshold: float = 0.1) -> bool:
    """Check if two bounding boxes overlap above threshold."""
    return bbox_overlap_ratio(box1, box2) >= threshold


def normalize_coords(coords: List[int], img_width: int, img_height: int) -> List[int]:
    """Ensure coordinates are within image bounds."""
    x_min, y_min, x_max, y_max = coords
    x_min = max(0, min(x_min, img_width))
    y_min = max(0, min(y_min, img_height))
    x_max = max(0, min(x_max, img_width))
    y_max = max(0, min(y_max, img_height))
    return [x_min, y_min, x_max, y_max]


def draw_filled_rectangle(image: np.ndarray, coords: List[int], color: str, alpha: float = 0.5) -> np.ndarray:
    """Draw a filled rectangle with transparency on image."""
    # Convert hex color to BGR
    color_hex = color.lstrip('#')
    b = int(color_hex[4:6], 16)
    g = int(color_hex[2:4], 16)
    r = int(color_hex[0:2], 16)
    bgr_color = (b, g, r)
    
    # Create overlay
    overlay = image.copy()
    x_min, y_min, x_max, y_max = coords
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), bgr_color, -1)
    
    # Blend with original image
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    color_hex = hex_color.lstrip('#')
    return tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))


def batch_list(items: List, batch_size: int) -> List[List]:
    """Split list into batches of specified size."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

