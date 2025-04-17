import cv2
import numpy as np

def create_canvas(height, width):
    """Create a blank white canvas"""
    return np.ones((height, width), dtype=np.uint8) * 255

def draw_stroke(canvas, point1, point2, thickness=10):
    """
    Draw a line between two points on the canvas
    
    Args:
        canvas: The canvas to draw on
        point1: First point (x, y)
        point2: Second point (x, y)
        thickness: Line thickness
        
    Returns:
        Updated canvas with stroke
    """
    if point1 is None or point2 is None:
        return canvas
    cv2.line(canvas, point1, point2, 0, thickness)
    return canvas

def draw_strokes(canvas, points, thickness=10):
    """
    Draw the accumulated strokes on the canvas
    
    Args:
        canvas: The canvas to draw on
        points: List of points with None separating strokes
        thickness: Line thickness
        
    Returns:
        Updated canvas with strokes
    """
    if not points:
        return canvas
        
    # Create a copy of the canvas
    result = canvas.copy()
    
    # Draw each stroke
    current_stroke = []
    for pt in points:
        if pt is None:
            # End of stroke
            if len(current_stroke) > 1:
                for i in range(1, len(current_stroke)):
                    cv2.line(result, current_stroke[i-1], current_stroke[i], 0, thickness)
            current_stroke = []
        else:
            current_stroke.append(pt)
            
    # Draw the last stroke if exists
    if len(current_stroke) > 1:
        for i in range(1, len(current_stroke)):
            cv2.line(result, current_stroke[i-1], current_stroke[i], 0, thickness)
            
    return result

def overlay_canvas(frame, canvas, alpha=0.3):
    """
    Overlay the drawing canvas on the frame
    
    Args:
        frame: The camera frame
        canvas: The drawing canvas
        alpha: Transparency factor (0-1)
        
    Returns:
        Combined frame with canvas overlay
    """
    # Convert canvas to BGR for overlay
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    
    # Create an alpha blend
    overlay = cv2.addWeighted(frame, 1-alpha, canvas_bgr, alpha, 0)
    
    return overlay

def process_canvas_for_segmentation(canvas, points):
    """
    Process the canvas for digit/symbol segmentation
    
    Args:
        canvas: The drawing canvas
        points: List of points with None separating strokes
        
    Returns:
        List of segmented elements as tuples (position, processed_img)
    """
    if not points:
        return []
        
    # Group points by strokes
    strokes = []
    current_stroke = []
    
    for pt in points:
        if pt is None:
            if current_stroke:
                strokes.append(current_stroke)
                current_stroke = []
        else:
            current_stroke.append(pt)
            
    if current_stroke:
        strokes.append(current_stroke)
        
    # Process each stroke to identify symbols and digits
    segments = []
    
    for stroke in strokes:
        if len(stroke) < 5:  # Skip very short strokes (noise)
            continue
            
        # Get bounding box
        x_vals = [p[0] for p in stroke]
        y_vals = [p[1] for p in stroke]
        
        padding = 10
        xmin = max(0, min(x_vals) - padding)
        xmax = min(canvas.shape[1] - 1, max(x_vals) + padding)
        ymin = max(0, min(y_vals) - padding)
        ymax = min(canvas.shape[0] - 1, max(y_vals) + padding)
        
        # Skip invalid bounding boxes
        if xmin >= xmax or ymin >= ymax:
            continue
            
        # Extract the region
        region = canvas[ymin:ymax, xmin:xmax]
        
        # Process the region
        processed = process_region_for_prediction(region)
        
        # Add to segments with position info
        segments.append((xmin, processed))
        
    # Sort by x-position (left to right)
    segments.sort(key=lambda x: x[0])
    
    return segments

def process_region_for_prediction(region):
    """
    Process a region for model prediction
    
    Args:
        region: The cropped region from canvas
        
    Returns:
        Processed image ready for model prediction
    """
    if region.size == 0:
        return None
        
    # Get dimensions
    h, w = region.shape
    
    # Make it square with white background
    size = max(h, w)
    squared = np.ones((size, size), dtype=np.uint8) * 255
    
    # Center the content
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    squared[y_offset:y_offset+h, x_offset:x_offset+w] = region
    
    # Resize to 28x28
    resized = cv2.resize(squared, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize
    normalized = resized.astype("float32") / 255.0
    
    # Reshape for model input (add batch and channel dimensions)
    tensor = normalized.reshape(1, 28, 28, 1)
    
    return tensor