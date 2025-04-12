#Drawing helper
import cv2


def draw_path(points, frame):
    for i in range(1, len(points)):
        cv2.line(frame, points[i-1], points[i], (255, 0, 0), 4)
    return frame