import cv2
import os
import numpy as np

label = input("Enter label for gesture (0-9, +, -, *, /): ")

save_path = f"data/{label}/"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
drawing = False
points = []

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    for i in range(1, len(points)):
        cv2.line(frame, points[i-1], points[i], (0, 0, 255), 3)

    cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Draw Gesture", frame)

    key = cv2.waitKey(1)

    if key == ord('d'):
        drawing = True
        points = []
    elif key == ord('s') and points:
        # Save gesture image
        x, y, w, h = cv2.boundingRect(np.array(points))
        cropped = frame[y:y+h, x:x+w]
        count = len(os.listdir(save_path))
        cv2.imwrite(f"{save_path}/{count}.png", cropped)
        print(f"Saved {save_path}/{count}.png")
    elif key == ord('q'):
        break
    elif drawing:
        x, y = cv2.getWindowImageRect("Draw Gesture")[:2]
        points.append((x+5, y+5))

cap.release()
cv2.destroyAllWindows()
