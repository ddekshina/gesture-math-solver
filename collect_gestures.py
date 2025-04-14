import cv2
import os
import numpy as np
import mediapipe as mp

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Create dataset directory if it doesn't exist
dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

# Possible labels
valid_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/']
label = input(f"Enter label for gesture ({', '.join(valid_labels)}): ")

if label not in valid_labels:
    print(f"Invalid label. Please choose from: {', '.join(valid_labels)}")
    exit()

# Create label directory
if label == '*':
    label = 'x'
if label == '/':
    label = 'divide'
save_path = os.path.join(dataset_path, label)
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
drawing = False
points = []
canvas = None

# Instructions
instructions = [
    "Press 'd' to start drawing",
    "Move your index finger to draw",
    "Press 's' to save the current gesture",
    "Press 'c' to clear the canvas",
    "Press 'q' to quit"
]

while True:
    ret, frame = cap.read()
    if not ret:
        continue
        
    frame = cv2.flip(frame, 1)  # Mirror for more intuitive drawing
    h, w, _ = frame.shape
    
    if canvas is None:
        canvas = np.ones((h, w), dtype=np.uint8) * 255
    
    # Process hand landmarks
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    # Draw hand landmarks and handle drawing
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get index finger tip position
            if drawing:
                index_tip = hand_landmarks.landmark[8]
                x, y = int(index_tip.x * w), int(index_tip.y * h)
                points.append((x, y))
    
    # Draw the current stroke
    if len(points) > 1:
        for i in range(1, len(points)):
            pt1, pt2 = points[i-1], points[i]
            cv2.line(canvas, pt1, pt2, 0, 10)  # Black on white
            cv2.line(frame, pt1, pt2, (0, 0, 255), 2)  # Red on camera
    
    # Show canvas overlay on frame
    canvas_display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    alpha = 0.4
    overlay = cv2.addWeighted(frame, 1-alpha, canvas_display, alpha, 0)
    
    # Add instructions and label info
    cv2.putText(overlay, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for i, text in enumerate(instructions):
        cv2.putText(overlay, text, (10, 60 + 25*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Gesture Collector", overlay)
    
    key = cv2.waitKey(1)
    
    if key == ord('d'):
        drawing = True
        points = []
    elif key == ord('s') and points:
        # Process the canvas for saving
        if len(points) > 0:
            # Get bounding box of the drawing
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            padding = 20
            
            xmin = max(0, min(x_vals) - padding)
            xmax = min(w, max(x_vals) + padding)
            ymin = max(0, min(y_vals) - padding)
            ymax = min(h, max(y_vals) + padding)
            
            # Crop the drawing
            if xmin < xmax and ymin < ymax:
                cropped = canvas[ymin:ymax, xmin:xmax]
                
                # Make square with white background
                h_crop, w_crop = cropped.shape
                size = max(h_crop, w_crop)
                squared = np.ones((size, size), dtype=np.uint8) * 255
                
                # Center the digit in the square
                y_offset = (size - h_crop) // 2
                x_offset = (size - w_crop) // 2
                squared[y_offset:y_offset+h_crop, x_offset:x_offset+w_crop] = cropped
                
                # Resize to 28x28
                resized = cv2.resize(squared, (28, 28), interpolation=cv2.INTER_AREA)
                
                # Save the image
                count = len(os.listdir(save_path))
                save_file = os.path.join(save_path, f"{count:03d}.png")
                cv2.imwrite(save_file, resized)
                print(f"Saved {save_file}")
                
                # Show the processed image
                cv2.imshow("Saved Image", cv2.resize(resized, (112, 112)))
        
        # Reset for next drawing
        canvas = np.ones((h, w), dtype=np.uint8) * 255
        points = []
        drawing = False
    elif key == ord('c'):
        canvas = np.ones((h, w), dtype=np.uint8) * 255
        points = []
        drawing = False
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()