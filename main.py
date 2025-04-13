import streamlit as st
st.set_page_config(page_title="Gesture Math Solver", layout="wide")
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from collections import deque
import traceback

# Load model with verification
try:
    model = load_model("digit_model.h5")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

# Streamlit setup
st.title("✍️ Gesture-Controlled Math Solver")

# Setup columns
col1, col2 = st.columns([3, 1])
FRAME_WINDOW = col1.image([])
status_text = col2.empty()

# State initialization
if "points" not in st.session_state:
    st.session_state.update({
        "points": [],
        "expression": "",
        "result": "",
        "last_action": None
    })

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
drawing = mp.solutions.drawing_utils

# Helper functions
def fingers_up(hand_landmarks):
    tips = [8, 12, 16, 20]  # Finger tip landmarks
    fingers = []
    for tip in tips:
        fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y)
    return fingers

def segment_digits(points, canvas):
    digit_imgs = []
    current_stroke = []
    operator_positions = []
    
    # Split continuous strokes and detect operators
    for pt in points + [None]:
        if pt is None or (current_stroke and len(current_stroke) > 10) and ((pt is None) or (np.linalg.norm(np.array(pt)-np.array(current_stroke[-1])) > 30)):
            if current_stroke:
                x_vals = [p[0] for p in current_stroke]
                y_vals = [p[1] for p in current_stroke]
                
                # Check if this is likely an operator (+,-, etc.)
                aspect_ratio = (max(x_vals)-min(x_vals))/(max(y_vals)-min(y_vals)+1e-6)
                if 0.5 < aspect_ratio < 2.0 and (max(y_vals)-min(y_vals)) < 50:
                    operator_positions.append((min(x_vals), max(x_vals)))
                    current_stroke = []
                    continue
                
                # Process as digit
                padding = 15
                xmin = max(0, min(x_vals)-padding)
                xmax = min(canvas.shape[1], max(x_vals)+padding)
                ymin = max(0, min(y_vals)-padding)
                ymax = min(canvas.shape[0], max(y_vals)+padding)
                
                digit = canvas[ymin:ymax, xmin:xmax]
                if digit.size > 0:
                    # Center the digit in a square
                    h, w = digit.shape
                    size = max(h, w)
                    centered = np.ones((size, size), dtype=np.uint8) * 255
                    y_offset = (size - h) // 2
                    x_offset = (size - w) // 2
                    centered[y_offset:y_offset+h, x_offset:x_offset+w] = digit
                    
                    # Resize to 28x28
                    digit = cv2.resize(centered, (28, 28))
                    # Invert colors (MNIST style)
                    digit = cv2.bitwise_not(digit)
                    # Normalize
                    digit = digit.astype("float32") / 255.0
                    
                    digit_imgs.append((xmin, digit.reshape(1, 28, 28, 1)))
            current_stroke = [] if pt is None else [pt]
        elif pt is not None:
            current_stroke.append(pt)
    
    # Sort digits left-to-right and process
    digit_imgs.sort(key=lambda x: x[0])
    processed_digits = []
    for x, digit in digit_imgs:
        # Check if this position has an operator
        for op_xmin, op_xmax in operator_positions:
            if op_xmin <= x <= op_xmax:
                processed_digits.append(('+', None))  # Simple case - just handle + for now
                break
        
        # Process digit
        processed_digits.append(('digit', digit))
    
    return processed_digits

def predict_expression(points, canvas):
    elements = segment_digits(points, canvas)
    expression = ""

    for elem in elements:
        if elem[0] == '+':
            expression += '+'
        elif elem[0] == 'digit':
            data = elem[1]
            pred = model.predict(data, verbose=0)
            digit = str(np.argmax(pred))
            confidence = np.max(pred)
            if confidence > 0.3:  # Lower confidence threshold
                expression += digit
            else:
                expression += "?"  # Mark uncertain predictions
    
    return expression


def evaluate(expr):
    try:
        # Simple safety check
        if not all(c in '0123456789+-' for c in expr):
            return "❌ Invalid"
        return str(eval(expr))
    except:
        return "❌ Invalid"

# UI Controls
with col2:
    activate = st.checkbox("🎥 Activate Webcam", value=True)
    if st.button("🧹 Clear All"):
        st.session_state.points = []
        st.session_state.expression = ""
        st.session_state.result = ""
    
    st.markdown("### 📝 Expression")
    expr_display = st.empty()
    
    st.markdown("### 🔢 Result")
    result_display = st.empty()

# Main processing loop
cap = cv2.VideoCapture(0)
while activate and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    # Create drawing canvas
    canvas = np.ones((h, w), dtype=np.uint8) * 255
    
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        
        # Get finger states
        fingers = fingers_up(hand)
        ix, iy = int(hand.landmark[8].x * w), int(hand.landmark[8].y * h)
        
        # Gesture handling
        if fingers == [1, 0, 0, 0]:  # Index finger: draw
            st.session_state.points.append((ix, iy))
            st.session_state.last_action = "drawing"
        elif fingers == [1, 1, 1, 0] and st.session_state.last_action != "solving":  # Three fingers: solve
            expr = predict_expression(st.session_state.points, canvas)
            result = evaluate(expr)
            st.session_state.expression = expr
            st.session_state.result = result
            st.session_state.last_action = "solving"
            st.rerun()
        elif fingers == [0, 0, 0, 0]:  # Fist: clear
            st.session_state.points = []
            st.session_state.last_action = "clearing"
        elif fingers == [1, 1, 1, 1]:  # Open hand: lift pen
            st.session_state.points.append(None)
            st.session_state.last_action = "lifting"
    
    # Draw the path
    for i in range(1, len(st.session_state.points)):
        pt1 = st.session_state.points[i-1]
        pt2 = st.session_state.points[i]
        if None in (pt1, pt2):
            continue
        cv2.line(canvas, pt1, pt2, 0, 10)
        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
    
    # Display UI elements
    expr_display.code(st.session_state.expression)
    result_display.code(st.session_state.result)
    
    # Show finger count for debugging
    cv2.putText(frame, f"Fingers: {sum(fingers) if results.multi_hand_landmarks else 0}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Combine frames and display
    combined = np.hstack([frame, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)])
    FRAME_WINDOW.image(combined, channels="BGR")

cap.release()