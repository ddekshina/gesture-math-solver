import streamlit as st
import cv2
import numpy as np
import os
import tensorflow as tf
from gesture_detector import HandDetector
from utils.drawing import create_canvas, draw_strokes, process_canvas_for_segmentation
from math_solver import solve_expression, solve_with_gemini

# Page configuration
st.set_page_config(page_title="Gesture Math Solver", layout="wide")

# App title and introduction
st.title("✍️ Gesture-Controlled Math Solver")
st.markdown("""
Draw math expressions with hand gestures and get instant solutions!
- 👆 **Index finger up**: Draw
- ✋ **Open palm**: Lift pen
- ✊ **Fist**: Clear canvas
- 👌 **Three fingers up**: Solve expression
""")

# Initialize session state
if "points" not in st.session_state:
    st.session_state.update({
        "points": [],
        "expression": "",
        "result": "",
        "last_action": None,
        "last_gesture": None,
        "use_gemini": False,
        "api_key": "",
        "debug_mode": False,
        "model_loaded": False,
        "label_map": None
    })

# Setup columns
col1, col2 = st.columns([3, 1])
with col1:
    # Camera feed display
    FRAME_WINDOW = st.image([])

# Model loading function
@st.cache_resource
def load_model():
    """Load the gesture recognition model"""
    try:
        model_path = "gesture_model.h5"
        if not os.path.exists(model_path):
            st.warning("Gesture model not found. Using digit model instead.")
            model_path = "digit_model.h5"
            if not os.path.exists(model_path):
                raise FileNotFoundError("No model file found")
                
        model = tf.keras.models.load_model(model_path)
        
        # Load label map if available
        label_map = None
        if os.path.exists("label_map.npy"):
            label_map = np.load("label_map.npy", allow_pickle=True).item()
        else:
            # Default label map for 0-9
            label_map = {i: str(i) for i in range(10)}
            # Add operators if using gesture model
            if model_path == "gesture_model.h5":
                # Assuming standard ordering of classes
                for i, op in enumerate(['+', '-', '*', '/'], start=10):
                    if i < model.output.shape[1]:
                        label_map[i] = op
                        
        return model, label_map
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Initialize model
model, label_map = load_model()
st.session_state.model_loaded = model is not None
st.session_state.label_map = label_map

# Controls in sidebar
with st.sidebar:
    st.header("Settings")
    
    # Camera control
    activate = st.checkbox("🎥 Activate Camera", value=True)
    
    # Action buttons
    if st.button("🧹 Clear Canvas"):
        st.session_state.points = []
        st.session_state.expression = ""
        st.session_state.result = ""
    
    # Gemini API settings
    st.session_state.use_gemini = st.checkbox("Use Gemini AI for step-by-step solutions", value=False)
    
    if st.session_state.use_gemini:
        st.session_state.api_key = st.text_input(
            "Gemini API Key", 
            value=st.session_state.api_key,
            type="password",
            help="Optional: Enter your Gemini API key to enable step-by-step solutions"
        )
    
    # Debug mode toggle
    st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)
    
    # Sample saving section
    st.header("Sample Collection")
    sample_label = st.text_input("Label", max_chars=1, help="Enter a digit or operator (+,-,*,/)")
    
    if st.button("Save Current Drawing as Sample"):
        if not sample_label or sample_label not in "0123456789+-*/":
            st.error("Please enter a valid label (0-9, +, -, *, /)")
        elif len(st.session_state.points) == 0:
            st.error("Please draw something first")
        else:
            try:
                # Create dataset directory if it doesn't exist
                dataset_path = os.path.join("dataset", sample_label)
                os.makedirs(dataset_path, exist_ok=True)
                
                # Process and save the current drawing
                h, w = 480, 640  # Default canvas size
                canvas = create_canvas(h, w)
                canvas = draw_strokes(canvas, st.session_state.points)
                
                # Extract the drawing
                points_array = [p for p in st.session_state.points if p is not None]
                if points_array:
                    x_vals = [p[0] for p in points_array]
                    y_vals = [p[1] for p in points_array]
                    
                    padding = 20
                    xmin = max(0, min(x_vals) - padding)
                    xmax = min(w, max(x_vals) + padding)
                    ymin = max(0, min(y_vals) - padding)
                    ymax = min(h, max(y_vals) + padding)
                    
                    # Crop and process
                    if xmin < xmax and ymin < ymax:
                        cropped = canvas[ymin:ymax, xmin:xmax]
                        
                        # Make square with white background
                        h_crop, w_crop = cropped.shape
                        size = max(h_crop, w_crop)
                        squared = np.ones((size, size), dtype=np.uint8) * 255
                        
                        # Center the content
                        y_offset = (size - h_crop) // 2
                        x_offset = (size - w_crop) // 2
                        squared[y_offset:y_offset+h_crop, x_offset:x_offset+w_crop] = cropped
                        
                        # Resize to 28x28
                        resized = cv2.resize(squared, (28, 28), interpolation=cv2.INTER_AREA)
                        
                        # Save the image
                        count = len([f for f in os.listdir(dataset_path) if f.endswith('.png')])
                        save_file = os.path.join(dataset_path, f"{count:03d}.png")
                        cv2.imwrite(save_file, resized)
                        st.success(f"Sample saved as {save_file}")
            except Exception as e:
                st.error(f"Error saving sample: {str(e)}")

# Display area for expression and results in the right column
with col2:
    st.markdown("### 📝 Expression")
    expr_display = st.code(st.session_state.expression)
    
    st.markdown("### 🔢 Result")
    result_display = st.code(st.session_state.result)
    
    # Add solver button
    if st.button("🧮 Solve Expression"):
        if st.session_state.expression:
            if st.session_state.use_gemini and st.session_state.api_key:
                st.session_state.result = solve_with_gemini(
                    st.session_state.expression, 
                    api_key=st.session_state.api_key
                )
            else:
                st.session_state.result = solve_expression(st.session_state.expression)
                
    # Debug information if enabled
    if st.session_state.debug_mode:
        st.markdown("### 🔍 Debug Info")
        st.write(f"Last Gesture: {st.session_state.last_gesture}")
        st.write(f"Points Count: {len(st.session_state.points)}")
        st.write(f"Model Loaded: {st.session_state.model_loaded}")
        st.write(f"Label Map: {st.session_state.label_map}")

# Main processing loop when camera is active
if activate and st.session_state.model_loaded:
    try:
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to open webcam")
            st.stop()
            
        # Initialize hand detector
        hand_detector = HandDetector(max_hands=1, detection_confidence=0.7)
        
        # Create initial canvas
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        canvas = create_canvas(h, w)
        
        # Processing loop
        while activate:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam")
                break
                
            # Flip frame horizontally for intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Detect hand
            frame = hand_detector.detect(frame)
            
            # Recognize gesture
            gesture, action = hand_detector.get_gesture(frame)
            st.session_state.last_gesture = gesture
            
            # Handle actions based on gesture
            if action == "draw":
                # Get index finger tip position
                index_tip = hand_detector.get_index_finger_tip(frame)
                if index_tip:
                    st.session_state.points.append(index_tip)
                    st.session_state.last_action = "drawing"
                    
            elif action == "lift_pen" and st.session_state.last_action == "drawing":
                # Add None to separate strokes
                st.session_state.points.append(None)
                st.session_state.last_action = "lifting"
                
            elif action == "clear":
                # Clear the canvas
                st.session_state.points = []
                st.session_state.expression = ""
                st.session_state.result = ""
                st.session_state.last_action = "clearing"
                
            elif action == "solve" and st.session_state.last_action != "solving":
                # Process the canvas for prediction
                if len(st.session_state.points) > 0:
                    # Update the canvas
                    canvas = create_canvas(h, w)
                    canvas = draw_strokes(canvas, st.session_state.points)
                    
                    # Process for segment recognition
                    segments = process_canvas_for_segmentation(canvas, st.session_state.points)
                    
                    # Generate expression from segments
                    expression = ""
                    for _, segment_data in segments:
                        if segment_data is not None:
                            # Make prediction
                            prediction = model.predict(segment_data, verbose=0)
                            pred_idx = np.argmax(prediction)
                            confidence = np.max(prediction)
                            
                            # Get label from map
                            if pred_idx in st.session_state.label_map:
                                label = st.session_state.label_map[pred_idx]
                                if confidence > 0.4:  # Confidence threshold
                                    expression += label
                                else:
                                    expression += "?"  # Mark uncertain predictions
                    
                    # Update expression and solve
                    st.session_state.expression = expression
                    if expression:
                        if st.session_state.use_gemini and st.session_state.api_key:
                            st.session_state.result = solve_with_gemini(
                                expression, 
                                api_key=st.session_state.api_key
                            )
                        else:
                            st.session_state.result = solve_expression(expression)
                    
                    st.session_state.last_action = "solving"
                    
            # Update canvas with current strokes
            canvas = create_canvas(h, w)
            canvas = draw_strokes(canvas, st.session_state.points)
            
            # Add debug info to frame
            if st.session_state.debug_mode:
                # Add gesture and action info
                cv2.putText(
                    frame, 
                    f"Gesture: {gesture} | Action: {action}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                # Show finger counts
                fingers = hand_detector.get_finger_status(frame)
                if fingers:
                    finger_text = "Fingers: " + "".join(["🖐️" if f else "👊" for f in fingers])
                    cv2.putText(
                        frame, 
                        finger_text, 
                        (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), 
                        2
                    )
            
            # Combine frame and canvas for display
            canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            alpha = 0.3
            overlay = cv2.addWeighted(frame, 1-alpha, canvas_rgb, alpha, 0)
            
            # Display side by side
            combined = np.hstack([frame, canvas_rgb])
            FRAME_WINDOW.image(combined, channels="BGR")
            
            # Update UI elements
            expr_display.code(st.session_state.expression)
            result_display.code(st.session_state.result)
            
        # Release camera when done
        cap.release()
        
    except Exception as e:
        st.error(f"Error in processing loop: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
else:
    # Display startup message
    if not st.session_state.model_loaded:
        st.error("Model not loaded. Please check the console for errors.")
    elif not activate:
        st.info("Camera is deactivated. Check the 'Activate Camera' box to start.")