#Streamlit app
import streamlit as st
import cv2
from gesture_detector import HandDetector
from math_solver import solve_math_expression

st.title("🖐️ Gesture-Controlled Math Solver")

run = st.checkbox('Activate Webcam')
FRAME_WINDOW = st.image([])

detector = HandDetector()
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detector.detect(frame)

    # Example: If gesture complete
    # expr = "3 + 5"
    # result = solve_math_expression(expr)
    # st.write("Expression:", expr)
    # st.write("Result:", result)

    FRAME_WINDOW.image(frame, channels="BGR")
