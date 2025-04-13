import cv2
import numpy as np

from keras.models import load_model

model = load_model("digit_model.h5")

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(1, 28, 28, 1)

def recognize_equation(image):
    digits = []  # Assume cropped segments of digits are passed
    expr = ""

    # Simulate with 1 digit for now
    digit = preprocess(image)
    prediction = model.predict(digit)
    label = np.argmax(prediction)
    expr += str(label)

    return expr
