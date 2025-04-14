import cv2
import numpy as np
import tensorflow as tf
import os

class GestureRecognizer:
    def __init__(self, model_path='gesture_model.h5', label_map_path='label_map.npy', fallback_model_path='digit_model.h5'):
        """
        Initialize the gesture recognizer
        
        Args:
            model_path: Path to the gesture model
            label_map_path: Path to the label mapping file
            fallback_model_path: Path to fallback digit model
        """
        self.model = None
        self.label_map = None
        
        try:
            # Try to load the gesture model first
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"Loaded gesture model from {model_path}")
                
                # Try to load the label map
                if os.path.exists(label_map_path):
                    self.label_map = np.load(label_map_path, allow_pickle=True).item()
                    print(f"Loaded label map: {self.label_map}")
                else:
                    # Create a default label map
                    print("Label map not found, creating default")
                    self.label_map = {i: str(i) for i in range(10)}
                    # Add operators
                    for i, op in enumerate(['+', '-', '*', '/'], start=10):
                        if i < self.model.output.shape[1]:
                            self.label_map[i] = op
            
            # Fall back to digit model if needed
            elif os.path.exists(fallback_model_path):
                self.model = tf.keras.models.load_model(fallback_model_path)
                print(f"Loaded digit model from {fallback_model_path}")
                # Create digit-only label map
                self.label_map = {i: str(i) for i in range(10)}
            
            else:
                raise FileNotFoundError(f"Model not found: {model_path} or {fallback_model_path}")
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def preprocess(self, image):
        """
        Preprocess an image for prediction
        
        Args:
            image: Grayscale image of a digit/symbol
            
        Returns:
            Preprocessed tensor ready for model input
        """
        if image is None or image.size == 0:
            return None
            
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Make it square with white background
        h, w = gray.shape
        size = max(h, w)
        squared = np.ones((size, size), dtype=np.uint8) * 255
        
        # Center the content
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        squared[y_offset:y_offset+h, x_offset:x_offset+w] = gray
        
        # Resize to 28x28
        resized = cv2.resize(squared, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values
        normalized = resized.astype("float32") / 255.0
        
        # Reshape for model input
        tensor = normalized.reshape(1, 28, 28, 1)
        
        return tensor

    def recognize(self, image, confidence_threshold=0.4):
        """
        Recognize a digit or symbol in the image
        
        Args:
            image: Input image (grayscale)
            confidence_threshold: Minimum confidence for prediction
            
        Returns:
            predicted symbol and confidence score
        """
        if self.model is None:
            return "?", 0.0
            
        # Preprocess the image
        tensor = self.preprocess(image)
        if tensor is None:
            return "?", 0.0
            
        # Make prediction
        try:
            predictions = self.model.predict(tensor, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx]
            
            # Get the label using the mapping
            if class_idx in self.label_map and confidence >= confidence_threshold:
                return self.label_map[class_idx], confidence
            else:
                return "?", confidence
                
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return "?", 0.0

    def recognize_segments(self, segments, confidence_threshold=0.4):
        """
        Recognize multiple segments and form an expression
        
        Args:
            segments: List of preprocessed image segments
            confidence_threshold: Minimum confidence for prediction
            
        Returns:
            Recognized expression string
        """
        expression = ""
        
        for segment in segments:
            symbol, confidence = self.recognize(segment, confidence_threshold)
            expression += symbol
            
        return expression