import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.5):
        """
        Initialize the hand detector with MediaPipe
        
        Args:
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum detection confidence threshold
            tracking_confidence: Minimum tracking confidence threshold
        """
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None
        
        # Define finger tip landmark indices
        self.FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.FINGER_PIPS = [2, 6, 10, 14, 18]  # PIP joints (one below the tip)
        self.FINGER_MCPS = [1, 5, 9, 13, 17]   # MCP joints (base of the finger)

    def detect(self, frame):
        """
        Detect hands in the frame
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            Processed frame with hand landmarks drawn
        """
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        self.results = self.hands.process(rgb)
        
        # Draw landmarks if hands detected
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp.solutions.hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0,0,255), thickness=2)
                )
        
        return frame

    def get_hand_position(self, frame):
        """
        Get the position of hand landmarks
        
        Args:
            frame: Input frame
            
        Returns:
            List of landmark positions or None if no hand detected
        """
        h, w, _ = frame.shape
        landmarks_pos = []
        
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]  # First hand only
            for lm in hand.landmark:
                # Convert to pixel coordinates
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks_pos.append((cx, cy))
            return landmarks_pos
        
        return None
    
    def get_finger_status(self, frame):
        """
        Get the up/down status of each finger
        
        Args:
            frame: Input frame
            
        Returns:
            List of booleans [thumb, index, middle, ring, pinky]
            True = finger is up, False = finger is down
        """
        fingers_up = [False] * 5
        h, w, _ = frame.shape
        
        if not self.results.multi_hand_landmarks:
            return fingers_up
        
        # Get the landmarks for the first hand
        hand_landmarks = self.results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark
        
        # Determine handedness
        handedness = "Right"  # Default assumption
        if self.results.multi_handedness:
            handedness = self.results.multi_handedness[0].classification[0].label
        
        # Check thumb (different logic based on hand)
        if handedness == "Right":
            # For right hand, thumb is up if thumb tip is to the right of thumb IP
            fingers_up[0] = landmarks[self.FINGER_TIPS[0]].x > landmarks[self.FINGER_PIPS[0]].x
        else:
            # For left hand, thumb is up if thumb tip is to the left of thumb IP
            fingers_up[0] = landmarks[self.FINGER_TIPS[0]].x < landmarks[self.FINGER_PIPS[0]].x
        
        # Check other fingers
        for i in range(1, 5):
            # Finger is up if tip is higher (y is smaller) than PIP joint
            fingers_up[i] = landmarks[self.FINGER_TIPS[i]].y < landmarks[self.FINGER_PIPS[i]].y
        
        return fingers_up
    
    def is_pointing(self, hand_landmarks):
        """Detects if the index finger is pointing (drawing mode)."""
        landmarks = hand_landmarks.landmark
        index_tip = landmarks[8]  # INDEX_FINGER_TIP
        index_mcp = landmarks[5]  # INDEX_FINGER_MCP

        other_finger_tips = [landmarks[12], landmarks[16], landmarks[20]]  # MIDDLE, RING, PINKY tips
        other_finger_mcps = [landmarks[9], landmarks[13], landmarks[17]]  # MIDDLE, RING, PINKY MCPs

        index_extended = index_tip.y < index_mcp.y - 0.05
        others_fully_curled = all(tip.y > mcp.y + 0.02 for tip, mcp in zip(other_finger_tips, other_finger_mcps))
        index_above_others = all(index_tip.y < tip.y - 0.02 for tip in other_finger_tips)

        return index_extended and others_fully_curled and index_above_others

    def is_palm_open(self, hand_landmarks):
        """Detects if the palm is open (lift pen mode)."""
        landmarks = hand_landmarks.landmark
        fingers = [
            (8, 5),  # INDEX_FINGER_TIP, INDEX_FINGER_MCP
            (12, 9),  # MIDDLE_FINGER_TIP, MIDDLE_FINGER_MCP
            (16, 13),  # RING_FINGER_TIP, RING_FINGER_MCP
            (20, 17)  # PINKY_TIP, PINKY_MCP
        ]
        open_fingers = sum(1 for tip, mcp in fingers if landmarks[tip].y < landmarks[mcp].y)
        return open_fingers >= 3  # At least 3 fingers need to be extended

    def is_fist(self, hand_landmarks):
        """Detects if the hand is in a fist (clear mode)."""
        landmarks = hand_landmarks.landmark
        fingers = [
            (8, 5),  # INDEX_FINGER_TIP, INDEX_FINGER_MCP
            (12, 9),  # MIDDLE_FINGER_TIP, MIDDLE_FINGER_MCP
            (16, 13),  # RING_FINGER_TIP, RING_FINGER_MCP
            (20, 17)  # PINKY_TIP, PINKY_MCP
        ]
        curled_fingers = sum(1 for tip, mcp in fingers if landmarks[tip].y > landmarks[mcp].y)
        return curled_fingers >= 3  # At least 3 fingers need to be curled

    def is_three_fingers_up(self, hand_landmarks):
        """Detects if three fingers are up (solve mode)."""
        landmarks = hand_landmarks.landmark
        
        # Check if index, middle, and thumb are extended
        index_up = landmarks[8].y < landmarks[5].y
        middle_up = landmarks[12].y < landmarks[9].y
        thumb_up = landmarks[4].x > landmarks[3].x  # For right hand
        
        # Check if ring and pinky are down
        ring_down = landmarks[16].y > landmarks[13].y
        pinky_down = landmarks[20].y > landmarks[17].y
        
        return index_up and middle_up and thumb_up and ring_down and pinky_down
    
    def get_gesture(self, frame):
        """
        Recognize common hand gestures with improved precision
        
        Args:
            frame: Input frame
            
        Returns:
            gesture: String describing the detected gesture
            action: Recommended action based on gesture
        """
        # Get landmarks for more precise gesture detection
        if not self.results or not self.results.multi_hand_landmarks:
            return "none", "none"
            
        hand_landmarks = self.results.multi_hand_landmarks[0]
        
        # Check for pointing gesture (index finger up, others down)
        if self.is_pointing(hand_landmarks):
            return "index_finger", "draw"
        
        # Check for open palm gesture
        elif self.is_palm_open(hand_landmarks):
            return "open_palm", "lift_pen"
        
        # Check for fist gesture
        elif self.is_fist(hand_landmarks):
            return "fist", "clear"
        
        # Check for three fingers up (solve)
        elif self.is_three_fingers_up(hand_landmarks):
            return "three_fingers", "solve"
        
        else:
            return "unknown", "none"
            
    def get_index_finger_tip(self, frame):
        """
        Get the position of the index finger tip
        
        Args:
            frame: Input frame
            
        Returns:
            (x, y) coordinates of index finger tip or None if not detected
        """
        h, w, _ = frame.shape
        
        if not self.results.multi_hand_landmarks:
            return None
            
        landmarks = self.results.multi_hand_landmarks[0].landmark
        index_tip = landmarks[8]  # Index finger tip is landmark 8
        
        x, y = int(index_tip.x * w), int(index_tip.y * h)
        return (x, y)