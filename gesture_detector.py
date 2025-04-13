import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, max_hands=1, detection_confidence=0.8):
        self.hands = mp.solutions.hands.Hands(max_num_hands=max_hands, min_detection_confidence=detection_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        return frame

    def get_finger_status(self, frame):
        status = []
        h, w, _ = frame.shape
        if self.results.multi_hand_landmarks:
            lm = self.results.multi_hand_landmarks[0].landmark

            # Index tip is landmark 8, PIP is 6
            index_tip = lm[8]
            index_pip = lm[6]

            # Middle finger tip and pip
            middle_tip = lm[12]
            middle_pip = lm[10]

            # Ring
            ring_tip = lm[16]
            ring_pip = lm[14]

            # Pinky
            pinky_tip = lm[20]
            pinky_pip = lm[18]

            # Thumb (sideways logic)
            thumb_tip = lm[4]
            thumb_ip = lm[3]

            def is_up(tip, pip): return tip.y < pip.y

            fingers = [
                thumb_tip.x < thumb_ip.x,  # Right hand
                is_up(index_tip, index_pip),
                is_up(middle_tip, middle_pip),
                is_up(ring_tip, ring_pip),
                is_up(pinky_tip, pinky_pip),
            ]
            return fingers  # [thumb, index, middle, ring, pinky]
        return [False] * 5
