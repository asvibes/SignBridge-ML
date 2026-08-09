"""
mediapipe_detector.py
----------------------
Thin wrapper around MediaPipe Hands so the rest of the pipeline
(collection, training, inference) never has to touch the MediaPipe
API directly. Swap this module out later if you switch hand-tracking
backends (e.g. a lighter-weight detector for Raspberry Pi deployment).
"""

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.7,
                 min_tracking_confidence=0.5, static_image_mode=False):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray):
        """
        Run detection on a single BGR frame.

        Returns:
            landmarks: list of 21 landmark objects (with .x, .y, .z) for the
                       first detected hand, or None if no hand is found.
            results:   raw MediaPipe results (useful for drawing overlays).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            first_hand = results.multi_hand_landmarks[0]
            return list(first_hand.landmark), results

        return None, results

    def draw_landmarks(self, frame_bgr: np.ndarray, results) -> np.ndarray:
        """Draw hand landmarks + connections on the frame for the UI."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )
        return frame_bgr

    def close(self):
        self.hands.close()
