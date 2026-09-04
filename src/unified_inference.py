"""
Combines the router with both trained models to produce one prediction
per frame: (branch, emotion_label, confidence, bbox_or_None).
"""

from .input_router import route

from .config import (
    FACE_MODEL_PATH,
    EMOJI_MODEL_PATH,
    FACE_EMOTIONS,
    EMOJI_EMOTIONS,
    FACE_INPUT_SIZE,
    EMOJI_INPUT_SIZE,
)

import tensorflow as tf
import numpy as np
import cv2


class UnifiedEmotionPredictor:

    def __init__(self):
        self.face_model = tf.keras.models.load_model(
            FACE_MODEL_PATH
        )

        self.emoji_model = tf.keras.models.load_model(
            EMOJI_MODEL_PATH
        )

    def _predict_face(self, frame_bgr, bbox):
        x, y, w, h = bbox

        crop = frame_bgr[y:y + h, x:x + w]

        if crop.size == 0:
            return None, 0.0

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY
        )

        resized = cv2.resize(
            gray,
            FACE_INPUT_SIZE
        )

        tensor = resized.reshape(
            1,
            *FACE_INPUT_SIZE,
            1
        ).astype("float32")

        probs = self.face_model.predict(
            tensor,
            verbose=0
        )[0]

        idx = int(np.argmax(probs))

        return (
            FACE_EMOTIONS[idx],
            float(probs[idx])
        )

    def _predict_emoji(self, frame_bgr):
        resized = cv2.resize(
            frame_bgr,
            EMOJI_INPUT_SIZE
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        tensor = rgb.reshape(
            1,
            *EMOJI_INPUT_SIZE,
            3
        ).astype("float32")

        probs = self.emoji_model.predict(
            tensor,
            verbose=0
        )[0]

        idx = int(np.argmax(probs))

        return (
            EMOJI_EMOTIONS[idx],
            float(probs[idx])
        )

    def predict(self, frame_bgr):
        """
        Returns:
            {
                "branch": "face" / "emoji" / None,
                "emotion": emotion label,
                "confidence": confidence score,
                "bbox": face bounding box or None
            }
        """

        branch, bbox = route(frame_bgr)

        if branch == "face":
            emotion, conf = self._predict_face(
                frame_bgr,
                bbox
            )

            return {
                "branch": "face",
                "emotion": emotion,
                "confidence": conf,
                "bbox": bbox
            }

        if branch == "emoji":
            emotion, conf = self._predict_emoji(
                frame_bgr
            )

            return {
                "branch": "emoji",
                "emotion": emotion,
                "confidence": conf,
                "bbox": None
            }

        return {
            "branch": None,
            "emotion": None,
            "confidence": 0.0,
            "bbox": None
        }
