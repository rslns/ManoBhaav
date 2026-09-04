"""
Decides: face branch, emoji branch, or neither.

Face detection (MediaPipe) is the primary signal — it's already trained
on real faces, so a confident detection is strong evidence. When no face
is found, we don't just assume "must be an emoji" — that's a common
beginner mistake (silently misclassifying random background clutter as
an emoji). Instead we gate with a cheap visual heuristic: emoji glyphs
are flat-colored/vector-like, so they have LOW edge density and LOW
color variance compared to a busy real-world scene.
"""
import cv2
import numpy as np
import mediapipe as mp

from .config import FACE_DETECTION_CONFIDENCE, EMOJI_EDGE_DENSITY_MAX

_mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=FACE_DETECTION_CONFIDENCE
)


def detect_face_bbox(frame_bgr):
    """Returns (x, y, w, h) of the most confident face, or None."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = _mp_face.process(rgb)
    if not result.detections:
        return None

    best = max(result.detections, key=lambda d: d.score[0])
    h, w, _ = frame_bgr.shape
    box = best.location_data.relative_bounding_box
    x, y = max(int(box.xmin * w), 0), max(int(box.ymin * h), 0)
    bw, bh = int(box.width * w), int(box.height * h)
    return (x, y, min(bw, w - x), min(bh, h - y))


def looks_like_emoji(frame_bgr):
    """
    Cheap heuristic gate: emoji renders are flat-colored with hard edges
    around a small glyph, so Canny edge density on the whole ROI is low
    compared to a photographic/textured scene.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = np.count_nonzero(edges) / edges.size
    return edge_density < EMOJI_EDGE_DENSITY_MAX


def route(frame_bgr):
    """
    Returns one of:
      ("face", (x, y, w, h))
      ("emoji", None)
      (None, None)   -- nothing recognizable
    """
    face_bbox = detect_face_bbox(frame_bgr)
    if face_bbox is not None:
        return "face", face_bbox

    if looks_like_emoji(frame_bgr):
        return "emoji", None

    return None, None
