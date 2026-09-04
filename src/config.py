"""
Central config for Manobhaav.
Keeping this separate avoids magic strings/numbers scattered across
training scripts and the inference pipeline — one source of truth.
"""
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")
EMOJI_DATASET_DIR = os.path.join(DATA_DIR, "emoji_dataset")
# user supplies this (see README)
FER_DATASET_DIR = os.path.join(DATA_DIR, "fer2013")

FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_emotion_model.keras")
EMOJI_MODEL_PATH = os.path.join(MODELS_DIR, "emoji_emotion_model.keras")

# --- Face branch: constrained to what FER2013 actually supports ---
# Real public datasets don't have "Thinking" or a clean "Love" label —
# see the note in the chat response. This mapping is the honest version.
FACE_EMOTIONS = ["angry", "disgust", "fear",
                 "happy", "neutral", "sad", "surprise"]
FACE_INPUT_SIZE = (48, 48)  # grayscale, matches FER2013

# --- Emoji branch: your full 8-class target, since we generate this data ourselves ---
EMOJI_EMOTIONS = [
    "happy", "sad", "angry", "neutral",
    "thinking", "surprised", "fear", "love",
]
EMOJI_INPUT_SIZE = (64, 64)  # RGB

# Unicode glyph per class used to SYNTHETICALLY render the emoji dataset.
# Swap these for whichever emoji you want each class to represent.
EMOJI_GLYPHS = {
    "happy": "\U0001F600",       # 😀
    "sad": "\U0001F622",         # 😢
    "angry": "\U0001F621",       # 😡
    "neutral": "\U0001F610",     # 😐
    "thinking": "\U0001F914",    # 🤔
    "surprised": "\U0001F632",   # 😲
    "fear": "\U0001F628",        # 😨
    "love": "\U0001F60D",        # 😍
}

# Router thresholds
FACE_DETECTION_CONFIDENCE = 0.5
EMOJI_EDGE_DENSITY_MAX = 0.12   # emoji glyphs are flat-colored -> low edge density
INFER_EVERY_N_FRAMES = 3         # throttle CNN inference for real-time FPS
