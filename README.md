Manobhaav — Unified Face + Emoji Emotion Recognition
Computer vision system that recognizes emotion from two different visual domains — a human face or an emoji — using one pipeline. It automatically detects which type of input it's looking at, then routes it to the right model.

Emotion recognition for faces already exists in plenty of projects. Emotion recognition for emojis basically doesn't — there's no public dataset for it. This project builds both, including generating its own emoji training data from scratch.

Why a router instead of one model? A face photo and an emoji glyph live in completely different visual domains — cramming both into a single classifier makes both worse. Splitting into a router + two specialized branches is the more reliable design.

Why transfer learning for faces, but not for emoji? FER2013 (~35k images) is too small to train a deep CNN from scratch without overfitting, so the face branch fine-tunes a frozen ImageNet backbone (MobileNetV2). The emoji branch trains a small CNN from scratch — the task is far simpler (flat glyphs, not real-world photos) and the synthetic dataset is effectively unlimited, so transfer learning's regularization benefit isn't needed there.

Why synthetic emoji data? No public emoji-emotion dataset exists. So the emoji branch is trained on emoji glyphs rendered directly from a font, with randomized rotation, background, scale, and blur applied as augmentation — giving a clean, unlimited, zero-licensing-risk dataset.

Emotion classes
Branch	Classes	Source
Face	angry, disgust, fear, happy, neutral, sad, surprise (+ optional custom: crying, thinking)	FER2013 (+ self-collected webcam samples for the two extra classes)
Emoji	happy, sad, angry, neutral, thinking, surprised, fear, love	Self-generated via font rendering
Note: "crying" and "thinking" aren't labels in any public facial-expression dataset — the face branch only gets them if you run the optional extended-training step and collect your own samples via collect_custom_faces.py.

Tech stack
Language: Python
Computer vision: OpenCV, MediaPipe (face detection)
Deep learning: TensorFlow / Keras, MobileNetV2 (transfer learning)
Data generation: Pillow (synthetic emoji rendering)
Deployment target: local / real-time webcam (CPU-friendly, inference throttled for FPS)
Project structure
manobhaav/
├── requirements.txt
├── main.py                          # entry point: webcam loop / single image / save video
├── src/
│   ├── config.py                    # class labels, paths, thresholds
│   ├── emoji_dataset_generator.py   # synthetic emoji dataset via font rendering
│   ├── models.py                    # CNN architectures for both branches
│   ├── train_face_model.py          # trains 7-class face model on FER2013
│   ├── train_face_model_extended.py # trains 9-class face model (+ custom classes)
│   ├── train_emoji_model.py         # trains 8-class emoji model
│   ├── collect_custom_faces.py      # webcam tool to collect custom-class samples
│   ├── input_router.py              # face vs. emoji vs. neither decision logic
│   ├── unified_inference.py         # combines router + both models
│   └── check_setup.py               # diagnostic: dataset health + prediction distribution
├── data/
│   ├── fer2013/train|test/<class>/  # you supply this (Kaggle FER2013)
│   ├── custom_faces/<class>/        # optional, self-collected
│   └── emoji_dataset/<class>/       # auto-generated
└── models/
    ├── face_emotion_model.keras
    └── emoji_emotion_model.keras
DataSet : https://www.kaggle.com/datasets/msambare/fer2013 

Setup
git clone <https://github.com/rslns/ManoBhaav>
cd manobhaav
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
You'll also need a color-emoji font installed for the synthetic dataset generator:

# Ubuntu/Debian
sudo apt-get install -y fonts-noto-color-emoji
# Windows ships Segoe UI Emoji by default (used as a fallback)
Getting the data
Face data: download FER2013 from Kaggle, extract into data/fer2013/ so it matches data/fer2013/train/<class>/*.jpg and data/fer2013/test/<class>/*.jpg.
Emoji data: generate it — no download needed:
python src/emoji_dataset_generator.py
(Optional) Custom face classes (crying, thinking): collect your own samples via webcam:
python src/collect_custom_faces.py
Training
python src/train_face_model.py            # base 7-class face model
# OR, for the extended 9-class version:
python src/train_face_model_extended.py

python src/train_emoji_model.py
Before trusting the results, sanity-check the dataset and trained models:

python src/check_setup.py
Running it
python main.py                          # live webcam, display only
python main.py --save outputs/demo.mp4  # live webcam + save annotated video
python main.py --image path/to/img.jpg  # single-image mode
Known limitations
FER2013's accuracy ceiling is genuinely ~65–70% even in published research — it's a small, noisy, low-resolution dataset. That's a dataset constraint, not a modeling bug.
The face-vs-emoji router is a heuristic, not a learned classifier — very cluttered backgrounds can occasionally get misrouted.
Emoji predictions depend on the rendering font — a model trained on one font's glyph style may not perfectly generalize to a different platform's emoji design.
Some predictions will just be wrong sometimes. It's a classifier, not an oracle — confidence scores are shown for exactly this reason.
Possible next steps
Replace the edge-density router heuristic with a small learned "face vs. glyph" classifier
Train on a larger/cleaner face dataset (AffectNet, RAF-DB) for higher accuracy
Expand the custom-class pipeline to more emotions beyond crying/thinking
