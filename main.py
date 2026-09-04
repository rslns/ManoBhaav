"""
Real-time webcam demo for Manobhaav.

Usage:
    python main.py
    python main.py --save output.mp4
    python main.py --image path/to/img.jpg
"""

import cv2
import argparse
import time
import sys
import os

from src.unified_inference import UnifiedEmotionPredictor
from src.config import INFER_EVERY_N_FRAMES

# Add src to Python's module search path BEFORE importing project modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)


def draw_overlay(frame, result, fps):
    branch, emotion, conf, bbox = (
        result["branch"],
        result["emotion"],
        result["confidence"],
        result["bbox"]
    )

    if branch == "face" and bbox is not None:
        x, y, w, h = bbox

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        label = f"{emotion} ({conf * 100:.1f}%)"

        cv2.putText(
            frame,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    elif branch == "emoji":
        label = f"EMOJI: {emotion} ({conf * 100:.1f}%)"

        cv2.putText(
            frame,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 200, 0),
            2
        )

    else:
        cv2.putText(
            frame,
            "No face or emoji detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    return frame


def run_webcam(save_path=None):
    predictor = UnifiedEmotionPredictor()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    writer = None

    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            save_path,
            fourcc,
            20.0,
            (w, h)
        )

    frame_count = 0

    last_result = {
        "branch": None,
        "emotion": None,
        "confidence": 0.0,
        "bbox": None
    }

    prev_time = time.time()

    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame_count % INFER_EVERY_N_FRAMES == 0:
            last_result = predictor.predict(frame)

        frame_count += 1

        now = time.time()

        fps = 1.0 / max(
            now - prev_time,
            1e-6
        )

        prev_time = now

        frame = draw_overlay(
            frame,
            last_result,
            fps
        )

        cv2.imshow(
            "Manobhaav - Emotion Recognition",
            frame
        )

        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    if writer is not None:
        writer.release()
        print(f"Saved annotated video -> {save_path}")

    cv2.destroyAllWindows()


def run_single_image(image_path):
    predictor = UnifiedEmotionPredictor()

    frame = cv2.imread(image_path)

    if frame is None:
        raise FileNotFoundError(image_path)

    result = predictor.predict(frame)

    frame = draw_overlay(
        frame,
        result,
        fps=0.0
    )

    out_path = "output_" + os.path.basename(image_path)

    cv2.imwrite(
        out_path,
        frame
    )

    print(
        f"branch={result['branch']} "
        f"emotion={result['emotion']} "
        f"confidence={result['confidence']:.3f}"
    )

    print(
        f"Saved annotated image -> {out_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save output .mp4"
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Run on a single image instead of webcam"
    )

    args = parser.parse_args()

    if args.image:
        run_single_image(args.image)
    else:
        run_webcam(
            save_path=args.save
        )
