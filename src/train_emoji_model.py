"""
Trains the emoji-emotion branch on the synthetically generated dataset.
Run emoji_dataset_generator.py first to create data/emoji_dataset/<class>/*.png
"""
import os
import tensorflow as tf

from config import EMOJI_DATASET_DIR, EMOJI_MODEL_PATH, EMOJI_INPUT_SIZE, EMOJI_EMOTIONS, MODELS_DIR
from models import build_emoji_emotion_model

BATCH_SIZE = 32
EPOCHS = 15


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        EMOJI_DATASET_DIR,
        labels="inferred",
        label_mode="int",
        class_names=EMOJI_EMOTIONS,
        color_mode="rgb",
        image_size=EMOJI_INPUT_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        subset="training",
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        EMOJI_DATASET_DIR,
        labels="inferred",
        label_mode="int",
        class_names=EMOJI_EMOTIONS,
        color_mode="rgb",
        image_size=EMOJI_INPUT_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        subset="validation",
        seed=42,
    )
    return (
        train_ds.prefetch(tf.data.AUTOTUNE),
        val_ds.prefetch(tf.data.AUTOTUNE),
    )


def main():
    if not os.path.isdir(EMOJI_DATASET_DIR) or not os.listdir(EMOJI_DATASET_DIR):
        raise RuntimeError(
            "Emoji dataset not found. Run: python src/emoji_dataset_generator.py first."
        )
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_ds, val_ds = load_datasets()
    model = build_emoji_emotion_model()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=4, restore_best_weights=True
    )
    model.fit(train_ds, validation_data=val_ds,
              epochs=EPOCHS, callbacks=[early_stop])

    model.save(EMOJI_MODEL_PATH)
    print(f"Saved emoji emotion model -> {EMOJI_MODEL_PATH}")


if __name__ == "__main__":
    main()
