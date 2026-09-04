"""
Trains the face-emotion branch on FER2013.

Expects FER_DATASET_DIR laid out as (this is the standard Kaggle FER2013
"folder" format — see README for how to get/convert it):

    data/fer2013/train/angry/*.jpg
    data/fer2013/train/happy/*.jpg
    ...
    data/fer2013/test/angry/*.jpg
    ...

Two-phase training (standard transfer-learning pattern):
  Phase 1: backbone frozen, train the new head only (fast, stabilizes head weights)
  Phase 2: unfreeze top of backbone, fine-tune end-to-end at a low LR
"""
import os
import tensorflow as tf

from config import FER_DATASET_DIR, FACE_MODEL_PATH, FACE_INPUT_SIZE, FACE_EMOTIONS, MODELS_DIR
from models import build_face_emotion_model, unfreeze_for_finetuning

BATCH_SIZE = 64
HEAD_EPOCHS = 8
FINETUNE_EPOCHS = 6


def load_datasets():
    train_dir = os.path.join(FER_DATASET_DIR, "train")
    test_dir = os.path.join(FER_DATASET_DIR, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        class_names=FACE_EMOTIONS,   # enforces consistent class index order
        color_mode="grayscale",
        image_size=FACE_INPUT_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        class_names=FACE_EMOTIONS,
        color_mode="grayscale",
        image_size=FACE_INPUT_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Light augmentation on the training set only — FER2013 is small,
    # and faces vary in lighting/pose, so this meaningfully helps generalization.
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_ds, val_ds = load_datasets()
    model, base = build_face_emotion_model()

    print("=== Phase 1: training head (backbone frozen) ===")
    model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS)

    print("=== Phase 2: fine-tuning top of backbone ===")
    model = unfreeze_for_finetuning(model, base, num_layers_to_unfreeze=30)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=FINETUNE_EPOCHS, callbacks=[early_stop],
    )

    model.save(FACE_MODEL_PATH)
    print(f"Saved face emotion model -> {FACE_MODEL_PATH}")


if __name__ == "__main__":
    main()
