"""
Model definitions for both branches.

Face branch: transfer learning (MobileNetV2) — FER2013 is small (~35k images),
so training a deep CNN from scratch overfits badly. Freezing an ImageNet
backbone and fine-tuning a small head is the standard fix.

Emoji branch: a small CNN trained from scratch is fine here — the synthetic
dataset is easy (flat glyphs on simple backgrounds), effectively unlimited
in size, and the task is much less visually complex than real face photos,
so we don't need transfer learning's regularization benefit.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow.keras import layers, models

from config import FACE_EMOTIONS, FACE_INPUT_SIZE, EMOJI_EMOTIONS, EMOJI_INPUT_SIZE


def build_face_emotion_model(num_classes=None):
    if num_classes is None:
        num_classes = len(FACE_EMOTIONS)

    # MobileNetV2 expects RGB + >=32x32; FER2013 is 48x48 grayscale,
    # so we upsample + replicate channels going in.
    inputs = layers.Input(shape=(*FACE_INPUT_SIZE, 1))
    x = layers.Resizing(96, 96)(inputs)
    x = layers.Concatenate()([x, x, x])  # 1 -> 3 channels
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)  # MobileNetV2 preprocessing range

    base = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False  # freeze backbone first; unfreeze later for fine-tuning

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="face_emotion_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def unfreeze_for_finetuning(model, base, num_layers_to_unfreeze=30):
    """Call after initial head training converges — unfreeze the top N
    layers of the backbone and re-compile with a much lower LR so we
    don't destroy the pretrained weights."""
    base.trainable = True
    for layer in base.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_emoji_emotion_model():
    inputs = layers.Input(shape=(*EMOJI_INPUT_SIZE, 3))
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(EMOJI_EMOTIONS), activation="softmax")(x)

    model = models.Model(inputs, outputs, name="emoji_emotion_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model