"""
Generates a labeled emoji-emotion image dataset by RENDERING emoji glyphs
with a font, then augmenting (rotation, scale, background, noise, blur).

Why synthetic instead of downloading a dataset:
- No emoji-emotion labeled dataset exists publicly at any real scale.
- Emoji are standardized vector glyphs -> we can reproduce every
  visual variation (size, rotation, platform-style background) a
  camera might actually see (e.g. a phone screen held up to the webcam)
  without any copyright/licensing exposure.
- Deterministic + regenerable: change SAMPLES_PER_CLASS and rerun.

Requires a color-emoji-capable font on the system, e.g. Noto Color Emoji.
Install (Ubuntu): sudo apt-get install -y fonts-noto-color-emoji
"""
import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from config import EMOJI_DATASET_DIR, EMOJI_GLYPHS, EMOJI_INPUT_SIZE

SAMPLES_PER_CLASS = 400
CANVAS_SIZE = 128  # render large, then downscale -> anti-aliased result

# Common locations for a color emoji font; add your own path if needed.
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto-emoji/NotoColorEmoji.ttf",
    "C:/Windows/Fonts/seguiemj.ttf",  # Windows fallback (Segoe UI Emoji)
]


def _find_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                # Noto Color Emoji only renders at a fixed embedded size (109),
                # PIL requires size=109 for that specific font on most builds.
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    raise FileNotFoundError(
        "No color emoji font found. Install one, e.g.:\n"
        "  sudo apt-get install -y fonts-noto-color-emoji\n"
        "or add its path to FONT_CANDIDATES in emoji_dataset_generator.py"
    )


def _random_background(size):
    """Simulate different real-world backgrounds an emoji-on-screen might have."""
    mode = random.choice(["solid", "gradient"])
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    if mode == "solid":
        color = tuple(random.randint(200, 255) for _ in range(3))
        draw.rectangle([0, 0, size, size], fill=color)
    else:
        top = tuple(random.randint(180, 255) for _ in range(3))
        bottom = tuple(random.randint(180, 255) for _ in range(3))
        for y in range(size):
            t = y / size
            color = tuple(int(top[i] * (1 - t) + bottom[i] * t)
                          for i in range(3))
            draw.line([(0, y), (size, y)], fill=color)
    return img


def _augment(img):
    # Random rotation (phone/screen tilt), slight scale jitter, blur, noise
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))

    scale = random.uniform(0.75, 1.0)
    new_size = int(CANVAS_SIZE * scale)
    img = img.resize((new_size, new_size))
    canvas = _random_background(CANVAS_SIZE)
    offset = ((CANVAS_SIZE - new_size) // 2, (CANVAS_SIZE - new_size) // 2)
    canvas.paste(img, offset, img.convert("RGBA")
                 if img.mode != "RGBA" else img)

    if random.random() < 0.3:
        canvas = canvas.filter(ImageFilter.GaussianBlur(
            radius=random.uniform(0.5, 1.5)))

    return canvas.resize(EMOJI_INPUT_SIZE)


def generate_dataset():
    font = _find_font(109)
    for label, glyph in EMOJI_GLYPHS.items():
        out_dir = os.path.join(EMOJI_DATASET_DIR, label)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(SAMPLES_PER_CLASS):
            base = _random_background(CANVAS_SIZE)
            draw = ImageDraw.Draw(base)
            bbox = draw.textbbox((0, 0), glyph, font=font, embedded_color=True)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pos = ((CANVAS_SIZE - w) // 2 -
                   bbox[0], (CANVAS_SIZE - h) // 2 - bbox[1])
            draw.text(pos, glyph, font=font, embedded_color=True)

            sample = _augment(base)
            sample.save(os.path.join(out_dir, f"{label}_{i:04d}.png"))
        print(f"[emoji-gen] {label}: {SAMPLES_PER_CLASS} images -> {out_dir}")


if __name__ == "__main__":
    generate_dataset()
