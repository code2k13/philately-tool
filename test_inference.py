import random
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# =========================
# CONFIG
# =========================
MODEL_PATH = Path("model.pt")
INPUT_DIR  = Path("./test_input")
OUTPUT_DIR = Path("./inference_outputs")

NUM_RANDOM_IMAGES = 50
CONF_THRESHOLD = 0.4
OVERLAY_ALPHA = 0.25

OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# SAFE IMAGE LOADER
# =========================
def load_image_safe(path: Path):
    """
    Loads JPG / PNG / WEBP / BMP safely.
    - Removes alpha channel
    - Normalizes to RGB
    - Converts to OpenCV BGR
    """
    img = Image.open(path).convert("RGB")
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

# =========================
# PICK RANDOM IMAGES
# =========================
images = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
    images.extend(INPUT_DIR.glob(ext))

if not images:
    raise RuntimeError(f"No images found in {INPUT_DIR}")

sample_images = random.sample(
    images,
    min(NUM_RANDOM_IMAGES, len(images))
)

print(f"Running inference on {len(sample_images)} images")

# =========================
# RUN INFERENCE
# =========================
for img_path in sample_images:

    # Load image safely (PNG alpha FIX)
    img = load_image_safe(img_path)

    # Run inference on the actual array
    results = model(
        img,
        conf=CONF_THRESHOLD,
        device=0,
        verbose=False
    )

    r = results[0]

    # This image is now guaranteed to match box coordinates
    img = img.copy()

    if r.boxes is not None and len(r.boxes) > 0:

        overlay = img.copy()

        # =========================
        # FILL PHASE
        # =========================
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),  # Blue
                -1
            )

        # Blend overlay
        img = cv2.addWeighted(
            overlay,
            OVERLAY_ALPHA,
            img,
            1 - OVERLAY_ALPHA,
            0
        )

        # =========================
        # BORDER + LABEL PHASE
        # =========================
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label_name = r.names[cls_id]

            # Border
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Label
            label_text = f"{label_name} {conf:.2f}"
            cv2.putText(
                img,
                label_text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    # =========================
    # SAVE OUTPUT
    # =========================
    out_path = OUTPUT_DIR / f"{img_path.stem}_pred.png"
    cv2.imwrite(str(out_path), img)
    print(f"Saved: {out_path}")

print("Inference complete.")
