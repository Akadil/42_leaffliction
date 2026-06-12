import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)


# ── Load artifacts ────────────────────────────────────────────────────────────
def load_artifacts(artifacts_dir: Path) -> tuple[tf.keras.Model, list[str]]:
    model_path = artifacts_dir / "best_model.keras"
    class_names_path = artifacts_dir / "class_names.txt"

    if not model_path.exists():
        print(f"Error: model not found at {model_path}")
        sys.exit(1)
    if not class_names_path.exists():
        print(f"Error: class_names.txt not found at {class_names_path}")
        sys.exit(1)

    model = tf.keras.models.load_model(str(model_path))
    class_names = class_names_path.read_text().strip().splitlines()
    return model, class_names


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(model: tf.keras.Model, class_names: list[str], image_path: Path) -> tuple[str, float]:
    img = tf.io.read_file(str(image_path))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img_batch = tf.expand_dims(img, axis=0)  # (1, 224, 224, 3)

    predictions = model.predict(img_batch, verbose=0)
    idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][idx])
    return class_names[idx], confidence


# ── Display ───────────────────────────────────────────────────────────────────
def display(image_path: Path, predicted_class: str, confidence: float) -> None:
    img = plt.imread(str(image_path))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(
        f"=== DL classification ===\nClass predicted: {predicted_class} ({confidence:.1%})",
        fontsize=13,
        fontweight="bold",
        color="royalblue",
        pad=12,
    )
    plt.tight_layout()
    plt.savefig("prediction.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Prediction saved to prediction.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python predict.py <artifacts_dir> <image_path>")
        sys.exit(1)

    artifacts_dir = Path(sys.argv[1])
    image_path = Path(sys.argv[2])

    if not artifacts_dir.is_dir():
        print(f"Error: {artifacts_dir} is not a valid directory")
        sys.exit(1)
    if not image_path.is_file():
        print(f"Error: {image_path} is not a valid file")
        sys.exit(1)

    model, class_names = load_artifacts(artifacts_dir)
    predicted_class, confidence = predict(model, class_names, image_path)

    print(f"Predicted class : {predicted_class}")
    print(f"Confidence      : {confidence:.1%}")

    display(image_path, predicted_class, confidence)


if __name__ == "__main__":
    main()