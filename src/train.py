# train.py
import sys
import zipfile
import numpy as np
import tensorflow as tf
from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN = 10
EPOCHS_UNFROZEN = 10
LEARNING_RATE_FROZEN = 1e-3
LEARNING_RATE_UNFROZEN = 1e-5
VALIDATION_SPLIT = 0.2
AUTOTUNE = tf.data.AUTOTUNE


# ── Dataset ───────────────────────────────────────────────────────────────────
def load_dataset(root_dir: Path) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        root_dir,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        root_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    class_names = train_ds.class_names
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


# ── Augmentation layer ────────────────────────────────────────────────────────
def build_augmentation_layer() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ], name="augmentation")


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> tf.keras.Model:
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
        pooling="avg",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = build_augmentation_layer()(inputs, training=True)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)


# ── Training ──────────────────────────────────────────────────────────────────
def train_frozen(model: tf.keras.Model, train_ds, val_ds) -> tf.keras.callbacks.History:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_FROZEN),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            "best_model.keras", save_best_only=True, monitor="val_accuracy"
        ),
    ]
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FROZEN,
        callbacks=callbacks,
    )


def train_unfrozen(model: tf.keras.Model, train_ds, val_ds) -> tf.keras.callbacks.History:
    # Unfreeze top 20 layers of backbone
    model.layers[3].trainable = True
    for layer in model.layers[3].layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_UNFROZEN),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            "best_model.keras", save_best_only=True, monitor="val_accuracy"
        ),
    ]
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_UNFROZEN,
        callbacks=callbacks,
    )


# ── Save artifacts ────────────────────────────────────────────────────────────
def save_artifacts(model: tf.keras.Model, class_names: list[str], root_dir: Path) -> None:
    model.save("best_model.keras")
    
    with open("class_names.txt", "w") as f:
        f.write("\n".join(class_names))

    zip_path = "model_artifacts.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("best_model.keras")
        zf.write("class_names.txt")
    print(f"Artifacts saved to {zip_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python train.py <dataset_dir>")
        sys.exit(1)

    root_dir = Path(sys.argv[1])
    if not root_dir.is_dir():
        print(f"Error: {root_dir} is not a valid directory")
        sys.exit(1)

    print("Loading dataset...")
    train_ds, val_ds, class_names = load_dataset(root_dir)
    print(f"Classes: {class_names}")
    print(f"Training batches: {len(train_ds)}, Validation batches: {len(val_ds)}")

    print("\nBuilding model...")
    model = build_model(num_classes=len(class_names))
    model.summary()

    print("\nPhase 1: Training frozen backbone...")
    train_frozen(model, train_ds, val_ds)

    print("\nPhase 2: Fine-tuning unfrozen layers...")
    train_unfrozen(model, train_ds, val_ds)

    print("\nSaving artifacts...")
    save_artifacts(model, class_names, root_dir)

    print("\nEvaluating on validation set...")
    model = tf.keras.models.load_model("best_model.keras")
    loss, accuracy = model.evaluate(val_ds)
    print(f"Validation accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()