# Augmentation.py
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageTransform


def augment_flip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def augment_rotate(img: Image.Image) -> Image.Image:
    return img.rotate(45, expand=True)


def augment_skew(img: Image.Image) -> Image.Image:
    w, h = img.size
    skew_factor = 0.3
    coeffs = (1, skew_factor, -skew_factor * h / 2, 0, 1, 0)
    return img.transform((w, h), ImageTransform.AffineTransform(coeffs))


def augment_shear(img: Image.Image) -> Image.Image:
    w, h = img.size
    shear_factor = 0.3
    coeffs = (1, shear_factor, 0, 0, 1, 0)
    return img.transform((w, h), ImageTransform.AffineTransform(coeffs))


def augment_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    margin = 0.1
    box = (int(w * margin), int(h * margin), int(w * (1 - margin)), int(h * (1 - margin)))
    return img.crop(box).resize((w, h), Image.LANCZOS)


def augment_distortion(img: Image.Image) -> Image.Image:
    w, h = img.size
    noise = np.random.randint(-10, 10, (h, w, len(img.getbands())), dtype=np.int16)
    arr = np.clip(np.array(img, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


AUGMENTATIONS: dict[str, callable] = {
    "Flip": augment_flip,
    "Rotate": augment_rotate,
    "Skew": augment_skew,
    "Shear": augment_shear,
    "Crop": augment_crop,
    "Distortion": augment_distortion,
}


def augment_image(image_path: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    stem = image_path.stem
    suffix = image_path.suffix

    for aug_name, aug_fn in AUGMENTATIONS.items():
        augmented = aug_fn(img)
        output_path = image_path.parent / f"{stem}_{aug_name}{suffix}"
        augmented.save(output_path)
        print(f"Saved: {output_path}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python Augmentation.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        print(f"Error: {image_path} is not a valid file")
        sys.exit(1)

    augment_image(image_path)


if __name__ == "__main__":
    main()