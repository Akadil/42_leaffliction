# Transformation.py
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def get_mask(blurred: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 20, 20])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def get_roi(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    roi = img.copy()
    green_overlay = np.zeros_like(img)
    green_overlay[mask > 0] = [0, 255, 0]
    roi = cv2.addWeighted(roi, 1.0, green_overlay, 0.5, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return roi


def analyze_object(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    analyzed = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(analyzed, [largest], -1, (255, 0, 255), 2)
        if len(largest) >= 5:
            ellipse = cv2.fitEllipse(largest)
            cv2.ellipse(analyzed, ellipse, (255, 0, 255), 2)
        m = cv2.moments(largest)
        if m["m00"] != 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            cv2.line(analyzed, (cx - 20, cy), (cx + 20, cy), (0, 0, 255), 2)
            cv2.line(analyzed, (cx, cy - 20), (cx, cy + 20), (0, 0, 255), 2)
    return analyzed


def get_pseudolandmarks(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    landmarks = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return landmarks

    largest = max(contours, key=cv2.contourArea)
    pts = largest[:, 0, :]  # shape (N, 2)

    # Sample evenly along the contour
    n_points = 20
    indices = np.linspace(0, len(pts) - 1, n_points, dtype=int)
    sampled = pts[indices]

    # Split into top/bottom/left/right by position
    mid_x = pts[:, 0].mean()
    mid_y = pts[:, 1].mean()

    for pt in sampled:
        if pt[1] < mid_y and abs(pt[0] - mid_x) < mid_x * 0.5:
            color = (0, 0, 255)    # top → red
        elif pt[1] > mid_y and abs(pt[0] - mid_x) < mid_x * 0.5:
            color = (0, 165, 255)  # bottom → orange
        elif pt[0] < mid_x:
            color = (255, 255, 0)  # left → yellow
        else:
            color = (255, 0, 0)    # right → blue
        cv2.circle(landmarks, tuple(pt), 5, color, -1)

    return landmarks


def plot_color_histogram(img: np.ndarray, ax: plt.Axes) -> None:
    channels = {
        "blue": (img[:, :, 0], "blue"),
        "green": (img[:, :, 1], "green"),
        "red": (img[:, :, 2], "red"),
    }
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    channels.update({
        "hue": (hsv[:, :, 0], "purple"),
        "saturation": (hsv[:, :, 1], "orange"),
        "value": (hsv[:, :, 2], "brown"),
    })
    for name, (channel, color) in channels.items():
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = hist / hist.sum() * 100
        ax.plot(hist, color=color, label=name, linewidth=0.8)

    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Proportion of pixels (%)")
    ax.legend(fontsize=7)
    ax.set_title("Color histogram")


def display_transformations(image_path: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    mask = get_mask(blurred)
    roi = get_roi(img, mask)
    analyzed = analyze_object(img, mask)
    landmarks = get_pseudolandmarks(img, mask)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(image_path.name)

    pairs = [
        ("Original", cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
        ("Gaussian blur", cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)),
        ("Mask", mask),
        ("ROI objects", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)),
        ("Analyze object", cv2.cvtColor(analyzed, cv2.COLOR_BGR2RGB)),
        ("Pseudolandmarks", cv2.cvtColor(landmarks, cv2.COLOR_BGR2RGB)),
    ]

    axes_flat = axes.flatten()
    for ax, (title, image) in zip(axes_flat, pairs):
        cmap = "gray" if len(image.shape) == 2 else None
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("on")

    plot_color_histogram(img, axes_flat[6])
    axes_flat[7].axis("off")

    plt.tight_layout()
    plt.savefig(f"{image_path.stem}_transformations.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {image_path.stem}_transformations.png")


def save_transformations(src: Path, dst: Path, use_mask: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    images = list(src.glob("*.[jJ][pP][gG]"))
    for image_path in images:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        blurred = cv2.GaussianBlur(img, (15, 15), 0)
        mask = get_mask(blurred)
        transformations = {
            "gaussian_blur": blurred,
            "mask": cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            "roi": get_roi(img, mask),
            "analyzed": analyze_object(img, mask),
            "landmarks": get_pseudolandmarks(img, mask),
        }
        if use_mask:
            transformations = {"mask": transformations["mask"]}

        for name, transformed in transformations.items():
            out_path = dst / f"{image_path.stem}_{name}.jpg"
            cv2.imwrite(str(out_path), transformed)
        print(f"Processed: {image_path.name}")


def main() -> None:
    if len(sys.argv) == 2:
        image_path = Path(sys.argv[1])
        if not image_path.is_file():
            print(f"Error: {image_path} is not a valid file")
            sys.exit(1)
        display_transformations(image_path)

    elif "-src" in sys.argv:
        src = Path(sys.argv[sys.argv.index("-src") + 1])
        dst = Path(sys.argv[sys.argv.index("-dst") + 1]) if "-dst" in sys.argv else src / "transformed"
        use_mask = "-mask" in sys.argv
        save_transformations(src, dst, use_mask)

    else:
        print("Usage:")
        print("  python Transformation.py <image_path>")
        print("  python Transformation.py -src <dir> -dst <dir> [-mask]")
        sys.exit(1)


if __name__ == "__main__":
    main()