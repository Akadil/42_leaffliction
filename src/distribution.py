# Distribution.py
import sys
import matplotlib.pyplot as plt
from pathlib import Path


def fetch_class_counts(root_dir: Path) -> dict[str, int]:
    # iterate through all subdirectories of root_dir and count # of images
    return {
        subdir.name: len(list(subdir.glob("*.[jJ][pP][gG]")))
        for subdir in sorted(root_dir.iterdir())
        if subdir.is_dir()
    }


def plot_distribution(class_counts: dict[str, int], plant_name: str) -> None:
    labels = list(class_counts.keys())
    values = list(class_counts.values())

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{plant_name} class distribution")

    ax_pie.pie(values, labels=labels, autopct="%1.1f%%")

    ax_bar.bar(labels, values, color=["blue", "red", "green", "purple"][:len(labels)])
    ax_bar.set_xticks(range(len(labels)))
    ax_bar.set_xticklabels(labels, rotation=15, ha="right")

    # plt.tight_layout()
    # plt.show()
    plt.savefig(f"{plant_name}_distribution.png")
    plt.close()


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python Distribution.py <directory>")
        sys.exit(1)

    root_dir = Path(sys.argv[1])
    if not root_dir.is_dir():
        print(f"Error: {root_dir} is not a valid directory")
        sys.exit(1)

    class_counts = fetch_class_counts(root_dir)
    if not class_counts:
        print(f"Error: no subdirectories found in {root_dir}")
        sys.exit(1)

    print(f"\n\n{class_counts}\n\n")
    plot_distribution(class_counts, root_dir.name)


if __name__ == "__main__":
    main()