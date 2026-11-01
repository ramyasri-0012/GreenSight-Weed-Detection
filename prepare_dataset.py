from pathlib import Path
import random
import shutil

# Original downloaded dataset
SOURCE = Path(r"C:\Users\Ramyasri\Downloads\kaggle\agri_data\data")

# Your project's dataset folder
PROJECT = Path(__file__).resolve().parent
DEST = PROJECT / "dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {SOURCE}")

    # Find images that have matching annotation files
    pairs = []
    for image in SOURCE.iterdir():
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
            label = image.with_suffix(".txt")
            if label.is_file():
                pairs.append((image, label))

    if len(pairs) < 2:
        raise RuntimeError(
            f"Found only {len(pairs)} image-label pairs. "
            "Check the SOURCE path and annotation filenames."
        )

    random.seed(42)
    random.shuffle(pairs)

    split_index = max(1, int(len(pairs) * 0.8))
    splits = {
        "train": pairs[:split_index],
        "val": pairs[split_index:],
    }

    for split, items in splits.items():
        for folder in ("images", "labels"):
            (DEST / folder / split).mkdir(parents=True, exist_ok=True)

        for image, label in items:
            shutil.copy2(image, DEST / "images" / split / image.name)
            shutil.copy2(label, DEST / "labels" / split / label.name)

        print(f"{split}: copied {len(items)} image-label pairs")

    print(f"\nDataset prepared at: {DEST}")
    print("Original downloaded files were not changed.")

if __name__ == "__main__":
    main()
