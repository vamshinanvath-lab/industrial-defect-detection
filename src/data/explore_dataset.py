
from pathlib import Path
from collections import Counter
from PIL import Image
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "extracted"
    / "casting_data"
    / "casting_data"
)

SPLITS = ["train", "test"]
CLASSES = ["def_front", "ok_front"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def calculate_hash(image_path):
    sha256 = hashlib.sha256()

    with image_path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def inspect_split(split):
    split_path = DATASET_ROOT / split

    total = 0
    invalid = []
    dimensions = Counter()
    hashes = {}

    print(f"\n--- {split.upper()} DATASET ---")

    for class_name in CLASSES:
        class_path = split_path / class_name

        if not class_path.exists():
            print(f"Missing folder: {class_path}")
            continue

        image_files = [
            p for p in class_path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"{class_name}: {len(image_files)} images")
        total += len(image_files)

        for image_path in image_files:
            try:
                with Image.open(image_path) as image:
                    image.verify()

                with Image.open(image_path) as image:
                    dimensions[image.size] += 1

                file_hash = calculate_hash(image_path)
                hashes.setdefault(file_hash, []).append(str(image_path))

            except Exception as error:
                invalid.append((str(image_path), str(error)))

    print(f"Total images: {total}")
    print(f"Image dimensions: {dict(dimensions)}")
    print(f"Invalid images: {len(invalid)}")
    print(f"Duplicate groups within split: {sum(1 for v in hashes.values() if len(v) > 1)}")

    for path, error in invalid[:10]:
        print(f"INVALID: {path} | {error}")

    return hashes


def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_ROOT}"
        )

    train_hashes = inspect_split("train")
    test_hashes = inspect_split("test")

    overlap = set(train_hashes) & set(test_hashes)

    print("\n--- TRAIN/TEST DUPLICATE CHECK ---")
    print(f"Matching image hashes across splits: {len(overlap)}")

    print("\nDataset exploration completed.")


if __name__ == "__main__":
    main()
