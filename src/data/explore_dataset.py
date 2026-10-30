
from pathlib import Path
from collections import Counter
from hashlib import sha256

from PIL import Image, UnidentifiedImageError

# Dataset paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT / "data" / "extracted" /
    "casting_data" / "casting_data"
)

TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"

CLASSES = ["def_front", "ok_front"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def inspect_split(split_name, split_dir):
    print(f"\n{'=' * 55}")
    print(f"{split_name.upper()} DATASET REPORT")
    print("=" * 55)

    total = 0
    class_counts = Counter()
    dimensions = Counter()
    invalid_images = []
    hashes = {}

    for class_name in CLASSES:
        class_dir = split_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in EXTENSIONS:
                continue

            try:
                with Image.open(image_path) as image:
                    image.verify()

                with Image.open(image_path) as image:
                    dimensions[image.size] += 1

                image_hash = sha256(
                    image_path.read_bytes()
                ).hexdigest()

                hashes[image_hash] = (
                    class_name,
                    image_path.name
                )

                class_counts[class_name] += 1
                total += 1

            except (
                OSError,
                UnidentifiedImageError,
                ValueError
            ) as error:
                invalid_images.append(
                    (str(image_path), str(error))
                )

    print(f"Total images: {total}")

    print("\nClass distribution:")
    for class_name in CLASSES:
        print(
            f"{class_name}: {class_counts[class_name]}"
        )

    print("\nImage dimensions:")
    for size, count in dimensions.most_common():
        print(f"{size}: {count} images")

    print(f"\nInvalid images: {len(invalid_images)}")

    if invalid_images:
        for path, error in invalid_images[:10]:
            print(f"Invalid: {path} | {error}")

    return {
        "total": total,
        "counts": class_counts,
        "hashes": hashes,
        "dimensions": dimensions,
        "invalid": invalid_images,
    }


def main():
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            "Dataset folders not found. Check extraction."
        )

    train = inspect_split("Training", TRAIN_DIR)
    test = inspect_split("Testing", TEST_DIR)

    # Check for exact duplicate files across train and test
    common_hashes = (
        set(train["hashes"]) & set(test["hashes"])
    )

    print(f"\n{'=' * 55}")
    print("DATA QUALITY SUMMARY")
    print("=" * 55)

    print("Total dataset images:", train["total"] + test["total"])
    print("Training images:", train["total"])
    print("Testing images:", test["total"])
    print("Exact duplicates across train/test:", len(common_hashes))

    print(
        "Total invalid images:",
        len(train["invalid"]) + len(test["invalid"])
    )

    if common_hashes:
        print("\nWARNING: Exact duplicate images found:")
        for image_hash in common_hashes:
            print(
                "Train:", train["hashes"][image_hash],
                "| Test:", test["hashes"][image_hash]
            )
    else:
        print("\nNo exact file duplicates across splits.")

    print("\nDataset exploration completed.")


if __name__ == "__main__":
    main()
