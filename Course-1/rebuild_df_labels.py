"""Rebuilds df_labels.csv from the images actually present in plants_dataset."""
import csv
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "plants_dataset"
CLASSNAME_FILE = DATASET_DIR / "classname.txt"
OUTPUT_CSV = DATASET_DIR / "df_labels.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# classname.txt has no spaces (e.g. "peperchili") but some real folders do
# (e.g. "peper chili"), so map class name -> actual folder on disk.
FOLDER_OVERRIDES = {
    "peperchili": "peper chili",
    "sweetpotatoes": "sweet potatoes",
}


def main():
    class_names = CLASSNAME_FILE.read_text().splitlines()
    class_names = [name.strip() for name in class_names if name.strip()]

    rows = []
    for category, class_name in enumerate(class_names):
        folder_name = FOLDER_OVERRIDES.get(class_name, class_name)
        folder_path = DATASET_DIR / folder_name

        image_paths = sorted(
            p for p in folder_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            relative_path = f"{folder_name}/{image_path.name}"
            rows.append((relative_path, category))

        print(f"{class_name}: {len(image_paths)} images")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image:FILE", "category"])
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
