"""Move non-augmented images from Test_Set_Folder and Validation_Set_Folder
into the matching class folders under plants_dataset, skipping duplicates."""

import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
SOURCE_FOLDERS = [BASE_DIR / "Test_Set_Folder", BASE_DIR / "Validation_Set_Folder"]
DEST_FOLDER = BASE_DIR / "plants_dataset"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def remove_aug_files_from_dest():
    removed = 0
    for image_path in DEST_FOLDER.rglob("aug_*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
            image_path.unlink()
            removed += 1
    return removed


def main():
    moved, skipped_duplicate, skipped_aug = 0, 0, 0

    for source_folder in SOURCE_FOLDERS:
        for class_dir in sorted(source_folder.iterdir()):
            if not class_dir.is_dir():
                continue

            dest_class_dir = DEST_FOLDER / class_dir.name
            dest_class_dir.mkdir(parents=True, exist_ok=True)

            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                if image_path.name.lower().startswith("aug_"):
                    skipped_aug += 1
                    continue

                dest_path = dest_class_dir / image_path.name
                if dest_path.exists():
                    skipped_duplicate += 1
                    print(f"[skip-duplicate] {image_path}")
                    continue

                shutil.move(str(image_path), str(dest_path))
                moved += 1

    removed_aug_from_dest = remove_aug_files_from_dest()

    for source_folder in SOURCE_FOLDERS:
        if source_folder.exists():
            shutil.rmtree(source_folder)

    print("\n--- Summary ---")
    print(f"Moved:                     {moved}")
    print(f"Skipped duplicates:        {skipped_duplicate}")
    print(f"Skipped aug_* (in source): {skipped_aug}")
    print(f"Removed aug_* (in dest):   {removed_aug_from_dest}")
    print(f"Deleted source folders:    {[str(f) for f in SOURCE_FOLDERS]}")


if __name__ == "__main__":
    main()
