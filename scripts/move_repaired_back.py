import os
import csv
import shutil
from PIL import Image

FULL_REPORT_PATH = r"C:\Users\ASUS\Desktop\Ders\blg521\bad_images_report_full_backup.csv"
TRUNCATED_DIR = r"C:\Users\ASUS\Desktop\Ders\blg521\data\truncated_images"

def move_repaired_back():
    if not os.path.exists(FULL_REPORT_PATH):
        print(f"Error: Full quality report {FULL_REPORT_PATH} not found.")
        return

    with open(FULL_REPORT_PATH, mode="r", encoding="utf-8") as f_csv:
        reader = csv.DictReader(f_csv)
        rows = list(reader)

    # Filter for only TRUNCATED images (JPEG EOI missing originally)
    truncated_rows = [row for row in rows if row["IssueType"] == "TRUNCATED"]

    print(f"Scanning {len(truncated_rows)} originally truncated images to find repaired ones...")

    moved_count = 0
    left_count = 0
    errors = 0

    for row in truncated_rows:
        rel_folder = row["Folder"]
        filename = row["Filename"]
        orig_path = row["FullPath"]
        
        # Path where the file is currently located (in truncated_images)
        current_path = os.path.join(TRUNCATED_DIR, rel_folder, filename)

        if not os.path.exists(current_path):
            continue

        # Test if it is readable now
        is_readable = False
        try:
            img = Image.open(current_path)
            img.load()
            is_readable = True
        except Exception:
            pass

        if is_readable:
            try:
                # Recreate the target directory structure
                os.makedirs(os.path.dirname(orig_path), exist_ok=True)
                # Move back
                shutil.move(current_path, orig_path)
                moved_count += 1
            except Exception as e:
                print(f"Error moving {filename} back: {e}")
                errors += 1
        else:
            left_count += 1

    print("\n" + "="*50)
    print("               RESTORE COMPLETED")
    print("="*50)
    print(f"Successfully moved back: {moved_count} repaired images.")
    print(f"Left isolated:           {left_count} unrecoverable images.")
    if errors > 0:
        print(f"Encountered errors:      {errors} files.")
    print("="*50)

if __name__ == "__main__":
    move_repaired_back()
