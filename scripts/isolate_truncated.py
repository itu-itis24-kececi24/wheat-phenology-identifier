import os
import csv
import shutil

FULL_REPORT_PATH = r"C:\Users\ASUS\Desktop\Ders\blg521\bad_images_report_full_backup.csv"
TRUNCATED_DIR = r"C:\Users\ASUS\Desktop\Ders\blg521\data\truncated_images"

def isolate_truncated():
    if not os.path.exists(FULL_REPORT_PATH):
        print(f"Error: Original full report {FULL_REPORT_PATH} not found.")
        return

    with open(FULL_REPORT_PATH, mode="r", encoding="utf-8") as f_csv:
        reader = csv.DictReader(f_csv)
        rows = list(reader)

    # Filter for only TRUNCATED images (JPEG EOI missing)
    truncated_rows = [row for row in rows if row["IssueType"] == "TRUNCATED"]

    print(f"Total problematic images in full backup: {len(rows)}")
    print(f"Truncated images to isolate: {len(truncated_rows)}")

    moved_count = 0
    errors = 0

    for row in truncated_rows:
        rel_folder = row["Folder"]
        filename = row["Filename"]
        full_path = row["FullPath"]

        if not os.path.exists(full_path):
            # Already moved or deleted
            continue

        try:
            # Create corresponding folder structure in truncated folder
            dest_dir = os.path.join(TRUNCATED_DIR, rel_folder)
            os.makedirs(dest_dir, exist_ok=True)
            
            dest_path = os.path.join(dest_dir, filename)
            shutil.move(full_path, dest_path)
            moved_count += 1
        except Exception as e:
            print(f"Error moving truncated image {filename}: {e}")
            errors += 1

    print("\n" + "="*50)
    print("               TRUNCATED ISOLATION COMPLETED")
    print("="*50)
    print(f"Successfully moved:  {moved_count} truncated/corrupted images.")
    print(f"Isolated images path: {TRUNCATED_DIR}")
    if errors > 0:
        print(f"Encountered errors:  {errors} files.")
    print("="*50)

if __name__ == "__main__":
    isolate_truncated()
