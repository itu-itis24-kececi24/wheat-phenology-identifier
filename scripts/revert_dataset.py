import os
import csv
import shutil

REPORT_PATH = r"C:\Users\ASUS\Desktop\Ders\blg521\bad_images_report.csv"
BACKUP_DIR = r"C:\Users\ASUS\Desktop\Ders\blg521\data\bad_images"

def revert():
    if not os.path.exists(REPORT_PATH):
        print(f"Error: Quality report {REPORT_PATH} not found.")
        return

    with open(REPORT_PATH, mode="r", encoding="utf-8") as f_csv:
        reader = csv.DictReader(f_csv)
        rows = list(reader)

    print(f"Reverting {len(rows)} images to their original folders...")

    reverted_count = 0
    errors = 0

    for row in rows:
        rel_folder = row["Folder"]
        filename = row["Filename"]
        orig_path = row["FullPath"]
        
        # Path where it is currently backed up
        current_backup_path = os.path.join(BACKUP_DIR, rel_folder, filename)

        if not os.path.exists(current_backup_path):
            continue

        try:
            # Recreate original folder if it got deleted
            os.makedirs(os.path.dirname(orig_path), exist_ok=True)
            # Move back
            shutil.move(current_backup_path, orig_path)
            reverted_count += 1
        except Exception as e:
            print(f"Error reverting {filename}: {e}")
            errors += 1

    print("\n" + "="*50)
    print("               REVERT COMPLETED")
    print("="*50)
    print(f"Successfully reverted: {reverted_count} images.")
    if errors > 0:
        print(f"Encountered errors:   {errors} files.")
    print("="*50)

    # Clean up empty backup directory tree
    try:
        if os.path.exists(BACKUP_DIR):
            shutil.rmtree(BACKUP_DIR)
            print("Cleaned up the temporary bad_images backup folder.")
    except Exception as e:
        print(f"Error cleaning up backup folder: {e}")

if __name__ == "__main__":
    revert()
