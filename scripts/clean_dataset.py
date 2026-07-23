import os
import csv
import shutil

REPORT_PATH = r"C:\Users\ASUS\Desktop\Ders\blg521\bad_images_report.csv"
BACKUP_DIR = r"C:\Users\ASUS\Desktop\Ders\blg521\data\bad_images"

def clean_dataset(action="move"):
    if not os.path.exists(REPORT_PATH):
        print(f"Error: Quality report {REPORT_PATH} not found. Run verify_images.py first.")
        return

    with open(REPORT_PATH, mode="r", encoding="utf-8") as f_csv:
        reader = csv.DictReader(f_csv)
        rows = list(reader)

    if not rows:
        print("No bad images found in the report.")
        return

    print(f"Found {len(rows)} problematic images to process.")
    print(f"Selected action: {action.upper()}")

    moved_count = 0
    deleted_count = 0
    errors = 0

    for idx, row in enumerate(rows, 1):
        rel_folder = row["Folder"]
        filename = row["Filename"]
        full_path = row["FullPath"]

        if not os.path.exists(full_path):
            # Already moved or deleted
            continue

        try:
            if action == "move":
                # Create corresponding folder structure in backup dir
                dest_dir = os.path.join(BACKUP_DIR, rel_folder)
                os.makedirs(dest_dir, exist_ok=True)
                
                dest_path = os.path.join(dest_dir, filename)
                shutil.move(full_path, dest_path)
                moved_count += 1
            elif action == "delete":
                os.remove(full_path)
                deleted_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            errors += 1

    print("\n" + "="*50)
    print("               CLEANUP COMPLETED")
    print("="*50)
    if action == "move":
        print(f"Successfully moved:  {moved_count} images to backup directory.")
        print(f"Backup directory:    {BACKUP_DIR}")
    else:
        print(f"Successfully deleted: {deleted_count} images.")
    if errors > 0:
        print(f"Encountered errors:   {errors} files.")
    print("="*50)

if __name__ == "__main__":
    print("This script will isolate the bad images identified by verify_images.py.")
    print("Options:")
    print("1. Move bad images to a backup folder (Recommended)")
    print("2. Delete bad images permanently")
    
    # We default to moving them to backup for safety
    clean_dataset(action="move")
