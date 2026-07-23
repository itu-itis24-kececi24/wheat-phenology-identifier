import os
import csv
import shutil

REPORT_PATH = r"C:\Users\ASUS\Desktop\Ders\blg521\bad_images_report.csv"
NEW_REPORT_PATH = r"C:\Users\ASUS\Desktop\Ders\blg521\bad_images_quality_report.csv"
BACKUP_DIR = r"C:\Users\ASUS\Desktop\Ders\blg521\data\bad_images"

def clean_selective():
    if not os.path.exists(REPORT_PATH):
        print(f"Error: Quality report {REPORT_PATH} not found.")
        return

    with open(REPORT_PATH, mode="r", encoding="utf-8") as f_csv:
        reader = csv.DictReader(f_csv)
        rows = list(reader)

    # Filter for only BLURRED or LOW_CONTRAST issues
    # Leave TRUNCATED files untouched (they remain in their original directories)
    target_issues = ["BLURRED", "LOW_CONTRAST"]
    filtered_rows = [row for row in rows if row["IssueType"] in target_issues]

    print(f"Total problematic images in original report: {len(rows)}")
    print(f"Images to isolate (BLURRED or LOW_CONTRAST): {len(filtered_rows)}")
    print(f"Images to keep in place (TRUNCATED, etc.): {len(rows) - len(filtered_rows)}")

    moved_count = 0
    errors = 0
    moved_rows = []

    for row in filtered_rows:
        rel_folder = row["Folder"]
        filename = row["Filename"]
        full_path = row["FullPath"]

        if not os.path.exists(full_path):
            print(f"Warning: File not found at original location: {full_path}")
            continue

        try:
            # Create corresponding folder structure in backup dir
            dest_dir = os.path.join(BACKUP_DIR, rel_folder)
            os.makedirs(dest_dir, exist_ok=True)
            
            dest_path = os.path.join(dest_dir, filename)
            shutil.move(full_path, dest_path)
            moved_count += 1
            moved_rows.append(row)
        except Exception as e:
            print(f"Error moving {filename}: {e}")
            errors += 1

    # Write the new filtered report containing only the isolated images
    if moved_rows:
        csv_headers = ["Folder", "Filename", "FullPath", "IssueType", "Details", "Width", "Height", "MeanBrightness", "StdDev", "LaplacianVar"]
        try:
            with open(NEW_REPORT_PATH, mode="w", newline="", encoding="utf-8") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=csv_headers)
                writer.writeheader()
                for row in moved_rows:
                    writer.writerow(row)
            print(f"\nFiltered report saved to: {NEW_REPORT_PATH}")
            
            # Replace the old report with the new one to keep it clean,
            # but keep a backup of the original full report first just in case
            backup_original = r"C:\Users\ASUS\Desktop\Ders\blg521\bad_images_report_full_backup.csv"
            shutil.copy2(REPORT_PATH, backup_original)
            shutil.move(NEW_REPORT_PATH, REPORT_PATH)
            print(f"Updated main report {REPORT_PATH} (original backed up to {backup_original})")
        except Exception as e:
            print(f"Error saving filtered report: {e}")

    print("\n" + "="*50)
    print("               SELECTIVE CLEANUP COMPLETED")
    print("="*50)
    print(f"Successfully moved:  {moved_count} quality-deficient images.")
    print(f"Remaining in place: {len(rows) - moved_count} files (including truncated ones).")
    print(f"Isolated images path: {BACKUP_DIR}")
    if errors > 0:
        print(f"Encountered errors:  {errors} files.")
    print("="*50)

if __name__ == "__main__":
    clean_selective()
