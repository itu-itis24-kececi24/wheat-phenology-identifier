import os
import csv
from PIL import Image

FULL_REPORT_PATH = r"C:\Users\ASUS\Desktop\Ders\blg521\bad_images_report_full_backup.csv"
TRUNCATED_DIR = r"C:\Users\ASUS\Desktop\Ders\blg521\data\truncated_images"

def repair_images():
    if not os.path.exists(FULL_REPORT_PATH):
        print(f"Error: Full quality report {FULL_REPORT_PATH} not found.")
        return

    with open(FULL_REPORT_PATH, mode="r", encoding="utf-8") as f_csv:
        reader = csv.DictReader(f_csv)
        rows = list(reader)

    # Filter for only TRUNCATED images (JPEG EOI missing)
    truncated_rows = [row for row in rows if row["IssueType"] == "TRUNCATED"]

    print(f"Total truncated images to repair: {len(truncated_rows)}")

    repaired_count = 0
    verification_passed = 0
    errors = 0

    for row in truncated_rows:
        rel_folder = row["Folder"]
        filename = row["Filename"]
        
        # Current path where the truncated images are located
        filepath = os.path.join(TRUNCATED_DIR, rel_folder, filename)

        if not os.path.exists(filepath):
            continue

        try:
            # 1. Append the missing EOI marker (\xFF\xD9) to the file
            with open(filepath, "ab") as f:
                f.write(b"\xff\xd9")
            
            repaired_count += 1

            # 2. Verify that PIL can now open and load the repaired image
            try:
                img = Image.open(filepath)
                img.load()
                verification_passed += 1
            except Exception as load_err:
                print(f"Verification failed for {filename} after repair: {load_err}")

        except Exception as e:
            print(f"Error repairing {filename}: {e}")
            errors += 1

    print("\n" + "="*50)
    print("               REPAIR COMPLETED")
    print("="*50)
    print(f"Successfully repaired (appended EOI): {repaired_count}")
    print(f"Verified readable by Pillow:          {verification_passed} ({(verification_passed/repaired_count)*100:.2f}%)")
    if errors > 0:
        print(f"Encountered write errors:             {errors} files.")
    print("="*50)

if __name__ == "__main__":
    repair_images()
