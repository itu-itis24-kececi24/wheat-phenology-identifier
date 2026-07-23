import os
import csv
import time
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_DIR = r"C:\Users\ASUS\Desktop\Ders\blg521\data"
REPORT_PATH = r"C:\Users\ASUS\Desktop\Ders\blg521\bad_images_report.csv"

def check_single_image(filepath_info):
    """
    Checks a single image for integrity and quality issues.
    This function runs in a child process.
    """
    rel_dir, filename = filepath_info
    filepath = os.path.join(DATA_DIR, rel_dir, filename)
    
    # Standard outputs in case of error
    w, h, mean_v, std_v, lap_v = 0, 0, 0.0, 0.0, 0.0
    
    try:
        size_bytes = os.path.getsize(filepath)
        if size_bytes == 0:
            return rel_dir, filename, filepath, "ZERO_SIZE", "File size is 0 bytes", w, h, mean_v, std_v, lap_v
    except Exception as e:
        return rel_dir, filename, filepath, "ERROR", f"Could not get file size: {e}", w, h, mean_v, std_v, lap_v

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']:
        return rel_dir, filename, filepath, "NOT_IMAGE", f"Non-image extension: {ext}", w, h, mean_v, std_v, lap_v

    # 1. Fast check for truncated JPEG by reading last two bytes
    is_jpeg = ext in ['.jpg', '.jpeg']
    if is_jpeg:
        try:
            with open(filepath, 'rb') as fh:
                fh.seek(-2, 2)
                last_two = fh.read()
                if last_two != b'\xff\xd9':
                    # Check if there is some trailing junk or if it is actually truncated
                    # Let's flag it as TRUNCATED for further verification
                    return rel_dir, filename, filepath, "TRUNCATED", "JPEG missing standard EOI marker \\xFF\\xD9", w, h, mean_v, std_v, lap_v
        except Exception as e:
            return rel_dir, filename, filepath, "CORRUPT", f"Failed reading file bytes: {e}", w, h, mean_v, std_v, lap_v

    # 2. Integrity check: Try to open the file container
    img = None
    try:
        img = Image.open(filepath)
    except Exception as e:
        return rel_dir, filename, filepath, "CORRUPT", f"Cannot open image container: {e}", w, h, mean_v, std_v, lap_v

    # Get original dimensions before draft downsampling changes them
    orig_w, orig_h = img.size

    # 3. Fast load with downsampling using draft mode
    try:
        img.draft('L', (512, 512))
        img.load()
    except Exception as e:
        return rel_dir, filename, filepath, "TRUNCATED", f"Failed decoding image data: {e}", orig_w, orig_h, mean_v, std_v, lap_v

    # Ensure image is in L (grayscale) mode and resized to 512x512 for consistent analysis
    try:
        if img.size != (512, 512):
            img = img.resize((512, 512), Image.Resampling.BILINEAR)
        else:
            img = img.convert('L')

        img_arr = np.array(img, dtype=np.float32)
        mean_val = float(np.mean(img_arr))
        std_val = float(np.std(img_arr))

        # Flat colors / Low contrast / Black or White check
        if std_val < 1.0:
            if mean_val < 10.0:
                return rel_dir, filename, filepath, "DARK", f"Completely black/dark (mean: {mean_val:.2f}, std: {std_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0
            elif mean_val > 245.0:
                return rel_dir, filename, filepath, "WHITE", f"Completely white/blank (mean: {mean_val:.2f}, std: {std_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0
            else:
                return rel_dir, filename, filepath, "LOW_CONTRAST", f"Uniform solid color (mean: {mean_val:.2f}, std: {std_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0

        if std_val < 5.0:
            return rel_dir, filename, filepath, "LOW_CONTRAST", f"Extremely low contrast (std: {std_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0

        if mean_val < 5.0:
            return rel_dir, filename, filepath, "DARK", f"Extremely dark/underexposed (mean: {mean_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0
        if mean_val > 250.0:
            return rel_dir, filename, filepath, "WHITE", f"Extremely bright/overexposed (mean: {mean_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0

        # Blur check: Laplacian variance on 512x512 array
        # 3x3 Laplacian: L(x, y) = I(x+1, y) + I(x-1, y) + I(x, y+1) + I(x, y-1) - 4*I(x, y)
        laplacian = (
            img_arr[1:-1, 2:] + img_arr[1:-1, :-2] +
            img_arr[2:, 1:-1] + img_arr[:-2, 1:-1] -
            4.0 * img_arr[1:-1, 1:-1]
        )
        laplacian_var = float(np.var(laplacian))

        # Very blurry image check
        if laplacian_var < 50.0:
            return rel_dir, filename, filepath, "BLURRED", f"Very blurry (Laplacian variance: {laplacian_var:.2f})", orig_w, orig_h, mean_val, std_val, laplacian_var

        return rel_dir, filename, filepath, "OK", "Valid image", orig_w, orig_h, mean_val, std_val, laplacian_var

    except Exception as e:
        return rel_dir, filename, filepath, "ERROR", f"Error during quality analysis: {e}", orig_w, orig_h, 0.0, 0.0, 0.0

def run_scan():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory {DATA_DIR} does not exist.")
        return

    print("Gathering files to scan...")
    files_to_check = []
    
    # Traverse directories to build task list
    for root, dirs, files in os.walk(DATA_DIR):
        rel_dir = os.path.relpath(root, DATA_DIR)
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            # Skip non-images
            if ext in ['.csv', '.json', '.txt', '.md', '.xlsx']:
                continue
            files_to_check.append((rel_dir, f))

    total_files = len(files_to_check)
    print(f"Found {total_files} candidate image files to check.")
    
    if total_files == 0:
        print("No image files to check. Exiting.")
        return

    # Start parallel scanning
    issues = []
    ok_count = 0
    issue_counts = {}
    
    start_time = time.time()
    
    # We use os.cpu_count() or leave it default (which uses all cores)
    cpu_cores = os.cpu_count() or 4
    print(f"Starting parallel scan using {cpu_cores} worker processes...")
    
    # ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        # We use map with a chunksize to minimize IPC overhead
        results = executor.map(check_single_image, files_to_check, chunksize=100)
        
        for idx, res in enumerate(results, 1):
            rel_dir, filename, filepath, status, detail, w, h, mean_v, std_v, lap_v = res
            
            if status in ["BLURRED", "LOW_CONTRAST"]:
                issues.append({
                    "Folder": rel_dir,
                    "Filename": filename,
                    "FullPath": filepath,
                    "IssueType": status,
                    "Details": detail,
                    "Width": w,
                    "Height": h,
                    "MeanBrightness": f"{mean_v:.2f}",
                    "StdDev": f"{std_v:.2f}",
                    "LaplacianVar": f"{lap_v:.2f}"
                })
                issue_counts[status] = issue_counts.get(status, 0) + 1
            else:
                ok_count += 1
                
            if idx % 2000 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                rem_files = total_files - idx
                eta = rem_files / rate if rate > 0 else 0
                print(f"Scanned {idx}/{total_files} files ({(idx/total_files)*100:.1f}%) | Speed: {rate:.1f} img/sec | ETA: {eta:.1f}s")

    elapsed_time = time.time() - start_time
    print(f"\nScan completed in {elapsed_time:.2f} seconds! (Average speed: {total_files/elapsed_time:.1f} img/sec)")

    # Write report to CSV
    csv_headers = ["Folder", "Filename", "FullPath", "IssueType", "Details", "Width", "Height", "MeanBrightness", "StdDev", "LaplacianVar"]
    try:
        with open(REPORT_PATH, mode="w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=csv_headers)
            writer.writeheader()
            for issue in sorted(issues, key=lambda x: (x["Folder"], x["Filename"])):
                writer.writerow(issue)
        print(f"Report successfully saved to: {REPORT_PATH}")
    except Exception as e:
        print(f"Error saving report to CSV: {e}")

    # Display scan summary
    print("\n" + "="*50)
    print("                SCAN SUMMARY REPORT")
    print("="*50)
    print(f"Total files scanned:       {total_files}")
    print(f"Valid images (OK):         {ok_count} ({(ok_count/total_files)*100:.2f}%)")
    print(f"Problematic files:         {len(issues)} ({(len(issues)/total_files)*100:.2f}%)")
    print("-"*50)
    print("Breakdown of Issues:")
    for status, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {status}: {count}")
    print("="*50)

if __name__ == "__main__":
    run_scan()
