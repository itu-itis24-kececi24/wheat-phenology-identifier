#!/usr/bin/env python3
import os
import sys
import csv
import time
import shutil
import argparse
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Default Relative Paths
DEFAULT_DATA_DIR = "./data"
REPORT_NAME = "dataset_cleanup_report.csv"

def check_and_repair_image(args):
    """
    Analyzes, repairs (if needed), and validates a single image.
    Optimized for maximum speed (NEAREST resampling, 256x256 draft mode).
    """
    rel_dir, filename, data_dir = args
    filepath = os.path.join(data_dir, rel_dir, filename)
    
    w, h, mean_v, std_v, lap_v = 0, 0, 0.0, 0.0, 0.0
    was_repaired = False
    
    # 1. Size check
    try:
        size_bytes = os.path.getsize(filepath)
        if size_bytes == 0:
            return rel_dir, filename, filepath, "CORRUPT", "File size is 0 bytes", w, h, mean_v, std_v, lap_v, was_repaired
    except Exception as e:
        return rel_dir, filename, filepath, "ERROR", f"Could not get file size: {e}", w, h, mean_v, std_v, lap_v, was_repaired

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']:
        return rel_dir, filename, filepath, "NOT_IMAGE", f"Non-image extension: {ext}", w, h, mean_v, std_v, lap_v, was_repaired

    # 2. Fast JPEG End-of-Image (EOI) check and in-place repair
    is_jpeg = ext in ['.jpg', '.jpeg']
    if is_jpeg:
        try:
            with open(filepath, 'rb') as fh:
                fh.seek(-2, 2)
                last_two = fh.read()
                if last_two != b'\xff\xd9':
                    # Attempt in-place repair by appending EOI marker
                    with open(filepath, 'ab') as fh_write:
                        fh_write.write(b'\xff\xd9')
                    was_repaired = True
        except Exception as e:
            return rel_dir, filename, filepath, "CORRUPT", f"Failed reading/writing bytes: {e}", w, h, mean_v, std_v, lap_v, was_repaired

    # 3. Integrity validation: check if Pillow can decode the image container
    img = None
    try:
        img = Image.open(filepath)
        orig_w, orig_h = img.size
    except Exception as e:
        return rel_dir, filename, filepath, "CORRUPT", f"Cannot open image container: {e}", w, h, mean_v, std_v, lap_v, was_repaired

    try:
        # High Speed: Draft downsample to 256x256 reduces JPEG decoding CPU load by ~4x
        img.draft('L', (256, 256))
        img.load()
    except Exception as e:
        return rel_dir, filename, filepath, "CORRUPT", f"Unrecoverable decoding error: {e}", orig_w, orig_h, mean_v, std_v, lap_v, was_repaired

    # 4. Fast Quality verification
    try:
        # High Speed: Use NEAREST resampling (up to 10x faster than BILINEAR)
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.Resampling.NEAREST)
        else:
            img = img.convert('L')

        img_arr = np.array(img, dtype=np.float32)
        mean_val = float(np.mean(img_arr))
        std_val = float(np.std(img_arr))

        # Check for extremely flat contrast or solid block colors
        if std_val < 1.0:
            if mean_val < 10.0:
                return rel_dir, filename, filepath, "DARK", f"Completely black/dark (mean: {mean_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired
            elif mean_val > 245.0:
                return rel_dir, filename, filepath, "WHITE", f"Completely white/blank (mean: {mean_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired
            else:
                return rel_dir, filename, filepath, "LOW_CONTRAST", f"Uniform solid color (mean: {mean_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired

        if std_val < 5.0:
            return rel_dir, filename, filepath, "LOW_CONTRAST", f"Extremely low contrast (std: {std_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired

        if mean_val < 5.0:
            return rel_dir, filename, filepath, "DARK", f"Extremely dark/underexposed (mean: {mean_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired
        if mean_val > 250.0:
            return rel_dir, filename, filepath, "WHITE", f"Extremely bright/overexposed (mean: {mean_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired

        # Blur check: Laplacian variance on the 256x256 grid
        laplacian = (
            img_arr[1:-1, 2:] + img_arr[1:-1, :-2] +
            img_arr[2:, 1:-1] + img_arr[:-2, 1:-1] -
            4.0 * img_arr[1:-1, 1:-1]
        )
        laplacian_var = float(np.var(laplacian))

        # Scaled blur threshold for 256x256 resolution is ~12.5 (equivalent to 50.0 on a 512x512 grid)
        if laplacian_var < 12.5:
            return rel_dir, filename, filepath, "BLURRED", f"Very blurry (Laplacian var: {laplacian_var:.2f})", orig_w, orig_h, mean_val, std_val, laplacian_var, was_repaired

        if was_repaired:
            return rel_dir, filename, filepath, "REPAIRED", "JPEG EOI appended, fully verified", orig_w, orig_h, mean_val, std_val, laplacian_var, was_repaired

        return rel_dir, filename, filepath, "OK", "Valid image", orig_w, orig_h, mean_val, std_val, laplacian_var, was_repaired

    except Exception as e:
        return rel_dir, filename, filepath, "ERROR", f"Error during quality analysis: {e}", orig_w, orig_h, 0.0, 0.0, 0.0, was_repaired

def main():
    parser = argparse.ArgumentParser(description="High-Speed Dataset Quality and Integrity Pipeline")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Path to 'data' directory containing stations")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of concurrent processes (defaults to all cores)")
    args = parser.parse_args()

    data_path = os.path.abspath(args.data_dir)
    bad_dir = os.path.join(data_path, "bad_images")
    truncated_dir = os.path.join(data_path, "truncated_images")

    if not os.path.exists(data_path):
        print(f"Error: Path '{data_path}' does not exist.")
        sys.exit(1)

    print("="*60)
    print("        HIGH-SPEED WHEAT DATASET CLEANUP PIPELINE")
    print("="*60)
    print(f"Dataset root:           {data_path}")
    print(f"Target 'bad' folder:    {bad_dir}")
    print(f"Target 'corrupt' folder:{truncated_dir}")
    print(f"Worker processes:       {args.workers}")
    print("="*60)

    # 1. Gathering files to check
    print("Gathering files to check...")
    files_to_check = []
    
    for root, dirs, files in os.walk(data_path):
        if "bad_images" in root or "truncated_images" in root:
            continue
            
        rel_dir = os.path.relpath(root, data_path)
        
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in ['.csv', '.json', '.txt', '.md', '.xlsx']:
                continue
            files_to_check.append((rel_dir, f, data_path))

    total_files = len(files_to_check)
    print(f"Found {total_files} candidate image files to process.")
    if total_files == 0:
        print("No files to process. Exiting.")
        sys.exit(0)

    # 2. Run Parallel analysis
    start_time = time.time()
    results = []
    
    print(f"Running quality analysis and repair pipeline...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        mapped_results = executor.map(check_and_repair_image, files_to_check, chunksize=150)
        
        for idx, res in enumerate(mapped_results, 1):
            results.append(res)
            if idx % 5000 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                rem = total_files - idx
                eta = rem / rate if rate > 0 else 0
                print(f"Processed {idx}/{total_files} files ({(idx/total_files)*100:.1f}%) | Speed: {rate:.1f} img/sec | ETA: {eta:.1f}s")

    elapsed_time = time.time() - start_time
    print(f"\nPipeline processing completed in {elapsed_time:.2f} seconds.")

    # 3. Handle moving files and compiling the report
    print("\nSorting files based on analysis...")
    
    csv_rows = []
    repaired_and_kept_count = 0
    moved_bad_count = 0
    moved_corrupt_count = 0
    ok_count = 0
    
    issue_breakdown = {}

    for res in results:
        rel_dir, filename, filepath, status, detail, w, h, mean_v, std_v, lap_v, was_repaired = res
        
        if status == "OK":
            ok_count += 1
            
        elif status == "REPAIRED":
            repaired_and_kept_count += 1
            issue_breakdown["REPAIRED (Kept in place)"] = issue_breakdown.get("REPAIRED (Kept in place)", 0) + 1
            
        elif status in ["BLURRED", "LOW_CONTRAST", "DARK", "WHITE"]:
            dest_folder = os.path.join(bad_dir, rel_dir)
            os.makedirs(dest_folder, exist_ok=True)
            
            try:
                shutil.move(filepath, os.path.join(dest_folder, filename))
                moved_bad_count += 1
                issue_breakdown[status] = issue_breakdown.get(status, 0) + 1
                csv_rows.append({
                    "Folder": rel_dir,
                    "Filename": filename,
                    "OriginalPath": filepath,
                    "Type": "QUALITY_ISSUE",
                    "Details": f"{status}: {detail}"
                })
            except Exception as e:
                print(f"Error moving {filename} to bad_images: {e}")
                
        elif status in ["CORRUPT", "ERROR"]:
            dest_folder = os.path.join(truncated_dir, rel_dir)
            os.makedirs(dest_folder, exist_ok=True)
            
            try:
                shutil.move(filepath, os.path.join(dest_folder, filename))
                moved_corrupt_count += 1
                issue_breakdown["CORRUPT (Unrecoverable)"] = issue_breakdown.get("CORRUPT (Unrecoverable)", 0) + 1
                csv_rows.append({
                    "Folder": rel_dir,
                    "Filename": filename,
                    "OriginalPath": filepath,
                    "Type": "CORRUPTED",
                    "Details": detail
                })
            except Exception as e:
                print(f"Error moving {filename} to truncated_images: {e}")

    # 4. Clean up empty folders
    for target in [bad_dir, truncated_dir]:
        if os.path.exists(target):
            for root, dirs, files in os.walk(target, topdown=False):
                for d in dirs:
                    dir_path = os.path.join(root, d)
                    try:
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                    except Exception:
                        pass

    # 5. Save report CSV
    report_path = os.path.join(data_path, REPORT_NAME)
    try:
        with open(report_path, mode="w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=["Folder", "Filename", "OriginalPath", "Type", "Details"])
            writer.writeheader()
            for row in sorted(csv_rows, key=lambda x: (x["Folder"], x["Filename"])):
                writer.writerow(row)
        print(f"Report saved to: {report_path}")
    except Exception as e:
        print(f"Error saving CSV report: {e}")

    # 6. Display final summary
    print("\n" + "="*50)
    print("                FINAL PIPELINE SUMMARY")
    print("="*50)
    print(f"Total files processed:        {total_files}")
    print(f"Originally clean images (OK): {ok_count} ({(ok_count/total_files)*100:.2f}%)")
    print(f"Successfully repaired JPEGs:  {repaired_and_kept_count} ({(repaired_and_kept_count/total_files)*100:.2f}%)")
    print(f"Moved to data/bad_images:     {moved_bad_count} ({(moved_bad_count/total_files)*100:.2f}%)")
    print(f"Moved to data/truncated_images: {moved_corrupt_count} ({(moved_corrupt_count/total_files)*100:.2f}%)")
    print("-"*50)
    print("Breakdown of actions:")
    for key, val in sorted(issue_breakdown.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {key}: {val}")
    print("="*50)

if __name__ == "__main__":
    main()
