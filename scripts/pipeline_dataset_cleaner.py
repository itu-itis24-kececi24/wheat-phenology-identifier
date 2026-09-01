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

SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']

BLUR_PRESETS = {
    "conservative": 15.0,  # Severe/total global blur only
    "medium": 45.0,        # Obvious blur & soft focus
    "aggressive": 100.0    # Mild out-of-focus & soft crop images
}

def analyze_and_clean_image(args):
    """
    Analyzes an image file for size, JPEG EOI integrity & repair, container decoding validity,
    dark/bright/contrast anomalies, monochrome drops, and global + local grid blur (excluding center pole column).
    """
    rel_dir, filename, data_dir, do_repair, blur_threshold, check_monochrome, check_local_blur = args
    filepath = os.path.join(data_dir, rel_dir, filename)
    rel_path = os.path.join(rel_dir, filename) if rel_dir != "." else filename
    
    w, h, mean_v, std_v, lap_v = 0, 0, 0.0, 0.0, 0.0
    was_repaired = False

    # 1. File size check
    try:
        size_bytes = os.path.getsize(filepath)
        if size_bytes == 0:
            return rel_dir, filename, filepath, rel_path, "CORRUPT", "0 bytes file size", w, h, mean_v, std_v, lap_v, was_repaired
    except Exception as e:
        return rel_dir, filename, filepath, rel_path, "ERROR", f"Cannot check file size: {e}", w, h, mean_v, std_v, lap_v, was_repaired

    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return rel_dir, filename, filepath, rel_path, "SKIP", f"Non-image extension: {ext}", w, h, mean_v, std_v, lap_v, was_repaired

    # 2. Fast JPEG End-of-Image (EOI) check and in-place repair
    is_jpeg = ext in ['.jpg', '.jpeg']
    if is_jpeg and do_repair:
        try:
            with open(filepath, 'rb') as fh:
                fh.seek(-2, 2)
                last_two = fh.read()
                if last_two != b'\xff\xd9':
                    with open(filepath, 'ab') as fh_write:
                        fh_write.write(b'\xff\xd9')
                    was_repaired = True
        except Exception as e:
            return rel_dir, filename, filepath, rel_path, "CORRUPT", f"Failed byte-level check/repair: {e}", w, h, mean_v, std_v, lap_v, was_repaired

    # 3. Decoding validation with Pillow
    try:
        with Image.open(filepath) as img_raw:
            orig_w, orig_h = img_raw.size
            
            # Check for Monochrome / Grayscale drops if requested
            is_monochrome = False
            if check_monochrome and img_raw.mode in ['RGB', 'RGBA']:
                img_rgb = img_raw.resize((64, 64), Image.Resampling.NEAREST)
                arr_rgb = np.array(img_rgb, dtype=np.float32)
                if arr_rgb.ndim == 3 and arr_rgb.shape[2] >= 3:
                    color_diff = np.mean(np.abs(arr_rgb[:, :, 0] - arr_rgb[:, :, 1]) +
                                         np.abs(arr_rgb[:, :, 1] - arr_rgb[:, :, 2]))
                    if color_diff < 3.0:
                        is_monochrome = True

            img_g = img_raw.convert('L').resize((256, 256), Image.Resampling.BILINEAR)
            img_arr = np.array(img_g, dtype=np.float32)

    except Exception as e:
        return rel_dir, filename, filepath, rel_path, "CORRUPT", f"Cannot open/decode image: {e}", w, h, mean_v, std_v, lap_v, was_repaired

    # 4. Global Quality Analysis
    try:
        mean_val = float(np.mean(img_arr))
        std_val = float(np.std(img_arr))

        if is_monochrome:
            return rel_dir, filename, filepath, rel_path, "MONOCHROME", "Grayscale / B&W camera sensor drop", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired

        # Check for solid color / extreme flat contrast / black / white
        if std_val < 1.0:
            if mean_val < 10.0:
                return rel_dir, filename, filepath, rel_path, "DARK", f"Completely black (mean: {mean_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired
            elif mean_val > 245.0:
                return rel_dir, filename, filepath, rel_path, "WHITE", f"Completely white (mean: {mean_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired
            else:
                return rel_dir, filename, filepath, rel_path, "LOW_CONTRAST", f"Uniform solid color (mean: {mean_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired

        if std_val < 5.0:
            return rel_dir, filename, filepath, rel_path, "LOW_CONTRAST", f"Extremely low contrast (std: {std_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired

        if mean_val < 5.0:
            return rel_dir, filename, filepath, rel_path, "DARK", f"Extremely dark/underexposed (mean: {mean_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired
        if mean_val > 250.0:
            return rel_dir, filename, filepath, rel_path, "WHITE", f"Extremely bright/overexposed (mean: {mean_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired

        # 5. Smart Global & Local Blur Check (4x4 Grid Patch Analysis)
        gh, gw = img_arr.shape
        grid_rows, grid_cols = 4, 4
        rh, rw = gh // grid_rows, gw // grid_cols

        patch_vars = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                if c in [1, 2]:
                    continue
                p = img_arr[r*rh:(r+1)*rh, c*rw:(c+1)*rw]
                lap_p = (p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1] - 4.0 * p[1:-1, 1:-1])
                patch_vars.append(float(np.var(lap_p)))

        patch_vars.sort()
        avg_side_var = float(np.mean(patch_vars)) if patch_vars else 0.0
        min_patch_var = patch_vars[0] if patch_vars else 0.0
        p25_patch_var = patch_vars[len(patch_vars) // 4] if patch_vars else 0.0

        if avg_side_var < blur_threshold:
            return rel_dir, filename, filepath, rel_path, "BLURRED", f"Global blur (avg Laplacian: {avg_side_var:.1f} < threshold: {blur_threshold:.1f})", orig_w, orig_h, mean_val, std_val, avg_side_var, was_repaired

        if check_local_blur:
            local_thresh = max(10.0, blur_threshold * 0.45)
            if p25_patch_var < local_thresh or min_patch_var < (local_thresh * 0.5):
                return rel_dir, filename, filepath, rel_path, "LOCAL_BLUR", f"Local smearing/motion blur (worst patch: {min_patch_var:.1f}, 25th percentile: {p25_patch_var:.1f} < threshold: {local_thresh:.1f})", orig_w, orig_h, mean_val, std_val, avg_side_var, was_repaired

        if was_repaired:
            return rel_dir, filename, filepath, rel_path, "REPAIRED", "JPEG EOI appended, valid image", orig_w, orig_h, mean_val, std_val, avg_side_var, was_repaired

        return rel_dir, filename, filepath, rel_path, "OK", "Valid image", orig_w, orig_h, mean_val, std_val, avg_side_var, was_repaired

    except Exception as e:
        return rel_dir, filename, filepath, rel_path, "ERROR", f"Error during quality analysis: {e}", orig_w, orig_h, 0.0, 0.0, 0.0, was_repaired

def main():
    parser = argparse.ArgumentParser(description="Full Dataset High-Speed Image Cleaner Pipeline")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to main dataset root directory")
    parser.add_argument("--mode", "-m", choices=["conservative", "medium", "aggressive"], default="medium", help="Preset sensitivity level (default: medium)")
    parser.add_argument("--blur-threshold", "-b", type=float, default=None, help="Custom blur threshold on 256x256 grid (overrides --mode)")
    parser.add_argument("--local-blur", action="store_true", default=True, help="Enable 4x4 spatial grid analysis for local/partial motion blurs")
    parser.add_argument("--no-local-blur", action="store_false", dest="local_blur", help="Disable local patch blur detection")
    parser.add_argument("--monochrome", action="store_true", help="Flag B&W / Grayscale / IR-mode images as bad")
    parser.add_argument("--no-repair", action="store_true", help="Disable automatic in-place JPEG repair")
    parser.add_argument("--workers", "-w", type=int, default=os.cpu_count(), help="Number of parallel worker processes")
    args = parser.parse_args()

    data_path = os.path.abspath(args.data_dir)
    if not os.path.exists(data_path):
        print(f"Error: Path '{data_path}' does not exist.")
        sys.exit(1)

    blur_thresh = args.blur_threshold if args.blur_threshold is not None else BLUR_PRESETS[args.mode]
    do_repair = not args.no_repair

    bad_dir = os.path.join(data_path, "bad_images")
    truncated_dir = os.path.join(data_path, "truncated_images")

    print("=" * 68)
    print("        FULL DATASET AGGRESSIVE & SMART CLEANUP PIPELINE")
    print("=" * 68)
    print(f"Dataset Root Directory: {data_path}")
    print(f"Sensitivity Mode:       {args.mode.upper()}")
    print(f"Blur Threshold:         {blur_thresh:.1f}")
    print(f"Local Patch Blur Check: {args.local_blur} (4x4 Grid Analysis)")
    print(f"Flag Monochrome/B&W:    {args.monochrome}")
    print(f"Auto JPEG EOI Repair:   {do_repair}")
    print(f"Target 'bad' Folder:    {bad_dir}")
    print(f"Target 'corrupt' Folder:{truncated_dir}")
    print(f"Worker Processes:       {args.workers}")
    print("=" * 68)

    # 1. Gathering files to check
    print("Gathering files to check...")
    files_to_check = []
    
    for root, dirs, files in os.walk(data_path):
        if "bad_images" in root or "truncated_images" in root or "corrupt_images" in root:
            continue
            
        rel_dir = os.path.relpath(root, data_path)
        
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                files_to_check.append((rel_dir, f, data_path, do_repair, blur_thresh, args.monochrome, args.local_blur))

    total_files = len(files_to_check)
    print(f"Found {total_files} candidate image files to process.")
    if total_files == 0:
        print("No files to process. Exiting.")
        sys.exit(0)

    # 2. Run Parallel Analysis
    start_time = time.time()
    results = []
    
    print("Running quality analysis and repair pipeline...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        mapped_results = executor.map(analyze_and_clean_image, files_to_check, chunksize=150)
        
        for idx, res in enumerate(mapped_results, 1):
            results.append(res)
            if idx % 2000 == 0 or idx == total_files:
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                rem = total_files - idx
                eta = rem / rate if rate > 0 else 0
                print(f"Processed {idx}/{total_files} files ({(idx/total_files)*100:.1f}%) | Speed: {rate:.1f} img/sec | ETA: {eta:.1f}s")

    elapsed_time = time.time() - start_time
    print(f"\nPipeline processing completed in {elapsed_time:.2f} seconds.")

    # 3. Handle moving files and compiling the report
    print("\nSorting files based on analysis...")
    csv_rows = []
    repaired_count = 0
    moved_bad_count = 0
    moved_corrupt_count = 0
    ok_count = 0
    issue_breakdown = {}

    for res in results:
        rel_dir, filename, filepath, rel_path, status, detail, w, h, mean_v, std_v, lap_v, was_repaired = res
        
        if status == "OK":
            ok_count += 1
            
        elif status == "REPAIRED":
            repaired_count += 1
            issue_breakdown["REPAIRED (Kept in place)"] = issue_breakdown.get("REPAIRED (Kept in place)", 0) + 1
            
        elif status in ["BLURRED", "LOCAL_BLUR", "LOW_CONTRAST", "DARK", "WHITE", "MONOCHROME"]:
            dest_folder = os.path.join(bad_dir, rel_dir) if rel_dir != "." else bad_dir
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
            dest_folder = os.path.join(truncated_dir, rel_dir) if rel_dir != "." else truncated_dir
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

    # Clean up empty subdirectories
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

    # Save CSV Report
    report_path = os.path.join(data_path, "bad_images_report.csv")
    with open(report_path, mode="w", newline="", encoding="utf-8") as f_csv:
        fieldnames = ["Folder", "Filename", "OriginalPath", "Type", "Details"]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + "=" * 68)
    print("                    DATASET CLEANUP SUMMARY")
    print("=" * 68)
    print(f"Total Scanned Files:         {total_files}")
    print(f"Valid Clean Images (Kept):   {ok_count}")
    print(f"Repaired JPEGs (Kept):       {repaired_count}")
    print(f"Moved to 'bad_images':       {moved_bad_count}")
    print(f"Moved to 'truncated_images': {moved_corrupt_count}")
    print("-" * 68)
    print("Issue Breakdown:")
    for issue_type, count in issue_breakdown.items():
        print(f"  - {issue_type:28}: {count}")
    print(f"\nDetailed CSV Report saved to: {report_path}")
    print("=" * 68)

if __name__ == "__main__":
    main()
