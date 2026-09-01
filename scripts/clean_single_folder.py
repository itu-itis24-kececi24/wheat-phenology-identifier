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

# Blur Sensitivity Presets (scaled for 256x256 spatial grid)
BLUR_PRESETS = {
    "conservative": 15.0,  # Catches only severe/total global blur
    "medium": 45.0,        # Catches obvious blur & soft focus
    "aggressive": 100.0    # Catches mild out-of-focus & soft crop images
}

def analyze_and_clean_file(args):
    """
    Analyzes an image file for:
    1. Size and JPEG EOI integrity (auto-repair if missing).
    2. Container decoding validity.
    3. Visual Quality:
       - Dark / Black images
       - Overexposed / White images
       - Uniform solid color / Low contrast
       - Monochrome / Grayscale sensor drops
       - Global & Local Blur (Patch Grid Analysis)
         - Divides image into a 4x4 spatial grid (16 patches).
         - Ignores center pole column so striped poles don't mask blur.
         - Checks worst patches to catch LOCAL smearing, wind motion, or partial lens blur.
    """
    filepath, filename, target_dir, do_repair, blur_threshold, check_monochrome, check_local_blur = args
    rel_path = os.path.relpath(filepath, target_dir)
    
    w, h, mean_v, std_v, lap_v = 0, 0, 0.0, 0.0, 0.0
    was_repaired = False

    # 1. File size check
    try:
        size_bytes = os.path.getsize(filepath)
        if size_bytes == 0:
            return rel_path, filename, filepath, "CORRUPT", "0 bytes file size", w, h, mean_v, std_v, lap_v, was_repaired
    except Exception as e:
        return rel_path, filename, filepath, "ERROR", f"Cannot check file size: {e}", w, h, mean_v, std_v, lap_v, was_repaired

    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return rel_path, filename, filepath, "SKIP", f"Non-image extension: {ext}", w, h, mean_v, std_v, lap_v, was_repaired

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
            return rel_path, filename, filepath, "CORRUPT", f"Failed byte-level check/repair: {e}", w, h, mean_v, std_v, lap_v, was_repaired

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

            # Resize to 256x256 grayscale using BILINEAR to smooth resampling artifacts
            img_g = img_raw.convert('L').resize((256, 256), Image.Resampling.BILINEAR)
            img_arr = np.array(img_g, dtype=np.float32)

    except Exception as e:
        return rel_path, filename, filepath, "CORRUPT", f"Cannot open/decode image: {e}", w, h, mean_v, std_v, lap_v, was_repaired

    # 4. Global Quality Analysis
    try:
        mean_val = float(np.mean(img_arr))
        std_val = float(np.std(img_arr))

        if is_monochrome:
            return rel_path, filename, filepath, "MONOCHROME", "Grayscale / B&W camera sensor drop", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired

        # Check for solid color / extreme flat contrast / black / white
        if std_val < 1.0:
            if mean_val < 10.0:
                return rel_path, filename, filepath, "DARK", f"Completely black (mean: {mean_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired
            elif mean_val > 245.0:
                return rel_path, filename, filepath, "WHITE", f"Completely white (mean: {mean_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired
            else:
                return rel_path, filename, filepath, "LOW_CONTRAST", f"Uniform solid color (mean: {mean_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired

        if std_val < 5.0:
            return rel_path, filename, filepath, "LOW_CONTRAST", f"Extremely low contrast (std: {std_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired

        if mean_val < 5.0:
            return rel_path, filename, filepath, "DARK", f"Extremely dark/underexposed (mean: {mean_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired
        if mean_val > 250.0:
            return rel_path, filename, filepath, "WHITE", f"Extremely bright/overexposed (mean: {mean_val:.2f})", orig_w, orig_h, mean_val, std_val, 0.0, was_repaired

        # 5. Smart Global & Local Blur Check (4x4 Grid Patch Analysis)
        gh, gw = img_arr.shape
        grid_rows, grid_cols = 4, 4
        rh, rw = gh // grid_rows, gw // grid_cols

        patch_vars = []
        for r in range(grid_rows):
            # Skip top/bottom text overlay margins if in extreme rows
            for c in range(grid_cols):
                # Skip center pole columns (column index 1 and 2 in 4-col grid)
                if c in [1, 2]:
                    continue
                
                p = img_arr[r*rh:(r+1)*rh, c*rw:(c+1)*rw]
                lap_p = (p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1] - 4.0 * p[1:-1, 1:-1])
                patch_vars.append(float(np.var(lap_p)))

        patch_vars.sort()
        avg_side_var = float(np.mean(patch_vars)) if patch_vars else 0.0
        min_patch_var = patch_vars[0] if patch_vars else 0.0
        p25_patch_var = patch_vars[len(patch_vars) // 4] if patch_vars else 0.0

        # Global blur check
        if avg_side_var < blur_threshold:
            return rel_path, filename, filepath, "BLURRED", f"Global blur (avg Laplacian: {avg_side_var:.1f} < threshold: {blur_threshold:.1f})", orig_w, orig_h, mean_val, std_val, avg_side_var, was_repaired

        # Local blur check (partial smearing, bottom/side motion blur)
        if check_local_blur:
            local_thresh = max(10.0, blur_threshold * 0.45)
            if p25_patch_var < local_thresh or min_patch_var < (local_thresh * 0.5):
                return rel_path, filename, filepath, "LOCAL_BLUR", f"Local smearing/motion blur (worst patch Laplacian: {min_patch_var:.1f}, 25th percentile: {p25_patch_var:.1f} < threshold: {local_thresh:.1f})", orig_w, orig_h, mean_val, std_val, avg_side_var, was_repaired

        if was_repaired:
            return rel_path, filename, filepath, "REPAIRED", "JPEG EOI appended, valid image", orig_w, orig_h, mean_val, std_val, avg_side_var, was_repaired

        return rel_path, filename, filepath, "OK", "Valid image", orig_w, orig_h, mean_val, std_val, avg_side_var, was_repaired

    except Exception as e:
        return rel_path, filename, filepath, "ERROR", f"Error during quality analysis: {e}", orig_w, orig_h, 0.0, 0.0, 0.0, was_repaired

def main():
    parser = argparse.ArgumentParser(description="Aggressive & Local-Blur Smart Image Cleaner")
    parser.add_argument("--folder", "-f", type=str, required=False, help="Target folder path to clean")
    parser.add_argument("--mode", "-m", choices=["conservative", "medium", "aggressive"], default="medium", help="Preset sensitivity level (default: medium)")
    parser.add_argument("--blur-threshold", "-b", type=float, default=None, help="Custom blur threshold on 256x256 grid (overrides --mode)")
    parser.add_argument("--local-blur", action="store_true", default=True, help="Enable 4x4 spatial grid analysis to catch local/partial motion blurs")
    parser.add_argument("--no-local-blur", action="store_false", dest="local_blur", help="Disable local patch blur detection")
    parser.add_argument("--monochrome", action="store_true", help="Flag B&W / Grayscale / IR-mode images as bad")
    parser.add_argument("--recursive", "-r", action="store_true", help="Include subdirectories inside target folder")
    parser.add_argument("--no-repair", action="store_true", help="Disable automatic in-place JPEG repair")
    parser.add_argument("--workers", "-w", type=int, default=os.cpu_count(), help="Number of parallel worker processes")
    args = parser.parse_args()

    target_folder = args.folder
    if not target_folder:
        target_folder = input("Enter target folder path to clean (press Enter for current folder): ").strip()
        if not target_folder:
            target_folder = "."

    target_path = os.path.abspath(target_folder)
    if not os.path.exists(target_path):
        print(f"Error: Directory '{target_path}' does not exist.")
        sys.exit(1)

    if args.blur_threshold is not None:
        blur_thresh = args.blur_threshold
    else:
        blur_thresh = BLUR_PRESETS[args.mode]

    bad_dir = os.path.join(target_path, "bad_images")
    corrupt_dir = os.path.join(target_path, "corrupt_images")
    do_repair = not args.no_repair

    print("=" * 68)
    print("      LOCAL-BLUR & AGGRESSIVE SINGLE FOLDER IMAGE CLEANER")
    print("=" * 68)
    print(f"Target Directory:    {target_path}")
    print(f"Sensitivity Mode:    {args.mode.upper()}")
    print(f"Blur Threshold:      {blur_thresh:.1f}")
    print(f"Local Patch Blur:    {args.local_blur} (4x4 Grid Analysis)")
    print(f"Flag Monochrome/B&W: {args.monochrome}")
    print(f"Recursive Search:    {args.recursive}")
    print(f"Auto JPEG Repair:    {do_repair}")
    print(f"Bad Images Subfolder: {bad_dir}")
    print(f"Corrupt Subfolder:   {corrupt_dir}")
    print(f"Worker Processes:    {args.workers}")
    print("=" * 68)

    # Gather files
    files_to_check = []
    if args.recursive:
        for root, dirs, files in os.walk(target_path):
            if "bad_images" in root or "corrupt_images" in root or "truncated_images" in root:
                continue
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    files_to_check.append((os.path.join(root, f), f, target_path, do_repair, blur_thresh, args.monochrome, args.local_blur))
    else:
        for f in os.listdir(target_path):
            fp = os.path.join(target_path, f)
            if os.path.isfile(fp):
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    files_to_check.append((fp, f, target_path, do_repair, blur_thresh, args.monochrome, args.local_blur))

    total_files = len(files_to_check)
    print(f"Found {total_files} candidate image files to process.")
    if total_files == 0:
        print("No image files found in target folder. Exiting.")
        sys.exit(0)

    start_time = time.time()
    results = []

    print("Analyzing image integrity and visual quality...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        mapped_results = executor.map(analyze_and_clean_file, files_to_check, chunksize=100)
        for idx, res in enumerate(mapped_results, 1):
            results.append(res)
            if idx % 1000 == 0 or idx == total_files:
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                print(f"Processed {idx}/{total_files} files ({(idx/total_files)*100:.1f}%) | Speed: {rate:.1f} img/sec")

    elapsed_time = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed_time:.2f} seconds.")

    # Sorting and moving bad/corrupt images
    print("\nOrganizing and isolating files...")
    report_rows = []
    counts = {"OK": 0, "REPAIRED": 0, "BAD": 0, "CORRUPT": 0, "SKIP": 0}
    bad_type_counts = {}

    for res in results:
        rel_path, filename, filepath, status, detail, w, h, mean_v, std_v, lap_v, was_repaired = res

        if status == "OK":
            counts["OK"] += 1
        elif status == "REPAIRED":
            counts["REPAIRED"] += 1
        elif status in ["BLURRED", "LOCAL_BLUR", "LOW_CONTRAST", "DARK", "WHITE", "MONOCHROME"]:
            counts["BAD"] += 1
            bad_type_counts[status] = bad_type_counts.get(status, 0) + 1
            rel_dir = os.path.dirname(rel_path)
            dest_folder = os.path.join(bad_dir, rel_dir) if rel_dir else bad_dir
            os.makedirs(dest_folder, exist_ok=True)
            dest_path = os.path.join(dest_folder, filename)
            try:
                shutil.move(filepath, dest_path)
                report_rows.append({
                    "Filename": filename,
                    "RelativePath": rel_path,
                    "Status": status,
                    "Action": f"Moved to bad_images",
                    "Details": detail
                })
            except Exception as e:
                print(f"Error moving {filename}: {e}")

        elif status in ["CORRUPT", "ERROR"]:
            counts["CORRUPT"] += 1
            rel_dir = os.path.dirname(rel_path)
            dest_folder = os.path.join(corrupt_dir, rel_dir) if rel_dir else corrupt_dir
            os.makedirs(dest_folder, exist_ok=True)
            dest_path = os.path.join(dest_folder, filename)
            try:
                shutil.move(filepath, dest_path)
                report_rows.append({
                    "Filename": filename,
                    "RelativePath": rel_path,
                    "Status": status,
                    "Action": f"Moved to corrupt_images",
                    "Details": detail
                })
            except Exception as e:
                print(f"Error moving {filename}: {e}")

    # Generate CSV report in target folder
    report_path = os.path.join(target_path, "image_cleanup_report.csv")
    with open(report_path, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["Filename", "RelativePath", "Status", "Action", "Details"])
        writer.writeheader()
        writer.writerows(report_rows)

    print("\n" + "=" * 68)
    print("                    CLEANUP SUMMARY")
    print("=" * 68)
    print(f"Total Scanned Images:    {total_files}")
    print(f"Valid Images (Kept):     {counts['OK']}")
    print(f"Repaired JPEGs (Kept):   {counts['REPAIRED']}")
    print(f"Quality-Deficient (Moved to bad_images):     {counts['BAD']}")
    for k, v in bad_type_counts.items():
        print(f"  - {k:15}: {v}")
    print(f"Corrupt Images (Moved to corrupt_images):    {counts['CORRUPT']}")
    print(f"Report Generated:        {report_path}")
    print("=" * 68)

if __name__ == "__main__":
    main()
