import os
import csv
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor

DATA_DIR = r"C:\Users\ASUS\Desktop\Ders\blg521\data\02.03"
REPORT_PATH = r"C:\Users\ASUS\Desktop\Ders\blg521\scripts\rigorous_report_02_03.csv"

def analyze_image_rigorous(args):
    rel_path, data_dir = args
    filepath = os.path.join(data_dir, rel_path)
    
    try:
        img = Image.open(filepath)
        width, height = img.size
        
        # Load image details
        img_rgb = img.convert('RGB')
        arr = np.array(img_rgb)
        
        # Split channels
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        
        mean_r = float(np.mean(r))
        mean_g = float(np.mean(g))
        mean_b = float(np.mean(b))
        
        std_r = float(np.std(r))
        std_g = float(np.std(g))
        std_b = float(np.std(b))
        
        # Grayscale check: Check if colors are almost identical across channels
        # Standard deviation of color differences
        diff_rg = np.abs(r.astype(np.int16) - g.astype(np.int16))
        diff_gb = np.abs(g.astype(np.int16) - b.astype(np.int16))
        diff_br = np.abs(b.astype(np.int16) - r.astype(np.int16))
        mean_diff = float((np.mean(diff_rg) + np.mean(diff_gb) + np.mean(diff_br)) / 3.0)
        
        # Gray check
        is_gray = mean_diff < 3.0
        
        # Exposure / Saturation: Percent of pixels that are extremely bright (e.g. > 250) or dark (< 5)
        bright_pixels_pct = float(np.sum(arr > 250) / arr.size) * 100.0
        dark_pixels_pct = float(np.sum(arr < 5) / arr.size) * 100.0
        
        # Convert to grayscale for Laplacian check
        gray = img.convert('L')
        # Resize for consistent laplacian check
        gray_resized = gray.resize((512, 512), Image.Resampling.BILINEAR)
        arr_gray = np.array(gray_resized, dtype=np.float32)
        
        laplacian = (
            arr_gray[1:-1, 2:] + arr_gray[1:-1, :-2] +
            arr_gray[2:, 1:-1] + arr_gray[:-2, 1:-1] -
            4.0 * arr_gray[1:-1, 1:-1]
        )
        lap_var = float(np.var(laplacian))
        
        # Categorize rigorously:
        status = "OK"
        detail = "Clean image"
        
        # 1. Grayscale (often night IR or dense white-out fog)
        if is_gray and (mean_r < 40 or mean_r > 200):
            status = "NIGHT_OR_FOG"
            detail = f"Grayscale profile detected (color diff: {mean_diff:.2f}, mean R: {mean_r:.1f})"
            
        # 2. Overexposed/Glare
        elif mean_r > 220 and mean_g > 220 and mean_b > 220:
            status = "OVEREXPOSED"
            detail = f"Extremely high mean brightness (R:{mean_r:.1f}, G:{mean_g:.1f}, B:{mean_b:.1f})"
            
        # 3. High Saturation (glare)
        elif bright_pixels_pct > 30.0:
            status = "GLARE_SUSPECT"
            detail = f"High percentage of saturated pixels ({bright_pixels_pct:.1f}%)"
            
        # 4. Low Contrast
        elif (std_r + std_g + std_b)/3.0 < 15.0:
            status = "LOW_CONTRAST"
            detail = f"Low color contrast (std dev: {(std_r+std_g+std_b)/3.0:.1f})"
            
        # 5. Borderline Blurry
        elif lap_var < 80.0:
            status = "BORDERLINE_BLUR"
            detail = f"Borderline blurry (Laplacian var: {lap_var:.2f})"
            
        # 6. Under-exposed
        elif mean_r < 25.0 and mean_g < 25.0 and mean_b < 25.0:
            status = "UNDEREXPOSED"
            detail = f"Very dark image (mean R: {mean_r:.1f})"
            
        return {
            "RelPath": rel_path,
            "Status": status,
            "Detail": detail,
            "Width": width,
            "Height": height,
            "MeanR": f"{mean_r:.1f}",
            "MeanG": f"{mean_g:.1f}",
            "MeanB": f"{mean_b:.1f}",
            "ColorDiff": f"{mean_diff:.2f}",
            "BrightPct": f"{bright_pixels_pct:.1f}%",
            "DarkPct": f"{dark_pixels_pct:.1f}%",
            "LaplacianVar": f"{lap_var:.2f}"
        }
    except Exception as e:
        return {
            "RelPath": rel_path,
            "Status": "ERROR",
            "Detail": str(e),
            "Width": 0, "Height": 0, "MeanR": "0", "MeanG": "0", "MeanB": "0",
            "ColorDiff": "0", "BrightPct": "0%", "DarkPct": "0%", "LaplacianVar": "0"
        }

def run_rigorous_scan():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found.")
        return
        
    print("Collecting files in 02.03...")
    files = []
    for root, dirs, filenames in os.walk(DATA_DIR):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                rel_path = os.path.relpath(os.path.join(root, f), DATA_DIR)
                path_parts = rel_path.split(os.sep)
                if "K1" in path_parts:
                    files.append((rel_path, DATA_DIR))
                
    total_files = len(files)
    print(f"Found {total_files} images to scan rigorously.")
    
    results = []
    cpu_cores = os.cpu_count() or 4
    
    print(f"Starting parallel scan with {cpu_cores} workers...")
    with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        mapped_results = executor.map(analyze_image_rigorous, files, chunksize=50)
        for idx, res in enumerate(mapped_results, 1):
            results.append(res)
            if idx % 500 == 0:
                print(f"Scanned {idx}/{total_files} files...")
                
    # Save CSV
    headers = ["RelPath", "Status", "Detail", "Width", "Height", "MeanR", "MeanG", "MeanB", "ColorDiff", "BrightPct", "DarkPct", "LaplacianVar"]
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=headers)
        writer.writeheader()
        for res in results:
            writer.writerow(res)
            
    print(f"Rigorous report saved to: {REPORT_PATH}")
    
    # Summary of findings
    status_counts = {}
    for res in results:
        status_counts[res["Status"]] = status_counts.get(res["Status"], 0) + 1
        
    print("\n" + "="*50)
    print("           RIGOROUS SCAN SUMMARY FOR 02.03")
    print("="*50)
    for k, v in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {k}: {v}")
    print("="*50)

if __name__ == "__main__":
    run_rigorous_scan()
