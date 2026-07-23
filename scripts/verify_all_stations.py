import os
import re
import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# Paths
DATA_DIR = r"C:\Users\ASUS\Desktop\Ders\blg521\data"
CSV_PATH = r"C:\Users\ASUS\Desktop\Ders\blg521\labeling_bbch_iso_dates.csv"
OUTPUT_DIR = r"C:\Users\ASUS\.gemini\antigravity\brain\2eb7377a-16b1-4588-afc7-a33a72f878a9"
SCRIPTS_DIR = r"C:\Users\ASUS\Desktop\Ders\blg521\scripts"

def parse_date_from_filename(filename):
    basename = os.path.basename(filename)
    match = re.search(r"-\d{4}_\d{2}_\d{2}", basename)
    if match:
        # returns YYYY-MM-DD
        return match.group(0)[1:].replace('_', '-')
    return None

def compute_excess_green(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.resize((256, 256), Image.Resampling.NEAREST)
            arr = np.array(img)
            h = arr.shape[0]
            # Crop to bottom half where wheat grows
            roi = arr[h // 2:, :, :]
            R = roi[:, :, 0].astype(float)
            G = roi[:, :, 1].astype(float)
            B = roi[:, :, 2].astype(float)
            denom = R + G + B
            denom[denom == 0] = 1.0
            r = R / denom
            g = G / denom
            b = B / denom
            exg = 2 * g - r - b
            return float(exg.mean())
    except Exception as e:
        return None

def process_single_image(args):
    filepath, date_str = args
    exg = compute_excess_green(filepath)
    if exg is not None:
        return {"Date": date_str, "Filename": os.path.basename(filepath), "ExG": exg}
    return None

def main():
    print("Loading labeling CSV...")
    df_csv = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df_csv)} labeled seasons.")
    
    summary_results = []
    
    # Process each row in the CSV
    for idx, row in df_csv.iterrows():
        st_val = row["Station Code"]
        st_str = f"{st_val:.2f}"
        parts = st_str.split('.')
        st_dir = f"{int(parts[0]):02d}.{parts[1]}"
        
        year_val = int(row["Year"])
        sowing_str = row["1-Sowing"]
        emergence_str = row["2 - Emergence"]
        tillering_str = row["3 - Tillering"]
        stem_str = row["4 - Stem Elongation"]
        heading_str = row["5 - Heading"]
        flowering_str = row["6 - Flowering"]
        maturity_str = row["7 - Maturity"]
        harvest_str = row["8 - Harvest"]
        
        cam_val = int(row["kamera"])
        cam_name = f"K{cam_val}"
        if st_dir == "02.06" and cam_val == 2:
            cam_name = "K3"
            
        print(f"\n==================================================")
        print(f"Row {idx:02d} | Station: {st_dir} | Year: {year_val} | Cam: {cam_name}")
        print(f"Sowing: {sowing_str} -> Harvest: {harvest_str}")
        
        # 1. Gather all files recursively under the station directory matching the camera and 10X zoom
        st_path = os.path.join(DATA_DIR, st_dir)
        raw_files = []
        if os.path.exists(st_path):
            # Glob files across any year subdirectory
            for y_folder in os.listdir(st_path):
                y_path = os.path.join(st_path, y_folder)
                if os.path.isdir(y_path) and y_folder.isdigit():
                    target_path = os.path.join(y_path, cam_name, "10X")
                    if os.path.isdir(target_path):
                        raw_files.extend(glob.glob(os.path.join(target_path, "*.jpeg")))
        
        print(f"Found {len(raw_files)} total raw 10X images in folder.")
        
        # 2. Parse dates and filter strictly within sowing and harvest boundaries
        sowing_date = pd.to_datetime(sowing_str)
        harvest_date = pd.to_datetime(harvest_str)
        
        filtered_tasks = []
        for f in raw_files:
            date_str = parse_date_from_filename(f)
            if date_str:
                file_date = pd.to_datetime(date_str)
                if sowing_date <= file_date <= harvest_date:
                    filtered_tasks.append((f, date_str))
                    
        print(f"Filtered to {len(filtered_tasks)} images within [Sowing, Harvest] boundaries.")
        
        if not filtered_tasks:
            print("No images found in the labeling window. Skipping computation.")
            continue
            
        # 3. Compute ExG in parallel
        computed_data = []
        num_workers = min(os.cpu_count() or 4, 8)
        print(f"Processing in parallel using {num_workers} workers...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single_image, task) for task in filtered_tasks]
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    computed_data.append(res)
                    
        df_exg = pd.DataFrame(computed_data)
        if df_exg.empty:
            print("Failed to extract ExG from any images. Skipping.")
            continue
            
        df_exg["Date"] = pd.to_datetime(df_exg["Date"])
        df_exg = df_exg.sort_values("Date")
        
        # Save raw ExG output for this specific station season
        csv_out_name = f"exg_analysis_row_{idx:02d}_{st_dir}_{year_val}.csv"
        csv_out_path = os.path.join(SCRIPTS_DIR, csv_out_name)
        df_exg.to_csv(csv_out_path, index=False)
        print(f"Saved computed ExG curves to {csv_out_path}")
        
        # Aggregate daily ExG
        daily_df = df_exg.groupby("Date")["ExG"].mean().reset_index()
        
        # 4. Programmatic Landmark Analysis
        # Let's find some key markers from the ExG time series:
        # A. Peak ExG Date: Peak greenness
        peak_idx = daily_df["ExG"].idxmax()
        peak_date = daily_df.loc[peak_idx, "Date"]
        peak_exg = daily_df.loc[peak_idx, "ExG"]
        
        # B. Emergence Trend Detection: First day where ExG starts rising above baseline.
        # Let's define baseline as the mean of the first 10 days or first 15% of the time series
        num_baseline_days = min(len(daily_df), 15)
        baseline_mean = daily_df.iloc[:num_baseline_days]["ExG"].mean()
        baseline_std = daily_df.iloc[:num_baseline_days]["ExG"].std() or 0.005
        
        # First date where ExG > baseline_mean + 2*baseline_std and is rising
        detected_emergence = None
        for i in range(len(daily_df)):
            curr_exg = daily_df.loc[i, "ExG"]
            if curr_exg > baseline_mean + 2 * baseline_std:
                detected_emergence = daily_df.loc[i, "Date"]
                break
        if not detected_emergence:
            # Fallback to first day where ExG > 0.04
            for i in range(len(daily_df)):
                if daily_df.loc[i, "ExG"] > 0.04:
                    detected_emergence = daily_df.loc[i, "Date"]
                    break
                    
        # C. Harvest drop: find when ExG drops to post-peak baseline
        # Typically the minimum value in the last 15 days of the season
        post_peak_df = daily_df[daily_df["Date"] > peak_date]
        detected_harvest = None
        if not post_peak_df.empty:
            # Find the date after peak greenness where ExG drops to the bottom and plateaus
            bottom_exg = post_peak_df["ExG"].min()
            # First date after peak where ExG is close to this bottom (within 0.015)
            for _, r_post in post_peak_df.iterrows():
                if r_post["ExG"] <= bottom_exg + 0.015:
                    detected_harvest = r_post["Date"]
                    break
        else:
            detected_harvest = daily_df.iloc[-1]["Date"]
            
        print(f"Landmarks identified:")
        print(f"  Detected Emergence Start: {detected_emergence.strftime('%Y-%m-%d') if detected_emergence else 'None'}")
        print(f"  Peak Greenness Date: {peak_date.strftime('%Y-%m-%d')} (ExG: {peak_exg:.4f})")
        print(f"  Detected Harvest Drop: {detected_harvest.strftime('%Y-%m-%d') if detected_harvest else 'None'}")
        
        # 5. Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(daily_df["Date"], daily_df["ExG"], color="forestgreen", linewidth=2, label="Excess Green Index (ExG)")
        
        # Superimpose Labeled dates
        stages = {
            "Emergence": emergence_str,
            "Tillering": tillering_str,
            "Stem Elongation": stem_str,
            "Heading": heading_str,
            "Flowering": flowering_str,
            "Maturity": maturity_str,
            "Harvest": harvest_str
        }
        
        colors = {
            "Emergence": "lime",
            "Tillering": "dodgerblue",
            "Stem Elongation": "teal",
            "Heading": "magenta",
            "Flowering": "orange",
            "Maturity": "goldenrod",
            "Harvest": "red"
        }
        
        for stage, date_str in stages.items():
            if pd.notna(date_str):
                d_val = pd.to_datetime(date_str)
                if daily_df["Date"].min() <= d_val <= daily_df["Date"].max():
                    plt.axvline(d_val, color=colors[stage], linestyle="--", alpha=0.8, linewidth=1.2)
                    closest_idx = (daily_df["Date"] - d_val).abs().idxmin()
                    val_exg = daily_df.loc[closest_idx, "ExG"]
                    plt.plot(d_val, val_exg, marker='o', color=colors[stage], markersize=6)
                    plt.text(d_val, val_exg + 0.01, stage[0:3] + ".", color=colors[stage], rotation=90, fontsize=8,
                             verticalalignment='bottom', horizontalalignment='right',
                             bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.1'))
                             
        plt.title(f"Growth Curve (ExG) - Row {idx:02d} | Station {st_dir} ({year_val})", fontsize=12, fontweight="bold")
        plt.xlabel("Date", fontsize=10)
        plt.ylabel("ExG Index", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        
        plot_name = f"exg_curve_row_{idx:02d}_{st_dir}_{year_val}.png"
        plot_path = os.path.join(OUTPUT_DIR, plot_name)
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"Saved plot to {plot_path}")
        
        # Calculate offsets
        lbl_em = pd.to_datetime(emergence_str)
        offset_em = (detected_emergence - lbl_em).days if detected_emergence and pd.notna(lbl_em) else None
        
        lbl_hv = pd.to_datetime(harvest_str)
        offset_hv = (detected_harvest - lbl_hv).days if detected_harvest and pd.notna(lbl_hv) else None
        
        summary_results.append({
            "Row": idx,
            "Station": st_dir,
            "Year": year_val,
            "Camera": cam_name,
            "Image Count": len(daily_df),
            "Peak ExG Date": peak_date.strftime("%Y-%m-%d"),
            "Peak ExG Val": peak_exg,
            "Labeled Emergence": emergence_str,
            "Detected Emergence": detected_emergence.strftime("%Y-%m-%d") if detected_emergence else "None",
            "Emergence Offset (Days)": offset_em,
            "Labeled Heading": heading_str,
            "Labeled Harvest": harvest_str,
            "Detected Harvest": detected_harvest.strftime("%Y-%m-%d") if detected_harvest else "None",
            "Harvest Offset (Days)": offset_hv
        })
        
    # Write summary CSV
    summary_df = pd.DataFrame(summary_results)
    summary_csv_path = os.path.join(SCRIPTS_DIR, "all_stations_verification_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print("\n==================================================")
    print("FINISHED PROCESS FOR ALL STATIONS!")
    print(f"Summary saved to: {summary_csv_path}")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
