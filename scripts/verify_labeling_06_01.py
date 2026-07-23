import os
import glob
import re
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Paths
DATA_DIR = r"C:\Users\ASUS\Desktop\Ders\blg521\data\06.01\2014\K1\10X"
CSV_PATH = r"C:\Users\ASUS\Desktop\Ders\blg521\labeling_bbch_iso_dates.csv"
OUTPUT_PLOT = r"C:\Users\ASUS\.gemini\antigravity\brain\2eb7377a-16b1-4588-afc7-a33a72f878a9\exg_curve_06_01.png"
OUTPUT_CSV = r"C:\Users\ASUS\Desktop\Ders\blg521\scripts\exg_analysis_06_01.csv"

# Labeled dates for Station 06.01 (ID 9, Year 2013/Harvest 2014)
LABELED_DATES = {
    "Sowing": "2013-11-14",
    "Emergence": "2014-03-07",
    "Tillering": "2014-04-16",
    "Stem Elongation": "2014-04-22",
    "Heading": "2014-05-21",
    "Flowering": "2014-06-13",
    "Maturity": "2014-06-25",
    "Harvest": "2014-07-22"
}

def parse_date_from_filename(filename):
    # Pattern: 06_01-YYYY_MM_DD-HH_MM-K1-10X.jpeg
    basename = os.path.basename(filename)
    match = re.search(r"06_01-(\d{4})_(\d{2})_(\d{2})", basename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None

def compute_excess_green(image_path):
    try:
        with Image.open(image_path) as img:
            # Downsample to 256x256 for high speed (retains spatial greenness averages)
            img = img.resize((256, 256), Image.Resampling.NEAREST)
            arr = np.array(img)
            
            # The wheat is in the bottom half of the image
            h = arr.shape[0]
            roi = arr[h // 2:, :, :]
            
            R = roi[:, :, 0].astype(float)
            G = roi[:, :, 1].astype(float)
            B = roi[:, :, 2].astype(float)
            
            denom = R + G + B
            denom[denom == 0] = 1.0
            
            r = R / denom
            g = G / denom
            b = B / denom
            
            # Excess Green index (ExG)
            exg = 2 * g - r - b
            return float(exg.mean())
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

def main():
    print("Searching for images in:", DATA_DIR)
    img_files = glob.glob(os.path.join(DATA_DIR, "*.jpeg"))
    print(f"Found {len(img_files)} images.")
    
    data = []
    for idx, filepath in enumerate(img_files):
        date_str = parse_date_from_filename(filepath)
        if date_str:
            exg = compute_excess_green(filepath)
            if exg is not None:
                data.append({
                    "Date": date_str,
                    "Filename": os.path.basename(filepath),
                    "ExG": exg
                })
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(img_files)} images...")

    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    
    # Save CSV output
    df.to_csv(OUTPUT_CSV, index=False)
    print("Saved raw ExG CSV to:", OUTPUT_CSV)
    
    # Aggregate daily ExG
    daily_df = df.groupby("Date")["ExG"].mean().reset_index()
    
    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(daily_df["Date"], daily_df["ExG"], color="forestgreen", linewidth=2, label="Excess Green Index (ExG)")
    
    # Draw vertical lines for phenological stages
    colors = {
        "Sowing": "brown",
        "Emergence": "lime",
        "Tillering": "dodgerblue",
        "Stem Elongation": "teal",
        "Heading": "magenta",
        "Flowering": "orange",
        "Maturity": "goldenrod",
        "Harvest": "red"
    }
    
    for stage, date_str in LABELED_DATES.items():
        date_val = pd.to_datetime(date_str)
        # Check if the date is within our plotted range
        if daily_df["Date"].min() <= date_val <= daily_df["Date"].max():
            plt.axvline(date_val, color=colors[stage], linestyle="--", alpha=0.8, linewidth=1.5, label=f"Labeled {stage} ({date_str})")
            
            # Find the ExG value closest to this date to place the text
            closest_idx = (daily_df["Date"] - date_val).abs().idxmin()
            closest_exg = daily_df.loc[closest_idx, "ExG"]
            
            plt.text(date_val, closest_exg + 0.02, stage, color=colors[stage], rotation=90, 
                     verticalalignment='bottom', horizontalalignment='right', 
                     fontweight='bold', bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'))
        else:
            print(f"Stage '{stage}' date {date_str} is out of image date range ({daily_df['Date'].min().strftime('%Y-%m-%d')} to {daily_df['Date'].max().strftime('%Y-%m-%d')})")
            
    plt.title("Wheat Growth Curve (Excess Green Index) - Station 06.01 (2014)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Excess Green (ExG) of ROI", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Dedup legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print("Saved analysis plot to:", OUTPUT_PLOT)
    
    # Calculate some stats for the report
    print("\nPhenological Stage ExG Metrics:")
    for stage, date_str in LABELED_DATES.items():
        date_val = pd.to_datetime(date_str)
        if daily_df["Date"].min() <= date_val <= daily_df["Date"].max():
            closest_idx = (daily_df["Date"] - date_val).abs().idxmin()
            actual_date = daily_df.loc[closest_idx, "Date"].strftime("%Y-%m-%d")
            exg_val = daily_df.loc[closest_idx, "ExG"]
            print(f"  {stage:15} Labeled: {date_str} | Closest File: {actual_date} | ExG: {exg_val:+.4f}")

if __name__ == "__main__":
    main()
