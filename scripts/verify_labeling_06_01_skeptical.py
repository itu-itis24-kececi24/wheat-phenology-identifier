import os
import glob
import re
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Paths
DATA_DIR = r"C:\Users\ASUS\Desktop\Ders\blg521\data\06.01\2014\K1\10X"
OUTPUT_PLOT = r"C:\Users\ASUS\.gemini\antigravity\brain\2eb7377a-16b1-4588-afc7-a33a72f878a9\exg_curve_06_01_skeptical.png"
OUTPUT_CSV = r"C:\Users\ASUS\Desktop\Ders\blg521\scripts\exg_analysis_06_01.csv"

# Labeled dates for Station 06.01 (ID 9, Year 2013/Harvest 2014)
LABELED_DATES = {
    "Emergence": "2014-03-07",
    "Tillering": "2014-04-16",
    "Stem Elongation": "2014-04-22",
    "Heading": "2014-05-21",
    "Flowering": "2014-06-13",
    "Maturity": "2014-06-25",
    "Harvest": "2014-07-22"
}

# Skeptically verified dates based on micro-visual inspection of 10X images
VERIFIED_DATES = {
    "Emergence": "2014-02-27",       # Green shoots visible under zoom
    "Tillering": "2014-03-25",       # Plant clustering/branching starts
    "Stem Elongation": "2014-04-22", # Vertical growth onset
    "Heading": "2014-05-18",         # Spikes already emerged in image
    "Flowering": "2014-06-13",       # Anthers visible
    "Maturity": "2014-06-25",        # Start of senescence/yellowing
    "Harvest": "2014-07-22"          # Last day of standing crop
}

def parse_date_from_filename(filename):
    basename = os.path.basename(filename)
    match = re.search(r"06_01-(\d{4})_(\d{2})_(\d{2})", basename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None

def compute_excess_green(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.resize((256, 256), Image.Resampling.NEAREST)
            arr = np.array(img)
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
            exg = 2 * g - r - b
            return float(exg.mean())
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

def main():
    print("Loading data from existing ExG analysis CSV if available...")
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        df["Date"] = pd.to_datetime(df["Date"])
        print(f"Loaded {len(df)} rows from CSV.")
    else:
        print("CSV not found, computing ExG from raw images...")
        img_files = glob.glob(os.path.join(DATA_DIR, "*.jpeg"))
        data = []
        for idx, filepath in enumerate(img_files):
            date_str = parse_date_from_filename(filepath)
            if date_str:
                exg = compute_excess_green(filepath)
                if exg is not None:
                    data.append({"Date": date_str, "Filename": os.path.basename(filepath), "ExG": exg})
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(img_files)} images...")
        df = pd.DataFrame(data)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df.to_csv(OUTPUT_CSV, index=False)
        print("Computed and saved ExG to CSV.")

    daily_df = df.groupby("Date")["ExG"].mean().reset_index()

    plt.figure(figsize=(15, 8))
    plt.plot(daily_df["Date"], daily_df["ExG"], color="forestgreen", linewidth=2.5, label="Excess Green Index (ExG)")

    colors = {
        "Emergence": "#2ca02c",
        "Tillering": "#1f77b4",
        "Stem Elongation": "#17becf",
        "Heading": "#e377c2",
        "Flowering": "#ff7f0e",
        "Maturity": "#bcbd22",
        "Harvest": "#d62728"
    }

    # Plot Labeled Dates as thin dashed lines
    for stage, date_str in LABELED_DATES.items():
        date_val = pd.to_datetime(date_str)
        if daily_df["Date"].min() <= date_val <= daily_df["Date"].max():
            plt.axvline(date_val, color=colors[stage], linestyle=":", alpha=0.5, linewidth=1.5)
            # Label near the top
            plt.text(date_val, plt.gca().get_ylim()[1] * 0.9, f"Labeled {stage[0:4]}.", color=colors[stage], rotation=90,
                     fontsize=8, alpha=0.7, verticalalignment='top', horizontalalignment='right')

    # Plot Verified Dates as solid vertical lines with markers
    for stage, date_str in VERIFIED_DATES.items():
        date_val = pd.to_datetime(date_str)
        if daily_df["Date"].min() <= date_val <= daily_df["Date"].max():
            plt.axvline(date_val, color=colors[stage], linestyle="-", alpha=0.9, linewidth=2.0, label=f"Verified {stage} ({date_str})")
            
            # Find the ExG value closest to this date
            closest_idx = (daily_df["Date"] - date_val).abs().idxmin()
            closest_exg = daily_df.loc[closest_idx, "ExG"]
            
            plt.plot(date_val, closest_exg, marker='o', markersize=8, color=colors[stage])
            plt.text(date_val, closest_exg + 0.015, stage, color=colors[stage], rotation=0,
                     verticalalignment='bottom', horizontalalignment='center',
                     fontweight='bold', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

    plt.title("Wheat Growth Curve (ExG) - Labeled vs. Skeptically Verified Dates (Station 06.01, 2014)", fontsize=14, fontweight="bold")
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
    print("Saved skeptical analysis plot to:", OUTPUT_PLOT)

if __name__ == "__main__":
    main()
