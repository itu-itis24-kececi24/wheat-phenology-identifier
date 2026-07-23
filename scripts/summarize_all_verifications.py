import os
import pandas as pd
import numpy as np

SCRIPTS_DIR = r"C:\Users\ASUS\Desktop\Ders\blg521\scripts"
CSV_PATH = r"C:\Users\ASUS\Desktop\Ders\blg521\labeling_bbch_iso_dates.csv"
SUMMARY_PATH = os.path.join(SCRIPTS_DIR, "all_stations_verification_summary.csv")

df_csv = pd.read_csv(CSV_PATH)
summary_records = []

for idx, row in df_csv.iterrows():
    st_val = row["Station Code"]
    st_str = f"{st_val:.2f}"
    parts = st_str.split('.')
    st_dir = f"{int(parts[0]):02d}.{parts[1]}"
    year_val = int(row["Year"])
    cam_val = int(row["kamera"])
    cam_name = f"K{cam_val}"
    if st_dir == "02.06" and cam_val == 2:
        cam_name = "K3"
        
    sowing_str = str(row["1-Sowing"])
    emergence_str = str(row["2 - Emergence"])
    heading_str = str(row["5 - Heading"])
    harvest_str = str(row["8 - Harvest"])
    
    csv_file = f"exg_analysis_row_{idx:02d}_{st_dir}_{year_val}.csv"
    csv_path = os.path.join(SCRIPTS_DIR, csv_file)
    
    if not os.path.exists(csv_path):
        summary_records.append({
            "Row": idx,
            "Station": st_dir,
            "Year": year_val,
            "Camera": cam_name,
            "Image Count": 0,
            "Curve Quality": "NO IMAGES",
            "Peak ExG Date": "N/A",
            "Peak ExG Val": 0.0,
            "Labeled Emergence": emergence_str,
            "Detected Emergence": "N/A",
            "Emergence Offset": "N/A",
            "Labeled Heading": heading_str,
            "Peak ExG Date": "N/A",
            "Heading Offset": "N/A",
            "Labeled Harvest": harvest_str,
            "Detected Harvest": "N/A",
            "Harvest Offset": "N/A"
        })
        continue

    df_exg = pd.read_csv(csv_path)
    df_exg["Date"] = pd.to_datetime(df_exg["Date"])
    daily = df_exg.groupby("Date")["ExG"].mean().reset_index()
    
    img_count = len(daily)
    peak_idx = daily["ExG"].idxmax()
    peak_date = daily.loc[peak_idx, "Date"]
    peak_exg = daily.loc[peak_idx, "ExG"]
    
    if peak_exg < 0.15:
        curve_quality = "FLAT / NO SIGNAL"
    elif peak_exg < 0.25:
        curve_quality = "MEDIUM SIGNAL"
    else:
        curve_quality = "STRONG SIGNAL"
        
    # Emergence threshold detection (ExG >= 0.05)
    em_date = None
    for _, r_d in daily.iterrows():
        if r_d["ExG"] >= 0.05:
            em_date = r_d["Date"]
            break
            
    # Harvest threshold detection (first drop post-peak ExG <= 0.08)
    hv_date = None
    post_peak = daily[daily["Date"] > peak_date]
    for _, r_d in post_peak.iterrows():
        if r_d["ExG"] <= 0.08:
            hv_date = r_d["Date"]
            break
            
    lbl_em = pd.to_datetime(emergence_str) if pd.notna(emergence_str) else None
    lbl_hd = pd.to_datetime(heading_str) if pd.notna(heading_str) else None
    lbl_hv = pd.to_datetime(harvest_str) if pd.notna(harvest_str) else None
    
    em_offset = (em_date - lbl_em).days if em_date and lbl_em else "N/A"
    hd_offset = (peak_date - lbl_hd).days if peak_date and lbl_hd else "N/A"
    hv_offset = (hv_date - lbl_hv).days if hv_date and lbl_hv else "N/A"
    
    summary_records.append({
        "Row": idx,
        "Station": st_dir,
        "Year": year_val,
        "Camera": cam_name,
        "Image Count": img_count,
        "Curve Quality": curve_quality,
        "Peak ExG Date": peak_date.strftime("%Y-%m-%d"),
        "Peak ExG Val": round(peak_exg, 4),
        "Labeled Emergence": emergence_str,
        "Detected Emergence": em_date.strftime("%Y-%m-%d") if em_date else "N/A",
        "Emergence Offset": em_offset,
        "Labeled Heading": heading_str,
        "Heading Offset": hd_offset,
        "Labeled Harvest": harvest_str,
        "Detected Harvest": hv_date.strftime("%Y-%m-%d") if hv_date else "N/A",
        "Harvest Offset": hv_offset
    })

df_res = pd.DataFrame(summary_records)
output_file = os.path.join(SCRIPTS_DIR, "comprehensive_verification_report.csv")
df_res.to_csv(output_file, index=False)
print("Saved comprehensive report to:", output_file)
print(df_res.to_string())
