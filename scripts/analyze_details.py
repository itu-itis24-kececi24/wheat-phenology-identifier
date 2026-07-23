import os
import pandas as pd

scripts_dir = r"C:\Users\ASUS\Desktop\Ders\blg521\scripts"
summary_path = os.path.join(scripts_dir, "all_stations_verification_summary.csv")
df_summary = pd.read_csv(summary_path)

print("Station Season Analysis for Reliable Stations:")
print("-" * 120)

for idx, row in df_summary.iterrows():
    row_id = int(row["Row"])
    st_dir = row["Station"]
    year_val = int(row["Year"])
    peak_exg = row["Peak ExG Val"]
    
    if peak_exg < 0.20:
        # Flat curve
        print(f"Row {row_id:02d} | Station {st_dir} ({year_val}) | Peak ExG: {peak_exg:.4f} | Status: FLAT/UNRELIABLE (Bare soil or sparse crop)")
        continue
        
    csv_file = f"exg_analysis_row_{row_id:02d}_{st_dir}_{year_val}.csv"
    csv_path = os.path.join(scripts_dir, csv_file)
    if not os.path.exists(csv_path):
        continue
        
    df_exg = pd.read_csv(csv_path)
    df_exg["Date"] = pd.to_datetime(df_exg["Date"])
    daily = df_exg.groupby("Date")["ExG"].mean().reset_index()
    
    # Peak Greenness
    peak_idx = daily["ExG"].idxmax()
    peak_date = daily.loc[peak_idx, "Date"]
    
    # Emergence threshold: First day where ExG crosses 0.05 (or 0.06 depending on baseline)
    # Let's find when ExG crosses 0.05
    em_date = None
    for _, r_d in daily.iterrows():
        if r_d["ExG"] >= 0.05:
            em_date = r_d["Date"]
            break
            
    # Harvest threshold: First day after peak where ExG drops below 0.10 (or near baseline)
    hv_date = None
    post_peak = daily[daily["Date"] > peak_date]
    for _, r_d in post_peak.iterrows():
        if r_d["ExG"] <= 0.08:
            hv_date = r_d["Date"]
            break
            
    lbl_em = row["Labeled Emergence"]
    lbl_hv = row["Labeled Harvest"]
    lbl_hd = row["Labeled Heading"]
    
    print(f"Row {row_id:02d} | Station {st_dir} ({year_val}) | Peak ExG: {peak_exg:.4f} | Peak Date: {peak_date.strftime('%Y-%m-%d')}")
    print(f"  Emergence -> Labeled: {lbl_em} | Detected (ExG>=0.05): {em_date.strftime('%Y-%m-%d') if em_date else 'None'}")
    print(f"  Heading   -> Labeled: {lbl_hd} | Peak Greenness: {peak_date.strftime('%Y-%m-%d')}")
    print(f"  Harvest   -> Labeled: {lbl_hv} | Detected (ExG<=0.08): {hv_date.strftime('%Y-%m-%d') if hv_date else 'None'}")
    print("-" * 120)
