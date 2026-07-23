import os
import pandas as pd

data_dir = r"C:\Users\ASUS\Desktop\Ders\blg521\data"
df_csv = pd.read_csv(r"C:\Users\ASUS\Desktop\Ders\blg521\labeling_bbch_iso_dates.csv")

for idx, row in df_csv.iterrows():
    st_val = row["Station Code"]
    st_str = f"{st_val:.2f}"
    parts = st_str.split('.')
    st_dir = f"{int(parts[0]):02d}.{parts[1]}"
    
    year_val = int(row["Year"])
    cam_val = int(row["kamera"])
    
    # Camera name mapping
    cam_name = f"K{cam_val}"
    if st_dir == "02.06" and cam_val == 2:
        cam_name = "K3"
        
    # Check if 10X folder exists for this season
    # Note that season might span two years, let's search in all year subfolders
    st_path = os.path.join(data_dir, st_dir)
    found_years = []
    found_10x = False
    
    if os.path.exists(st_path):
        for y in os.listdir(st_path):
            y_path = os.path.join(st_path, y)
            if os.path.isdir(y_path) and y.isdigit():
                target_path = os.path.join(y_path, cam_name, "10X")
                if os.path.isdir(target_path):
                    found_years.append(y)
                    found_10x = True
                    
    print(f"Row {idx:02d} | Station {st_dir} | Year {year_val} | Cam {cam_name} | Found Years: {found_years} | 10X Found: {found_10x}")
