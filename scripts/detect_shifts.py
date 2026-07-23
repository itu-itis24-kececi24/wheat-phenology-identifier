import os
import glob
import pandas as pd
import numpy as np
from PIL import Image

data_dir = r"C:\Users\ASUS\Desktop\Ders\blg521\data\06.02"
# Collect all K1 10X images from 2016 and 2017
files = []
for y in ["2016", "2017"]:
    p = os.path.join(data_dir, y, "K1", "10X", "*.jpeg")
    files.extend(glob.glob(p))
    
files = sorted(files)
print(f"Total files: {len(files)}")

# Compute average RGB for each image to detect shifts
records = []
for f in files:
    try:
        with Image.open(f) as img:
            img = img.resize((32, 32)) # tiny resize for speed
            arr = np.array(img)
            mean_r = arr[:, :, 0].mean()
            mean_g = arr[:, :, 1].mean()
            mean_b = arr[:, :, 2].mean()
            records.append({
                "Filename": os.path.basename(f),
                "R": mean_r,
                "G": mean_g,
                "B": mean_b
            })
    except Exception as e:
        pass

df = pd.DataFrame(records)
# Find large shifts in mean color between consecutive images
df["diff_R"] = df["R"].diff().abs()
df["diff_G"] = df["G"].diff().abs()
df["diff_B"] = df["B"].diff().abs()
df["total_diff"] = df["diff_R"] + df["diff_G"] + df["diff_B"]

# Sort by largest shifts
shifts = df[df["total_diff"] > 30]
print("\nLargest sudden visual shifts:")
print(shifts[["Filename", "R", "G", "B", "total_diff"]].to_string())
