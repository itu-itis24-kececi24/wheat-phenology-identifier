import argparse

from multiscale_phenology import build_multiscale_daily_dataframe

parser = argparse.ArgumentParser()
parser.add_argument("--label-path", "--excel-path", dest="label_path", default="../labeling.xlsx")
parser.add_argument("--data-path", default="../data")
args = parser.parse_args()

df = build_multiscale_daily_dataframe(args.label_path, args.data_path, preferred_camera="AUTO")
print(df.head())
print(df["label"].value_counts().sort_index())
print(df[["macro_path", "micro_path"]].notna().mean())
camera_cols = [col for col in ["label_camera", "camera_preference"] if col in df.columns]
if camera_cols:
    print(df[camera_cols].drop_duplicates().sort_values(camera_cols).to_string(index=False))
