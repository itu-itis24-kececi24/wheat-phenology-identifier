import argparse

from multiscale_phenology import build_multiscale_daily_dataframe, print_station_image_edges

parser = argparse.ArgumentParser()
parser.add_argument("--label-path", "--excel-path", dest="label_path", default="labeling_bbch_iso_dates.csv")
parser.add_argument("--data-path", default="data")
parser.add_argument("--stream", choices=["micro", "macro", "both"], default="both")
parser.add_argument("--camera", default="AUTO", help="AUTO uses the label table kamera/Camera column when present.")
args = parser.parse_args()

preferred_camera = None if args.camera.upper() == "ALL" else args.camera
df = build_multiscale_daily_dataframe(args.label_path, args.data_path, preferred_camera=preferred_camera)
print(df.head())
print(df["label"].value_counts().sort_index())
print(df[["macro_path", "micro_path"]].notna().mean())
if {"label_camera", "camera_preference"}.issubset(df.columns):
    print(df[["station_year", "label_camera", "camera_preference"]].drop_duplicates().sort_values("station_year"))
print_station_image_edges(
    df,
    stream=args.stream,
    base_dir=args.data_path,
    title="First/last resolved images by station-year:",
)
