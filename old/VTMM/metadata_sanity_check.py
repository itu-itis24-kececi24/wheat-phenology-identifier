from multiscale_phenology import build_multiscale_daily_dataframe

df = build_multiscale_daily_dataframe("../labeling.xlsx", "../data", preferred_camera="K1")
print(df.head())
print(df["label"].value_counts().sort_index())
print(df[["macro_path", "micro_path"]].notna().mean())