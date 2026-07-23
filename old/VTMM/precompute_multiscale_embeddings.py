import argparse
import os
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from multiscale_phenology import (
    BASE_CLASSES,
    ViTBackboneFeatureExtractor,
    build_multiscale_daily_dataframe,
)

try:
    import torchvision.transforms as T
except Exception:  # pragma: no cover
    T = None


def path_key(path: str) -> str:
    return os.path.abspath(os.path.normpath(str(path)))


class ImagePathDataset(Dataset):
    def __init__(self, paths: List[str], image_size: int = 224):
        if T is None:
            raise ImportError("torchvision is required for image transforms")
        self.paths = [path_key(p) for p in paths]
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return path, self.transform(image)


@torch.no_grad()
def encode_paths(model, paths: List[str], batch_size: int, num_workers: int, device: torch.device) -> dict:
    dataset = ImagePathDataset(paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    encoded = {}
    model.eval()
    for batch_paths, images in loader:
        images = images.to(device, non_blocking=True)
        features = model(images).detach().cpu()
        for path, feature in zip(batch_paths, features):
            encoded[path_key(path)] = feature
    return encoded


def unique_existing_paths(series: pd.Series) -> List[str]:
    paths = []
    for value in series.dropna().tolist():
        key = path_key(value)
        if os.path.isfile(key):
            paths.append(key)
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(description="Precompute frozen ViT image embeddings for multi-scale phenology training.")
    parser.add_argument("--excel-path", default="labeling.xlsx")
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--out-dir", default="results_multiscale")
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--transition-days", type=int, default=2)
    parser.add_argument("--preplant-days", type=int, default=30, help="Number of days before seeding to keep as OffSeason.")
    parser.add_argument("--postharvest-days", type=int, default=30, help="Number of days after harvest to keep as OffSeason.")
    parser.add_argument("--camera", default="K1", help="Camera folder to use, e.g. K1 or K2. Use ALL for both.")
    parser.add_argument("--ignore-status-csv", action="store_true")
    parser.add_argument("--image-backbone", default="vit_b_16")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = args.cache_path or os.path.join(args.out_dir, "vit_embeddings.pt")
    metadata_path = os.path.join(args.out_dir, "multiscale_daily_metadata.csv")
    device = torch.device(args.device)

    daily_df = build_multiscale_daily_dataframe(
        args.excel_path,
        args.data_path,
        include_preplant_days=args.preplant_days,
        include_postharvest_days=args.postharvest_days,
        transition_days=args.transition_days,
        classes=BASE_CLASSES,
        preferred_camera=None if args.camera.upper() == "ALL" else args.camera,
        use_status_csv=not args.ignore_status_csv,
    )
    if daily_df.empty:
        raise RuntimeError("No paired daily rows were created. Check data_path, folder names, and filename dates.")

    daily_df.to_csv(metadata_path, index=False)
    macro_paths = unique_existing_paths(daily_df["macro_path"])
    micro_paths = unique_existing_paths(daily_df["micro_path"])
    print(f"Saved metadata: {metadata_path}")
    print(f"Macro images: {len(macro_paths)}")
    print(f"Micro images: {len(micro_paths)}")

    model = ViTBackboneFeatureExtractor(args.image_backbone, pretrained=args.pretrained).to(device)
    macro_embeddings = encode_paths(model, macro_paths, args.batch_size, args.num_workers, device)
    micro_embeddings = encode_paths(model, micro_paths, args.batch_size, args.num_workers, device)

    torch.save(
        {
            "feature_dim": model.out_dim,
            "image_backbone": args.image_backbone,
            "pretrained": args.pretrained,
            "macro": macro_embeddings,
            "micro": micro_embeddings,
        },
        cache_path,
    )
    print(f"Saved embedding cache: {cache_path}")


if __name__ == "__main__":
    main()
