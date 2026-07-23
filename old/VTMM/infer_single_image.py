import argparse
import json
import os

import torch
from PIL import Image

try:
    import torchvision.transforms as T
except Exception:  # pragma: no cover
    T = None

from multiscale_phenology import BASE_CLASSES, MultiScaleTemporalTransformer


def load_image(path: str, image_size: int = 224) -> torch.Tensor:
    if T is None:
        raise ImportError("torchvision is required for image transforms")
    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(path).convert("RGB")
    return transform(image)


def make_repeated_window(image: torch.Tensor, window_days: int) -> torch.Tensor:
    return image.unsqueeze(0).repeat(window_days, 1, 1, 1)


def main():
    parser = argparse.ArgumentParser(description="Run one-image inference with a trained full-image phenology model.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--image-path", default=None, help="Single image to use. By default treated as macro/1X.")
    parser.add_argument("--macro-path", default=None, help="Optional 1X/canopy image path.")
    parser.add_argument("--micro-path", default=None, help="Optional 10X/leaf image path.")
    parser.add_argument("--stream", choices=["macro", "micro", "both"], default="macro")
    parser.add_argument("--window-days", type=int, default=None, help="Override checkpoint window length.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.image_path and (args.macro_path or args.micro_path):
        raise ValueError("Use either --image-path or --macro-path/--micro-path, not both.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("uses_embedding_cache"):
        raise RuntimeError(
            "This checkpoint was trained from cached embeddings. "
            "Use a full-image checkpoint for infer_single_image.py."
        )

    classes = checkpoint.get("classes", BASE_CLASSES)
    window_days = args.window_days or int(checkpoint.get("window_days", 31))
    device = torch.device(args.device)

    model = MultiScaleTemporalTransformer(num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    zero_frame = torch.zeros(3, 224, 224)

    macro_image = None
    micro_image = None
    if args.image_path:
        image = load_image(args.image_path)
        if args.stream in ("macro", "both"):
            macro_image = image
        if args.stream in ("micro", "both"):
            micro_image = image
    else:
        if args.macro_path:
            macro_image = load_image(args.macro_path)
        if args.micro_path:
            micro_image = load_image(args.micro_path)

    if macro_image is None and micro_image is None:
        raise ValueError("Provide --image-path or at least one of --macro-path / --micro-path.")

    macro = make_repeated_window(macro_image if macro_image is not None else zero_frame, window_days)
    micro = make_repeated_window(micro_image if micro_image is not None else zero_frame, window_days)
    mask = torch.ones(window_days, dtype=torch.bool)

    with torch.no_grad():
        logits = model(
            macro.unsqueeze(0).to(device),
            micro.unsqueeze(0).to(device),
            mask.unsqueeze(0).to(device),
        )
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()

    top_k = min(args.top_k, len(classes))
    values, indices = torch.topk(probs, k=top_k)
    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "window_days": window_days,
        "prediction": classes[int(indices[0])],
        "confidence": float(values[0]),
        "top_k": [
            {"label": classes[int(idx)], "probability": float(prob)}
            for prob, idx in zip(values, indices)
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

