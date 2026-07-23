"""Load a separately saved image-backbone state dictionary."""

from collections.abc import Mapping
from pathlib import Path

import torch


_WRAPPER_KEYS = ("backbone_state_dict", "state_dict", "model")
_PREFIXES = (
    "module.extractor.backbone.",
    "_orig_mod.extractor.backbone.",
    "extractor.backbone.",
    "module.encoder.backbone.",
    "encoder.backbone.",
    "module.backbone.",
    "backbone.",
    "module.",
    "_orig_mod.",
    "",
)


def _tensor_state(payload):
    if not isinstance(payload, Mapping):
        raise ValueError("The selected .pt file must contain a PyTorch state dictionary.")
    for key in _WRAPPER_KEYS:
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            payload = candidate
            break
    state = {str(key): value for key, value in payload.items() if torch.is_tensor(value)}
    if not state:
        raise ValueError("The selected .pt file does not contain tensor weights.")
    return state


def read_backbone_checkpoint(weights_path: str):
    """Read a backbone checkpoint once and recover its architecture source when recorded."""
    path = Path(weights_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image-backbone weights not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_source = None
    if isinstance(payload, Mapping):
        args = payload.get("args")
        if isinstance(args, Mapping):
            model_source = args.get("image_backbone")
        model_source = payload.get("image_backbone", model_source)
    return payload, str(model_source) if model_source else None


def infer_dinov3_config(payload):
    """Infer the DINOv3 ViT architecture needed to load a standalone state dict."""
    source = _tensor_state(payload)

    def find(suffix):
        matches = [value for key, value in source.items() if key.endswith(suffix)]
        return matches[0] if matches else None

    cls_token = find("backbone.embeddings.cls_token")
    patch_weight = find("backbone.embeddings.patch_embeddings.weight")
    register_tokens = find("backbone.embeddings.register_tokens")
    up_proj = find("backbone.model.layer.0.mlp.up_proj.weight")
    if cls_token is None or patch_weight is None or up_proj is None:
        return None
    layer_ids = {
        int(key.split(".layer.", 1)[1].split(".", 1)[0])
        for key in source
        if ".backbone.model.layer." in key
    }
    hidden_size = int(cls_token.shape[-1])
    return {
        "hidden_size": hidden_size,
        "intermediate_size": int(up_proj.shape[0]),
        "num_hidden_layers": max(layer_ids) + 1,
        "num_attention_heads": hidden_size // 64,
        "num_register_tokens": int(register_tokens.shape[1]) if register_tokens is not None else 0,
        "patch_size": int(patch_weight.shape[-1]),
        "image_size": 224,
    }


def load_backbone_weights(encoder, weights_path: str, payload=None) -> None:
    """Load raw or wrapped weights into ``encoder.backbone``."""
    if payload is None:
        payload, _ = read_backbone_checkpoint(weights_path)
    source = _tensor_state(payload)
    target_keys = set(encoder.backbone.state_dict())
    candidates = []
    for prefix in _PREFIXES:
        candidate = {
            key[len(prefix) :]: value
            for key, value in source.items()
            if key.startswith(prefix) and key[len(prefix) :] in target_keys
        }
        candidates.append((len(candidate), prefix, candidate))
    matched, prefix, state = max(candidates, key=lambda item: item[0])
    if matched == 0:
        raise ValueError(
            "The selected file has no weights matching the chosen image-backbone architecture."
        )
    if matched != len(target_keys):
        raise ValueError(
            f"Image-backbone weights are incomplete: matched {matched}/{len(target_keys)} "
            f"tensors using prefix {prefix!r}."
        )
    encoder.backbone.load_state_dict(state, strict=True)
