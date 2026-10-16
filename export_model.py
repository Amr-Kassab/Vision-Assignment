"""Package the trained 12-channel pretrained ResNet34 U-Net into a self-contained deployment bundle.

Inputs (all produced by the Week 2 notebook, in its `week2_outputs/` folder):
    pretrained_resnet34_u_net_12ch.pt   the weights (a state_dict)
    channel_minmax.json                 per-channel min/max of the TRAIN set (12 values each)
    input_zscore_stats.json             per-channel mean/std of the final network input (16 values; the first 12 are used)

Output folder (default model_bundle/):
    water_unet_resnet34_12ch.pt         weights, verified to load into the inference architecture
    preprocessing.json                  stats + threshold + architecture info needed at inference time

Example:
    python export_model.py --checkpoint week2_outputs/pretrained_resnet34_u_net_12ch.pt \
        --minmax week2_outputs/channel_minmax.json --zstats week2_outputs/input_zscore_stats.json
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch
import segmentation_models_pytorch as smp

from inference import NUM_CHANNELS, WEIGHTS_FILE, PREPROC_FILE, build_model

BAND_NAMES = ['Coastal aerosol', 'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2',
              'QA Band', 'Merit DEM', 'Copernicus DEM', 'ESA WorldCover', 'Water occurrence probability']


def load_state_dict(path):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model_state_dict"):          # tolerate wrapped checkpoints
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
    if not isinstance(obj, dict):
        raise SystemExit(f"{path} does not contain a state_dict.")
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in obj.items()}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_bundle(checkpoint, minmax_path, zstats_path, out_dir, threshold=0.5, encoder="resnet34"):
    state = load_state_dict(checkpoint)

    # 1. Make sure this really is the 12-channel model.
    first = state.get("encoder.conv1.weight")
    if first is None:
        raise SystemExit("No 'encoder.conv1.weight' in the checkpoint - is this a ResNet U-Net checkpoint?")
    if first.shape[1] != NUM_CHANNELS:
        raise SystemExit(f"First conv takes {first.shape[1]} input channels, expected {NUM_CHANNELS}. "
                         f"Did you pick the 16-channel checkpoint by mistake?")

    # 2. Strict load into the inference architecture + smoke test.
    model = build_model(encoder, NUM_CHANNELS)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, NUM_CHANNELS, 128, 128))
    assert out.shape == (1, 1, 128, 128) and torch.isfinite(out).all(), "smoke test forward pass failed"

    # 3. Preprocessing statistics.
    with open(minmax_path) as f:
        mm = json.load(f)
    with open(zstats_path) as f:
        zs = json.load(f)
    if len(mm["min"]) != NUM_CHANNELS or len(mm["max"]) != NUM_CHANNELS:
        raise SystemExit(f"channel_minmax.json must have {NUM_CHANNELS} values per key.")
    if len(zs["mean"]) < NUM_CHANNELS:
        raise SystemExit(f"input_zscore_stats.json has only {len(zs['mean'])} channels, need >= {NUM_CHANNELS}.")
    # The z-score stats were computed on the 16-channel input; channels 0..11 are exactly the 12 min/max-scaled
    # bands, so the first 12 entries are also the correct stats for the 12-channel model.

    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, WEIGHTS_FILE)
    torch.save(model.state_dict(), weights_path)

    cfg = {
        "architecture": "Unet",
        "encoder": encoder,
        "in_channels": NUM_CHANNELS,
        "weights_file": WEIGHTS_FILE,
        "weights_sha256": sha256(weights_path),
        "smp_version": smp.__version__,
        "torch_version": torch.__version__,
        "threshold": threshold,
        "band_names": BAND_NAMES,
        "min": [float(v) for v in mm["min"]],
        "max": [float(v) for v in mm["max"]],
        "mean": [float(v) for v in zs["mean"][:NUM_CHANNELS]],
        "std": [float(v) for v in zs["std"][:NUM_CHANNELS]],
        "preprocessing": "x = clip((raw - min) / (max - min), 0, 1); x = (x - mean) / std",
    }
    with open(os.path.join(out_dir, PREPROC_FILE), "w") as f:
        json.dump(cfg, f, indent=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Exported to {out_dir}/  ({n_params/1e6:.2f}M parameters, "
          f"{os.path.getsize(weights_path)/1e6:.1f} MB, sha256 {cfg['weights_sha256'][:12]}...)")
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--minmax", required=True)
    ap.add_argument("--zstats", required=True)
    ap.add_argument("--out", default="model_bundle")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--encoder", default="resnet34")
    a = ap.parse_args()
    export_bundle(a.checkpoint, a.minmax, a.zstats, a.out, a.threshold, a.encoder)


if __name__ == "__main__":
    main()
