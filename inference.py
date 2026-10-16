"""Inference for the Week 2 model: ImageNet-pretrained ResNet34 U-Net, 12 input channels.

Usage as a library:
    seg = WaterSegmenter("model_bundle")
    mask = seg.predict(raw_array)          # (H, W) uint8, 1 = water, 0 = not water

Usage from the command line:
    python inference.py tile.tif --bundle model_bundle --out mask.png

The input must be the RAW 12-band tile (same values as the .tif files in the training
dataset). All preprocessing (min/max scaling, z-score) is done in here, using the
statistics saved in the bundle, so the caller never has to normalise anything.
"""
import argparse
import io
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import tifffile
from PIL import Image

NUM_CHANNELS = 12
DIVISOR = 32          # a ResNet U-Net halves the image 5 times, so H and W must be multiples of 32
MAX_SIDE = 1024       # refuse absurdly large inputs (memory); training tiles are 128x128
MIN_SIDE = 8
RGB_BANDS = (3, 2, 1)  # Red, Green, Blue positions in the 12-band stack (used only for previews)
WEIGHTS_FILE = "water_unet_resnet34_12ch.pt"
PREPROC_FILE = "preprocessing.json"


class InputError(ValueError):
    """The uploaded data cannot be used (wrong shape, wrong type, NaN, ...). Maps to HTTP 400."""


# --------------------------------------------------------------------------------------
# Reading / validating input
# --------------------------------------------------------------------------------------
def to_chw(arr):
    """Return a channel-first (12, H, W) float32 array, or raise InputError.

    Same convention as the training notebook: if the first axis is 12 it is already (C,H,W);
    otherwise a last axis of 12 is treated as channels-last (H,W,C) and transposed.
    """
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise InputError(f"Expected a 3-D array with 12 channels, got shape {tuple(arr.shape)}.")
    if arr.shape[0] == NUM_CHANNELS:
        pass
    elif arr.shape[-1] == NUM_CHANNELS:
        arr = np.transpose(arr, (2, 0, 1))
    else:
        raise InputError(f"Expected 12 channels (shape (12,H,W) or (H,W,12)), got shape {tuple(arr.shape)}.")
    if not (np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)):
        raise InputError(f"Array dtype {arr.dtype} is not numeric.")
    arr = arr.astype(np.float32)
    h, w = arr.shape[1:]
    if min(h, w) < MIN_SIDE or max(h, w) > MAX_SIDE:
        raise InputError(f"Image size {h}x{w} is outside the supported range "
                         f"({MIN_SIDE}..{MAX_SIDE} pixels per side; training tiles are 128x128).")
    if not np.isfinite(arr).all():
        raise InputError("The image contains NaN or infinite values. Fill or remove them before uploading.")
    return arr


def read_array(data, filename):
    """Decode uploaded bytes (.tif/.tiff/.npy) into a raw array, or raise InputError."""
    ext = os.path.splitext(filename or "")[1].lower()
    try:
        if ext in (".tif", ".tiff"):
            return tifffile.imread(io.BytesIO(data))
        if ext == ".npy":
            return np.load(io.BytesIO(data), allow_pickle=False)   # never unpickle uploads
    except Exception as e:
        raise InputError(f"Could not read '{filename}' as a {ext} file: {type(e).__name__}.") from e
    raise InputError(f"Unsupported file type '{ext}'. Upload a 12-band .tif/.tiff or .npy file.")


def stretch(band, p_lo=2, p_hi=98):
    lo, hi = np.percentile(band, [p_lo, p_hi])
    return np.zeros_like(band) if hi <= lo else np.clip((band - lo) / (hi - lo), 0, 1)


def rgb_preview(raw_chw):
    """(H,W,3) uint8 true-colour preview from the raw bands (2-98 percentile stretch)."""
    rgb = np.stack([stretch(raw_chw[i]) for i in RGB_BANDS], axis=-1)
    return (rgb * 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def build_model(encoder="resnet34", in_channels=NUM_CHANNELS):
    """The same architecture the notebook trained. smp builds the first conv with `in_channels`
    inputs directly; the weights are then overwritten by the checkpoint, so no ImageNet download
    (and no internet) is needed here."""
    return smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=in_channels, classes=1)


class WaterSegmenter:
    def __init__(self, bundle_dir, device=None):
        with open(os.path.join(bundle_dir, PREPROC_FILE)) as f:
            self.cfg = json.load(f)
        c = self.cfg
        for key in ("min", "max", "mean", "std"):
            if len(c[key]) != NUM_CHANNELS:
                raise ValueError(f"preprocessing.json: '{key}' must have {NUM_CHANNELS} values, has {len(c[key])}")
        self.ch_min = np.array(c["min"], dtype=np.float32).reshape(-1, 1, 1)
        self.ch_max = np.array(c["max"], dtype=np.float32).reshape(-1, 1, 1)
        self.mean = np.array(c["mean"], dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.array(c["std"], dtype=np.float32).reshape(-1, 1, 1)
        self.threshold = float(c.get("threshold", 0.5))

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(c.get("encoder", "resnet34"), NUM_CHANNELS)
        state = torch.load(os.path.join(bundle_dir, c.get("weights_file", WEIGHTS_FILE)),
                           map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    # -- preprocessing: identical to the training notebook (min/max -> clip -> z-score) ----------
    def preprocess(self, raw_chw):
        """raw_chw: (12,H,W) RAW values -> standardised float32 network input."""
        x = np.clip((raw_chw - self.ch_min) / np.maximum(self.ch_max - self.ch_min, 1e-6), 0, 1)
        return ((x - self.mean) / self.std).astype(np.float32)

    @torch.no_grad()
    def predict_proba(self, raw):
        """raw: array-like in any accepted layout. Returns (H,W) float32 water probability."""
        raw = to_chw(raw)
        h, w = raw.shape[1:]
        x = torch.from_numpy(self.preprocess(raw)).unsqueeze(0).to(self.device)
        ph, pw = (-h) % DIVISOR, (-w) % DIVISOR
        if ph or pw:
            mode = "reflect" if (ph < h and pw < w) else "replicate"
            x = F.pad(x, (0, pw, 0, ph), mode=mode)
        prob = torch.sigmoid(self.model(x))[0, 0, :h, :w]
        return prob.float().cpu().numpy()

    def predict(self, raw, threshold=None):
        """Returns (H,W) uint8 mask: 1 = water, 0 = not water."""
        t = self.threshold if threshold is None else float(threshold)
        return (self.predict_proba(raw) > t).astype(np.uint8)


def mask_to_png_bytes(mask01):
    """0/1 mask -> 8-bit PNG with values 0 (not water) / 255 (water)."""
    buf = io.BytesIO()
    Image.fromarray((mask01 * 255).astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description="Run water segmentation on one 12-band tile.")
    ap.add_argument("image", help="path to a 12-band .tif/.tiff/.npy tile")
    ap.add_argument("--bundle", default="model_bundle")
    ap.add_argument("--out", default="mask.png", help="output PNG path (0 = not water, 255 = water)")
    ap.add_argument("--threshold", type=float, default=None, help="probability threshold (default: from bundle)")
    args = ap.parse_args()

    with open(args.image, "rb") as f:
        raw = read_array(f.read(), args.image)
    seg = WaterSegmenter(args.bundle)
    mask = seg.predict(raw, args.threshold)
    with open(args.out, "wb") as f:
        f.write(mask_to_png_bytes(mask))
    print(f"{args.image}: {mask.shape[0]}x{mask.shape[1]} mask written to {args.out} "
          f"(water fraction {mask.mean():.3f})")


if __name__ == "__main__":
    main()
