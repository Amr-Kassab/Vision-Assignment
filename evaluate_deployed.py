"""Check that the DEPLOYED pipeline (bundle + inference.py) reproduces the notebook's metrics.

This is the most important deployment test: if preprocessing at inference differs from training, the
API would still return masks, just wrong ones. Run it on real tiles with their ground-truth labels.

    python evaluate_deployed.py --images data/images --labels data/labels --stems week2_outputs/test_stems.json

--stems (optional) is a JSON list of tile names to use, e.g. the notebook's test split. Without it every
matched tile is used - including tiles the model trained on, so the numbers will be optimistic.
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

from inference import WaterSegmenter, read_array


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder with .tif tiles")
    ap.add_argument("--labels", required=True, help="folder with .png masks (>0 = water)")
    ap.add_argument("--bundle", default="model_bundle")
    ap.add_argument("--stems", default=None, help="optional JSON list of tile names (no extension)")
    ap.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args()

    imgs = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(a.images, "*.tif"))}
    lbls = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(a.labels, "*.png"))}
    stems = sorted(set(imgs) & set(lbls))
    if a.stems:
        with open(a.stems) as f:
            wanted = set(json.load(f))
        stems = [s for s in stems if s in wanted]
    if not stems:
        raise SystemExit("No matching image/label pairs found.")

    seg = WaterSegmenter(a.bundle)
    tp = fp = fn = tn = 0
    for s in stems:
        with open(imgs[s], "rb") as f:
            raw = read_array(f.read(), imgs[s])
        pred = seg.predict(raw, a.threshold).astype(bool)
        gt = np.array(Image.open(lbls[s])) > 0
        tp += int((pred & gt).sum()); fp += int((pred & ~gt).sum())
        fn += int((~pred & gt).sum()); tn += int((~pred & ~gt).sum())

    eps = 1e-9
    prec, rec = tp / (tp + fp + eps), tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    iou = tp / (tp + fp + fn + eps)
    print(f"Tiles: {len(stems)}")
    print(f"Precision {prec:.4f} | Recall {rec:.4f} | F1 {f1:.4f} | IoU {iou:.4f}")
    if rec > 0.99 and abs(prec - iou) < 0.01:
        print("WARNING: recall ~1 and precision ~ IoU means the model predicts 'water' everywhere. "
              "Check that channel_minmax.json / input_zscore_stats.json come from the same run as the weights.")


if __name__ == "__main__":
    main()
