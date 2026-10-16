# Water Segmentation - Deployment (Week 3)

Serves the **ImageNet-pretrained ResNet34 U-Net with 12 input channels** (the best model from Weeks 1-2) as a
Flask REST API with a small web page. Upload a 12-band tile, get back a per-pixel water mask.

```
water_api/
  water_segmentation_kaggle.ipynb  complete self-contained notebook to run this app on Kaggle
  export_model.py       trained checkpoint + stats  ->  model_bundle/  (weights + preprocessing.json)
  inference.py          WaterSegmenter class + command-line inference
  app.py                Flask API (/predict, /health) and the frontend route
  static/index.html     the web page (no external libraries, works offline)
  evaluate_deployed.py  re-computes IoU/F1 through the deployed pipeline (the real correctness check)
  tests/test_api.py     25 automated tests (plumbing, preprocessing, error handling)
  requirements.txt
```

---

## ⚡ Quickstart: Run directly on Kaggle

You can run this entire application directly on Kaggle with public browser access:

1. In Kaggle, create a new notebook (or click **File > Upload Notebook**) and upload `water_segmentation_kaggle.ipynb`.
2. Turn **Internet ON** in the notebook settings (right sidebar: **Settings ⚙️ -> Internet -> On**).
3. (Optional) Select **GPU (T4 x2 or P100)** for fast inference (~10-20 ms).
4. Run all cells! The notebook will:
   - Self-generate the entire app structure and frontend in `/kaggle/working/water_app`.
   - Auto-detect your model weights (or initialize a demo model if not attached yet).
   - Launch the Flask server and expose it over HTTPS via **Cloudflare Tunnel (`cloudflared`)**.
   - Embed the live interactive web UI inside the notebook and run API tests with visualizations.

---

## 1. Get the trained files out of Kaggle

At the end of the Week 2 notebook, the save cell writes a folder `week2_outputs/`. Download these three files
(Kaggle: *Output* tab of the notebook, or the file browser on the right):

| File | What it is |
|---|---|
| `pretrained_resnet34_u_net_12ch.pt` | the trained weights |
| `channel_minmax.json` | per-channel min/max from the training set (preprocessing step 1) |
| `input_zscore_stats.json` | per-channel mean/std (preprocessing step 2) |

They must all come from the **same notebook run**. Weights trained with one set of statistics will give wrong masks
with another.

*Optional, but recommended:* to check the API reproduces the notebook's test-set numbers, add this one line at the
end of the save cell, re-run it, and also download `test_stems.json`:

```python
json.dump(test_stems, open(os.path.join(SAVE_DIR, 'test_stems.json'), 'w'))
```

## 2. Install and export

```bash
pip install -r requirements.txt

python export_model.py \
    --checkpoint week2_outputs/pretrained_resnet34_u_net_12ch.pt \
    --minmax     week2_outputs/channel_minmax.json \
    --zstats     week2_outputs/input_zscore_stats.json
```

This creates `model_bundle/` containing `water_unet_resnet34_12ch.pt` and `preprocessing.json`. The script refuses
the wrong checkpoint (for example the 16-channel one), loads the weights strictly into the inference architecture,
and runs a smoke-test forward pass. No internet is needed at inference time (no ImageNet download).

## 3. Verify the pipeline on real tiles (do this before trusting the API)

```bash
python evaluate_deployed.py --images data/images --labels data/labels \
    --stems week2_outputs/test_stems.json
```

The IoU/F1 printed here should match the notebook's test-set numbers for this model (tiny differences from float
rounding are normal). If they do not, the preprocessing statistics do not belong to these weights. A telltale
sign of that mistake is *recall about 1.0 and precision about equal to IoU*, which means "everything is water".
Without `--stems` all tiles are used, including training tiles, so the numbers will look better than the test set.

## 4. Run the API

```bash
python app.py --bundle model_bundle          # http://127.0.0.1:5000
```

Open <http://127.0.0.1:5000> for the web page, or call the API directly.

### Command line

```bash
curl -F image=@tile.tif http://127.0.0.1:5000/predict                    # JSON
curl -F image=@tile.tif "http://127.0.0.1:5000/predict?format=png" -o mask.png   # just the mask
curl -F image=@tile.tif -F threshold=0.4 http://127.0.0.1:5000/predict    # custom threshold
python inference.py tile.tif --bundle model_bundle --out mask.png         # no server needed
```

### Python

```python
import requests, base64
r = requests.post("http://127.0.0.1:5000/predict", files={"image": open("tile.tif", "rb")})
r.raise_for_status()
info = r.json()
print(info["water_fraction"], info["inference_ms"])
open("mask.png", "wb").write(base64.b64decode(info["mask_png_base64"]))
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | the web page |
| GET | `/health` | model info: architecture, encoder, channels, threshold, device, weights SHA-256 |
| POST | `/predict` | multipart form, file field **`image`** (`.tif`, `.tiff`, `.npy`; 12 bands) |

`/predict` options (form field or query string): `threshold` (0-1, default 0.5, from `preprocessing.json`) and
`format` (`json` default, or `png`).

**JSON response:** `height`, `width`, `threshold`, `water_pixels`, `water_fraction`, `inference_ms`, plus three
base64 PNGs: `mask_png_base64` (0 = not water, 255 = water), `rgb_png_base64` (true-colour preview) and
`overlay_png_base64`.
**PNG response:** the 8-bit mask, with headers `X-Water-Fraction` and `X-Threshold`.

**Errors** are JSON `{"error": "..."}` with status 400 (bad file: wrong channel count, NaN values, unreadable,
wrong type, bad threshold), 413 (over 32 MB) or 500.

### Input rules

- Send the **raw** 12-band tile, exactly like the `.tif` files in the training data. Do **not** normalise it
  yourself: min/max scaling and z-scoring are applied inside the API with the saved statistics.
- Layout can be `(12, H, W)` or `(H, W, 12)`. Band order is the dataset's: Coastal, Blue, Green, Red, NIR, SWIR1,
  SWIR2, QA, Merit DEM, Copernicus DEM, ESA WorldCover, Water occurrence.
- The model was trained on 128x128 tiles. Other sizes from 8 to 1024 pixels per side are accepted (the image is
  padded to a multiple of 32 and cropped back), but accuracy on very different sizes has not been measured.
- NaN or infinite values are rejected rather than silently changed.

## 5. Tests

```bash
python -m unittest discover -s tests -v
```

The tests use a randomly initialised network with made-up statistics. They verify the plumbing (shapes, formats,
thresholding, error handling, that preprocessing matches the notebook formula, that a notebook-style checkpoint
loads), **not** segmentation quality. Quality is what `evaluate_deployed.py` is for.

## Notes

- **Security:** the server binds to `127.0.0.1` by default and has no authentication. Do not expose it to the
  internet as is. Uploads are parsed in memory, never saved, and `.npy` files are loaded with pickling disabled.
- **Production-style serving:** `MODEL_DIR=model_bundle gunicorn -w 1 --threads 4 -b 0.0.0.0:5000 "app:create_app()"`
  (install `gunicorn` first; one worker because the model lives in memory).
- **GitHub:** the weights file is about 98 MB, close to GitHub's 100 MB limit and over its 50 MB warning
  threshold. Use Git LFS or a GitHub Release asset for `model_bundle/*.pt`. The included `.gitignore` leaves it out.
- **Versions:** the weights were trained with `segmentation-models-pytorch` 0.5.0. A different major version of
  that library may name layers differently and fail to load.
