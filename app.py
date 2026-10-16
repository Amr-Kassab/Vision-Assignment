"""Flask REST API for water segmentation (12-channel pretrained ResNet34 U-Net).

Run (development):
    python app.py --bundle model_bundle            # http://127.0.0.1:5000
Run (production-style, one worker because the model lives in memory):
    MODEL_DIR=model_bundle gunicorn -w 1 --threads 4 -b 0.0.0.0:5000 "app:create_app()"

Endpoints
    GET  /          the HTML frontend
    GET  /health    model + service info (JSON)
    POST /predict   multipart/form-data, file field "image" (.tif/.tiff/.npy, 12 bands)
                    optional field/query "threshold" in [0,1]
                    ?format=json (default): JSON with the mask, a preview and statistics
                    ?format=png           : the raw mask as an 8-bit PNG (0 = not water, 255 = water)
"""
import argparse
import base64
import io
import logging
import os
import threading
import time

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image
from werkzeug.exceptions import HTTPException

from inference import InputError, WaterSegmenter, mask_to_png_bytes, read_array, rgb_preview, to_chw

log = logging.getLogger("water_api")
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
HERE = os.path.dirname(os.path.abspath(__file__))


def _png_b64(arr, mode):
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _overlay(rgb, mask01):
    """Blend a semi-transparent blue over the water pixels of the RGB preview."""
    out = rgb.astype(np.float32)
    water = mask01.astype(bool)
    out[water] = 0.45 * out[water] + 0.55 * np.array([30, 110, 255], dtype=np.float32)
    return out.astype(np.uint8)


def create_app(bundle_dir=None):
    bundle_dir = bundle_dir or os.environ.get("MODEL_DIR", "model_bundle")
    segmenter = WaterSegmenter(bundle_dir)
    lock = threading.Lock()        # one forward pass at a time; keeps memory use predictable

    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["SEGMENTER"] = segmenter

    # ---------------------------------------------------------------- errors -> JSON
    @app.errorhandler(InputError)
    def _input_error(e):
        return jsonify(error=str(e)), 400

    @app.errorhandler(HTTPException)
    def _http_error(e):
        msg = "Upload too large (limit %d MB)." % (MAX_UPLOAD_BYTES // 2**20) if e.code == 413 else e.description
        return jsonify(error=msg), e.code

    @app.errorhandler(Exception)
    def _unexpected(e):
        log.exception("Unhandled error")
        return jsonify(error="Internal server error."), 500

    # ---------------------------------------------------------------- routes
    @app.get("/")
    def index():
        return send_from_directory(os.path.join(HERE, "static"), "index.html")

    @app.get("/health")
    def health():
        c = segmenter.cfg
        return jsonify(status="ok", architecture=c.get("architecture"), encoder=c.get("encoder"),
                       in_channels=c.get("in_channels"), threshold=segmenter.threshold,
                       device=str(segmenter.device), weights_sha256=c.get("weights_sha256"))

    @app.post("/predict")
    def predict():
        f = request.files.get("image")
        if f is None or not f.filename:
            raise InputError("No file uploaded. Send multipart/form-data with a file in the field 'image'.")

        thr_raw = request.form.get("threshold", request.args.get("threshold"))
        threshold = None
        if thr_raw not in (None, ""):
            try:
                threshold = float(thr_raw)
            except ValueError:
                raise InputError("'threshold' must be a number between 0 and 1.")
            if not (0.0 <= threshold <= 1.0):
                raise InputError("'threshold' must be between 0 and 1.")

        fmt = request.args.get("format", "json").lower()
        if fmt not in ("json", "png"):
            raise InputError("'format' must be 'json' or 'png'.")

        raw = to_chw(read_array(f.read(), f.filename))
        t0 = time.perf_counter()
        with lock:
            mask = segmenter.predict(raw, threshold)
        ms = (time.perf_counter() - t0) * 1000
        used_thr = segmenter.threshold if threshold is None else threshold
        frac = float(mask.mean())

        if fmt == "png":
            resp = send_file(io.BytesIO(mask_to_png_bytes(mask)), mimetype="image/png")
            resp.headers["X-Water-Fraction"] = f"{frac:.6f}"
            resp.headers["X-Threshold"] = f"{used_thr}"
            return resp

        rgb = rgb_preview(raw)
        return jsonify(
            height=int(mask.shape[0]), width=int(mask.shape[1]),
            threshold=used_thr,
            water_pixels=int(mask.sum()), water_fraction=frac,
            inference_ms=round(ms, 1),
            mask_png_base64=_png_b64(mask * 255, "L"),
            rgb_png_base64=_png_b64(rgb, "RGB"),
            overlay_png_base64=_png_b64(_overlay(rgb, mask), "RGB"),
        )

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.environ.get("MODEL_DIR", "model_bundle"))
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 only if you want other machines to reach it")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    create_app(args.bundle).run(host=args.host, port=args.port, threaded=True)
