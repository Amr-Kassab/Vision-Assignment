"""Tests for export_model.py, inference.py and app.py.

They use a RANDOMLY initialised network with made-up normalisation stats, so they check the plumbing
(shapes, formats, error handling, preprocessing, checkpoint compatibility) - not segmentation quality.
Quality is checked on real tiles with evaluate_deployed.py.

Run from the project folder:   python -m unittest discover -s tests -v
"""
import base64
import io
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import tifffile
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import create_app                      # noqa: E402
from export_model import export_bundle          # noqa: E402
from inference import WaterSegmenter, to_chw    # noqa: E402

RNG = np.random.default_rng(0)
MIN = np.zeros(12); MAX = np.full(12, 10000.0)


def notebook_style_checkpoint(path):
    """Build the model exactly like the notebook does (3-channel smp model, first conv swapped for a
    12-channel one) and save its state_dict - proves the real checkpoint format loads at inference."""
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    old = next(m for m in model.encoder.modules() if isinstance(m, nn.Conv2d) and m.in_channels == 3)
    new = nn.Conv2d(12, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
    nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
    model.encoder.conv1 = new
    torch.save(model.state_dict(), path)


def tif_bytes(arr):
    buf = io.BytesIO(); tifffile.imwrite(buf, arr); return buf.getvalue()


def npy_bytes(arr):
    buf = io.BytesIO(); np.save(buf, arr); return buf.getvalue()


def fake_tile(h=128, w=128, c=12, channels_last=False):
    a = RNG.uniform(0, 10000, size=(c, h, w)).astype(np.float32)
    return np.transpose(a, (1, 2, 0)) if channels_last else a


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.tmp.name
        notebook_style_checkpoint(os.path.join(d, "ckpt.pt"))
        with open(os.path.join(d, "mm.json"), "w") as f:
            json.dump({"min": MIN.tolist(), "max": MAX.tolist()}, f)
        # 16 entries on purpose: the notebook saves z-stats for the 16-channel input
        with open(os.path.join(d, "z.json"), "w") as f:
            json.dump({"mean": np.linspace(0.2, 0.6, 16).tolist(), "std": np.linspace(0.2, 0.3, 16).tolist()}, f)
        cls.bundle = os.path.join(d, "bundle")
        export_bundle(os.path.join(d, "ckpt.pt"), os.path.join(d, "mm.json"), os.path.join(d, "z.json"), cls.bundle)
        cls.app = create_app(cls.bundle)
        cls.client = cls.app.test_client()
        cls.seg = cls.app.config["SEGMENTER"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def post(self, data, name="tile.tif", query="", extra=None):
        form = {"image": (io.BytesIO(data), name)}
        form.update(extra or {})
        return self.client.post("/predict" + query, data=form, content_type="multipart/form-data")


class TestExport(Base):
    def test_bundle_contents(self):
        with open(os.path.join(self.bundle, "preprocessing.json")) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["in_channels"], 12)
        self.assertEqual(len(cfg["mean"]), 12)            # 16 -> first 12
        self.assertEqual(len(cfg["weights_sha256"]), 64)
        self.assertTrue(os.path.exists(os.path.join(self.bundle, cfg["weights_file"])))

    def test_rejects_16_channel_checkpoint(self):
        d = self.tmp.name
        m = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=16, classes=1)
        torch.save(m.state_dict(), os.path.join(d, "ckpt16.pt"))
        with self.assertRaises(SystemExit):
            export_bundle(os.path.join(d, "ckpt16.pt"), os.path.join(d, "mm.json"), os.path.join(d, "z.json"),
                          os.path.join(d, "bad_bundle"))


class TestPreprocessing(Base):
    def test_matches_notebook_formula(self):
        raw = fake_tile()
        raw[:, :5, :5] = -500.0                       # below train min -> must clip to 0
        raw[:, 5:9, 5:9] = 99999.0                    # above train max -> must clip to 1
        # notebook: normalize_channels() then (img - Z_MEAN[:c]) / Z_STD[:c]
        mn = MIN.astype(np.float32).reshape(-1, 1, 1); mx = MAX.astype(np.float32).reshape(-1, 1, 1)
        x = np.clip((raw - mn) / np.maximum(mx - mn, 1e-6), 0, 1)
        mean = np.linspace(0.2, 0.6, 16)[:12].astype(np.float32).reshape(-1, 1, 1)
        std = np.linspace(0.2, 0.3, 16)[:12].astype(np.float32).reshape(-1, 1, 1)
        expected = (x - mean) / std
        np.testing.assert_allclose(self.seg.preprocess(raw), expected, rtol=1e-5, atol=1e-5)

    def test_channels_last_equals_channels_first(self):
        a = fake_tile()
        p1 = self.seg.predict_proba(a)
        p2 = self.seg.predict_proba(np.transpose(a, (1, 2, 0)))
        np.testing.assert_allclose(p1, p2, atol=1e-6)

    def test_deterministic(self):
        a = fake_tile()
        np.testing.assert_array_equal(self.seg.predict(a), self.seg.predict(a))


class TestPredictEndpoint(Base):
    def test_json_response(self):
        r = self.post(tif_bytes(fake_tile()))
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertEqual((j["height"], j["width"]), (128, 128))
        mask = np.array(Image.open(io.BytesIO(base64.b64decode(j["mask_png_base64"]))))
        self.assertEqual(mask.shape, (128, 128))
        self.assertTrue(set(np.unique(mask)) <= {0, 255})
        self.assertAlmostEqual(j["water_fraction"], (mask > 0).mean(), places=6)
        self.assertEqual(j["water_pixels"], int((mask > 0).sum()))
        for k in ("rgb_png_base64", "overlay_png_base64"):
            self.assertEqual(Image.open(io.BytesIO(base64.b64decode(j[k]))).size, (128, 128))

    def test_png_response(self):
        r = self.post(tif_bytes(fake_tile()), query="?format=png")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.mimetype, "image/png")
        img = Image.open(io.BytesIO(r.data))
        self.assertEqual((img.mode, img.size), ("L", (128, 128)))
        self.assertTrue(set(np.unique(np.array(img))) <= {0, 255})
        self.assertIn("X-Water-Fraction", r.headers)

    def test_png_matches_library_prediction(self):
        a = fake_tile()
        r = self.post(tif_bytes(a), query="?format=png")
        api_mask = (np.array(Image.open(io.BytesIO(r.data))) > 0).astype(np.uint8)
        np.testing.assert_array_equal(api_mask, self.seg.predict(a))

    def test_channels_last_tif_and_npy(self):
        a = fake_tile(channels_last=True)
        self.assertEqual(self.post(tif_bytes(a)).status_code, 200)
        self.assertEqual(self.post(npy_bytes(a), name="tile.npy").status_code, 200)

    def test_non_multiple_of_32_size_is_cropped_back(self):
        j = self.post(tif_bytes(fake_tile(100, 70))).get_json()
        self.assertEqual((j["height"], j["width"]), (100, 70))

    def test_threshold_is_applied(self):
        # Compare against the library on the same tile. (A random untrained net saturates the sigmoid, so
        # some probabilities are exactly 0.0 or 1.0 - don't assume "threshold 0 => everything is water".)
        tile = fake_tile()
        data = tif_bytes(tile)
        prob = self.seg.predict_proba(tile)
        fractions = []
        for thr in (0.1, 0.5, 0.9):
            j = self.post(data, extra={"threshold": str(thr)}).get_json()
            self.assertEqual(j["threshold"], thr)
            self.assertAlmostEqual(j["water_fraction"], float((prob > thr).mean()), places=6)
            fractions.append(j["water_fraction"])
        self.assertEqual(fractions, sorted(fractions, reverse=True))      # stricter threshold => never more water
        self.assertEqual(self.post(data, extra={"threshold": "1"}).get_json()["water_fraction"], 0.0)  # sigmoid <= 1

    def test_default_threshold_comes_from_bundle(self):
        j = self.post(tif_bytes(fake_tile())).get_json()
        self.assertEqual(j["threshold"], self.seg.threshold)

    # ---- bad input -> clean 4xx JSON errors
    def assert_error(self, resp, code):
        self.assertEqual(resp.status_code, code)
        self.assertIn("error", resp.get_json())

    def test_no_file(self):
        self.assert_error(self.client.post("/predict", data={}, content_type="multipart/form-data"), 400)

    def test_wrong_channel_count(self):
        self.assert_error(self.post(tif_bytes(fake_tile(c=11))), 400)
        self.assert_error(self.post(tif_bytes(fake_tile(c=3))), 400)

    def test_wrong_ndim(self):
        self.assert_error(self.post(npy_bytes(np.zeros((128, 128), np.float32)), name="x.npy"), 400)

    def test_unsupported_extension(self):
        self.assert_error(self.post(b"hello", name="tile.png"), 400)
        self.assert_error(self.post(b"hello", name="noext"), 400)

    def test_corrupt_files(self):
        self.assert_error(self.post(b"this is not a tiff", name="bad.tif"), 400)
        self.assert_error(self.post(b"this is not an npy", name="bad.npy"), 400)

    def test_pickled_npy_refused(self):
        buf = io.BytesIO(); np.save(buf, np.array([{"a": 1}], dtype=object), allow_pickle=True)
        self.assert_error(self.post(buf.getvalue(), name="evil.npy"), 400)

    def test_nan_rejected(self):
        a = fake_tile(); a[0, 0, 0] = np.nan
        self.assert_error(self.post(tif_bytes(a)), 400)

    def test_oversize_image_rejected(self):
        self.assert_error(self.post(npy_bytes(np.zeros((12, 1100, 40), np.float32)), name="big.npy"), 400)

    def test_bad_threshold_and_format(self):
        data = tif_bytes(fake_tile())
        self.assert_error(self.post(data, extra={"threshold": "abc"}), 400)
        self.assert_error(self.post(data, extra={"threshold": "1.5"}), 400)
        self.assert_error(self.post(data, query="?format=gif"), 400)

    def test_upload_too_large(self):
        self.app.config["MAX_CONTENT_LENGTH"] = 1000
        try:
            self.assert_error(self.post(tif_bytes(fake_tile())), 413)
        finally:
            self.app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


class TestOtherRoutes(Base):
    def test_health(self):
        j = self.client.get("/health").get_json()
        self.assertEqual((j["status"], j["in_channels"], j["encoder"]), ("ok", 12, "resnet34"))

    def test_frontend_served(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Water segmentation", r.data)
        r.close()

    def test_json_404_and_405(self):
        self.assertEqual(self.client.get("/nope").status_code, 404)
        self.assertIn("error", self.client.get("/nope").get_json())
        self.assertEqual(self.client.get("/predict").status_code, 405)


if __name__ == "__main__":
    unittest.main(verbosity=2)
