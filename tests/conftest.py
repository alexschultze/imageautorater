"""Shared helpers. The real face/CLIP/IQA models are replaced by small stand-ins, so the tests need no downloads, no GPU
and no torch weights; they exercise the real naming, clustering, sidecar and gallery logic."""
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import imageautorater.pipeline as pl

PIXEL = {"A": 30, "B": 90, "C": 150, "D": 210}  # which "person" a synthetic photo shows = its red channel
_rng = np.random.default_rng(0)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


IDENTITY = {k: _unit(_rng.normal(size=512)) for k in PIXEL}


class FakeFaces:
    """One face per photo; the face's embedding is the identity encoded in the photo's red channel."""

    def __init__(self, *a, **k):
        self.noise = np.random.default_rng(1)

    def find(self, img, min_face, min_prob=0.9):
        red = img.getpixel((0, 0))[0]
        who = min(PIXEL, key=lambda k: abs(PIXEL[k] - red))
        emb = _unit(IDENTITY[who] + self.noise.normal(size=512) * 0.01)
        return [{"box": [0.2, 0.2, 0.6, 0.6], "prob": 0.99, "emb": [round(float(x), 3) for x in emb]}]


class FakeClip:
    LABELS = pl.DEFAULT_LABELS

    def __init__(self, *a, **k):
        self.text_vecs = [_unit(np.random.default_rng(i + 5).normal(size=512)) for i in range(len(self.LABELS))]

    def image(self, img):
        return self.text_vecs[3].tolist()  # always "looks like" label #3

    def text(self, prompts):
        return [v.tolist() for v in self.text_vecs]


def make_photo(root: Path, rel: str, who: str, size: int = 100):
    """size: a real replacement of a photo has a different file size, which is what change detection relies on
    (file size + modified time in whole seconds, so that re-syncing a file does not look like an edit)."""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), (PIXEL[who], 5, 5)).save(p, format="PNG")  # PNG: exact pixel values
    return p


def make_jpeg(root: Path, rel: str, with_exif: bool = True, seed: int = 0, maker_note: bytes = b"\x01\x02\x03MAKERNOTE\x00\xff"):
    """A noisy JPEG (so the picture data is non-trivial) with ordinary camera-style EXIF and a maker note."""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    kwargs = {}
    if with_exif:
        exif = Image.Exif()
        exif[0x010F], exif[0x0110], exif[0x0112] = "TestMake", "TestModel", 6
        sub = exif.get_ifd(0x8769)
        sub[0x9003], sub[0x927C] = "2024:01:02 03:04:05", maker_note
        kwargs["exif"] = exif.tobytes()
    Image.fromarray(pixels).save(p, "JPEG", quality=85, **kwargs)
    return p


def write_results(folder: Path, spec: dict):
    """Write an iqa_results.json. spec: rel -> dict(rating=, models={name: (stars, score)}, tags=[...], people=[...]);
    'tags'/'people' left out means 'not analysed'."""
    images = {}
    for rel, s in spec.items():
        models = {m: {"rating": r, "score": sc} for m, (r, sc) in s.get("models", {}).items()}
        e = {"rating": s.get("rating"), "rating_mean": s.get("mean", s.get("rating")), "models": models}
        if "tags" in s:
            e["tags"] = {t: 0.5 for t in s["tags"]}
        if "people" in s:
            e["people"] = [{"name": n, "similarity": 0.9, "cluster": 1, "box": [0, 0, 1, 1]} for n in s["people"]]
        images[rel] = e
    (folder / pl.RESULTS_NAME).write_text(json.dumps({"mode": "absolute", "models": {}, "images": images}), encoding="utf-8")


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(pl, "nvidia_gpus", lambda: [])   # tests never depend on, or try to repair, the real GPU setup
    monkeypatch.setattr(pl, "Faces", FakeFaces)
    monkeypatch.setattr(pl, "Clip", FakeClip)
    monkeypatch.setattr(pl, "available_models", lambda: ["musiq", "topiq_nr", "clipiqa", "liqe", "nima",
                                                         "arniqa", "hyperiqa", "maniqa", "dbcnn", "niqe", "fid"])


@pytest.fixture
def cli(stubbed, tmp_path, monkeypatch):
    """Run the real command line with the stand-in models; returns (stdout, results dict of the photo folder)."""
    faces_dir = tmp_path / "_faces"

    def run(photo_folder: Path, *args):
        argv = ["imageautorater-rate", str(photo_folder), "--faces-dir", str(faces_dir), *args]
        monkeypatch.setattr("sys.argv", argv)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            pl.main()
        res = photo_folder / pl.RESULTS_NAME
        return out.getvalue(), (json.loads(res.read_text(encoding="utf-8")) if res.exists() else None)

    run.faces_dir = faces_dir
    return run


def people_of(results: dict) -> dict:
    return {k: [p["name"] for p in v.get("people", [])] for k, v in sorted(results["images"].items())}
