import json
import re

import pytest
from conftest import make_photo

import imageautorater.gallery as gal
import imageautorater.pipeline as pl


def test_preset_follows_the_hardware_and_models_override_it(stubbed, tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_run_models(folder, paths, models, device, max_side, cache, cache_path, **kw):
        seen.update(models=list(models), max_side=max_side)
        for m in models:
            cache["scores"][m] = {"a.png": 50.0}
            cache["models"][m] = {"lower_better": False, "range": [0, 100]}

    monkeypatch.setattr(pl, "run_models", fake_run_models)
    p = tmp_path / "P"
    make_photo(p, "a.png", "A")

    def run(*args):
        monkeypatch.setattr("sys.argv", ["x", str(p), "--no-faces", "--no-classify", *args])
        pl.main()

    run("--device", "cpu")
    assert seen["models"] == pl.CPU_MODELS and seen["max_side"] == 1024
    run("--device", "cpu", "--preset", "gpu", "--max-side", "1024")
    assert seen["models"] == pl.GPU_MODELS
    run("--device", "cpu", "--preset", "all", "--max-side", "1024")
    assert "fid" not in seen["models"] and "musiq" in seen["models"]      # not a per-photo scorer
    run("--device", "cpu", "--models", "niqe", "--max-side", "1024")
    assert seen["models"] == ["niqe"]


def test_norm_score_direction_and_range():
    assert gal.norm_score(80, {"range": [0, 100]}, (0, 1)) == 80
    assert gal.norm_score(5, {"range": [0, 20], "lower_better": True}, (0, 1)) == 75
    assert gal.norm_score(3, {}, (2, 4)) == 50          # unknown range: observed min/max
    assert gal.norm_score(150, {"range": [0, 100]}, (0, 1)) == 100


@pytest.fixture
def photo_folder(tmp_path):
    root = tmp_path / "photos"
    models = {"musiq": {"lower_better": False, "range": [0, 100]}, "niqe": {"lower_better": True, "range": [0, 20]}}
    images = {}
    for i, rel in enumerate(["a/1.png", "a/2.png", "b/3.png", "top.png"]):
        make_photo(root, rel, "A")
        images[rel] = {"rating_mean": 1 + i, "models": {
            "musiq": {"rating": 1 + i, "score": 20.0 * (i + 1)}, "niqe": {"rating": 1 + i, "score": 16.0 - 4 * i}},
            "tags": {"landscape": 0.5}, "people": [{"name": "Alice"}] if i == 0 else []}
    (root / pl.RESULTS_NAME).write_text(json.dumps({"mode": "absolute", "models": models, "images": images}), encoding="utf-8")
    return root


def page_data(html: str) -> dict:
    return json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1).replace("<\\/", "</"))


def test_gallery_page_is_self_contained_and_carries_the_ranking(photo_folder, monkeypatch):
    monkeypatch.setattr("sys.argv", ["imageautorater-gallery", str(photo_folder), "--top", "2"])
    gal.main()
    html = (photo_folder / "gallery.html").read_text(encoding="utf-8")
    data = page_data(html)
    assert data["top"] == 2 and sorted(data["models"]) == ["musiq", "niqe"]
    assert data["lower"] == {"musiq": False, "niqe": True}           # the page sorts lower-is-better models correctly
    assert {i["f"] for i in data["items"]} == {"a", "b", ""}          # grouped by sub folder; "" = the top folder
    best = max(data["items"], key=lambda i: i["o"])
    assert best["p"] == "top.png" and best["n"] == [] and "landscape" in best["g"]
    assert (photo_folder / ".iqa_thumbs").is_dir() and len(list((photo_folder / ".iqa_thumbs").glob("*.jpg"))) == 4
    assert "<script src=" not in html and "http://" not in html.replace("http://purl.org", "")   # no external resources


def test_gallery_reuses_thumbnails_and_skips_missing_photos(photo_folder, monkeypatch, capsys):
    argv = ["imageautorater-gallery", str(photo_folder)]
    monkeypatch.setattr("sys.argv", argv)
    gal.main()
    thumbs = {p.name: p.stat().st_mtime_ns for p in (photo_folder / ".iqa_thumbs").glob("*.jpg")}
    (photo_folder / "b" / "3.png").unlink()
    gal.main()
    assert {p.name: p.stat().st_mtime_ns for p in (photo_folder / ".iqa_thumbs").glob("*.jpg")} == thumbs   # nothing re-made
    assert len(page_data((photo_folder / "gallery.html").read_text(encoding="utf-8"))["items"]) == 3
