import json
import os
import re
from pathlib import Path
from urllib.parse import unquote

import pytest
from conftest import make_photo

import imageautorater.gallery as gal
import imageautorater.pipeline as pl


def test_preset_follows_the_hardware_and_models_override_it(stubbed, tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_run_models(folder, paths, models, device, max_side, cache, cache_path):
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


def page_index(html: str) -> dict:
    return json.loads(re.search(r'<script id="index" type="application/json">(.*?)</script>', html, re.S).group(1).replace("<\\/", "</"))


def group_items(root, index: dict, key: str) -> list:
    """The items of one top-level group, read from its data file like the page does."""
    g = next(g for g in index["groups"] if g["k"] == key)
    text = (root / unquote(g["file"])).read_text(encoding="utf-8")
    return json.loads(re.search(r"\]=(.*);\s*$", text, re.S).group(1))


def all_items(root, index: dict) -> list:
    return [it for g in index["groups"] for it in group_items(root, index, g["k"])]


def test_gallery_page_is_self_contained_and_carries_the_ranking(photo_folder, monkeypatch):
    monkeypatch.setattr("sys.argv", ["imageautorater-gallery", str(photo_folder), "--top", "2", "--jobs", "1"])
    gal.main()
    html = (photo_folder / "gallery.html").read_text(encoding="utf-8")
    index = page_index(html)
    items = all_items(photo_folder, index)
    assert index["top"] == 2 and index["models"] == ["musiq", "niqe"]
    assert index["lower"] == [False, True]                                # the page ranks lower-is-better models correctly
    assert index["folders"] == {"a": 2, "b": 1, "": 1}                    # photos per folder; "" = the top folder
    assert [g["k"] for g in index["groups"]] == ["", "a", "b"]            # one data file per top-level folder
    assert {i["f"] for i in items} == {"a", "b", ""}
    best = max(items, key=lambda i: i["o"])
    assert best["p"] == "top.png" and best["n"] == [] and "landscape" in best["g"]
    assert best["u"] == "top.png" and best["b"] == 1 and best["t"].startswith(".iqa_thumbs/")
    assert len(best["m"]) == 2 and best["m"][0][1] == 80.0               # per-model values aligned with index["models"]
    assert index["people"] == ["Alice"] and index["tags"] == ["landscape"]
    assert len(list((photo_folder / ".iqa_thumbs").glob("*.jpg"))) == 4
    assert "http://" not in html.replace("http://purl.org", "") and "https://" not in html   # no external resources
    assert "<script src=" not in html                                       # data files are added by the page on demand


def test_gallery_embed_puts_everything_into_one_file(photo_folder, monkeypatch):
    monkeypatch.setattr("sys.argv", ["imageautorater-gallery", str(photo_folder), "--embed", "--jobs", "1"])
    gal.main()
    html = (photo_folder / "gallery.html").read_text(encoding="utf-8")
    assert html.count("window.__IQA_DATA=") == 3 and not (photo_folder / "gallery_data").exists()


def test_gallery_reuses_thumbnails_and_skips_missing_photos(photo_folder, monkeypatch, capsys):
    argv = ["imageautorater-gallery", str(photo_folder), "--jobs", "1"]
    monkeypatch.setattr("sys.argv", argv)
    gal.main()
    thumbs = {p.name: p.stat().st_mtime_ns for p in (photo_folder / ".iqa_thumbs").glob("*.jpg")}
    (photo_folder / "b" / "3.png").unlink()
    gal.main()
    assert {p.name: p.stat().st_mtime_ns for p in (photo_folder / ".iqa_thumbs").glob("*.jpg")} == thumbs   # nothing re-made
    index = page_index((photo_folder / "gallery.html").read_text(encoding="utf-8"))
    assert len(all_items(photo_folder, index)) == 3
    assert [g["k"] for g in index["groups"]] == ["", "a"]
    assert len(list((photo_folder / "gallery_data").glob("g_*.js"))) == 2          # the file of the emptied group "b" is removed


def test_thumbnails_are_capped_in_size_in_one_folder_and_made_in_parallel(tmp_path, monkeypatch):
    from PIL import Image

    root = tmp_path / "photos"
    images = {}
    for i in range(6):
        (root / f"y{i % 2}").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1200, 800), (i * 30, 5, 5)).save(root / f"y{i % 2}" / f"p{i}.jpg")
        images[f"y{i % 2}/p{i}.jpg"] = {"rating_mean": 3, "models": {"musiq": {"rating": 3, "score": 50.0}}}
    (root / "y0" / "broken.jpg").write_bytes(b"not a picture")
    images["y0/broken.jpg"] = {"rating_mean": 1, "models": {"musiq": {"rating": 1, "score": 5.0}}}
    (root / pl.RESULTS_NAME).write_text(json.dumps({"models": {"musiq": {"range": [0, 100]}}, "images": images}), encoding="utf-8")
    thumbs = tmp_path / "thumbs"
    monkeypatch.setattr("sys.argv", ["x", str(root), "--jobs", "2", "--thumb-dir", str(thumbs), "--thumb-size", "300"])
    gal.main()
    made = list(thumbs.glob("*.jpg"))
    assert len(made) == 6 and not list(thumbs.glob("*.tmp"))              # the broken file gets none; nothing half-written
    assert all(max(Image.open(p).size) == 300 for p in made)
    items = all_items(root, page_index((root / "gallery.html").read_text(encoding="utf-8")))
    broken = next(i for i in items if i["p"].endswith("broken.jpg"))
    assert broken["t"] == "y0/broken.jpg"                                  # no thumbnail: falls back to the original


def test_thumbs_only_writes_no_page_and_odd_file_names_are_escaped(tmp_path, monkeypatch):
    root = tmp_path / "photos"
    make_photo(root, "2024/a b#1.png", "A")
    (root / pl.RESULTS_NAME).write_text(json.dumps({"models": {}, "images": {"2024/a b#1.png": {"rating_mean": 2, "models": {"musiq": {"rating": 2, "score": 40.0}}}}}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["x", str(root), "--thumbs-only", "--jobs", "1"])
    gal.main()
    assert not (root / "gallery.html").exists() and len(list((root / ".iqa_thumbs").glob("*.jpg"))) == 1
    monkeypatch.setattr("sys.argv", ["x", str(root), "--jobs", "1"])
    gal.main()
    item = all_items(root, page_index((root / "gallery.html").read_text(encoding="utf-8")))[0]
    assert item["u"] == "2024/a%20b%231.png" and item["p"] == "2024/a b#1.png"


def test_page_logic_in_node(tmp_path):
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    script = Path(__file__).with_name("gallery_logic_test.js")
    r = subprocess.run([node, str(script), str(Path(gal.__file__).with_name("gallery_page") / "logic.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_page_in_a_browser(tmp_path, monkeypatch):
    """Loads the real page from file:// in Chromium (skipped when playwright or a browser is missing)."""
    sync_api = pytest.importorskip("playwright.sync_api")
    root = tmp_path / "photos"
    images = {}
    for y in ("2023", "2024"):
        for i in range(4):
            rel = f"{y}/sub/p{i}.png" if i % 2 else f"{y}/p{i}.png"
            make_photo(root, rel, "A")
            images[rel] = {"rating_mean": 1 + i, "models": {"musiq": {"rating": 1 + i, "score": 10.0 * (i + 1)},
                                                            "topiq": {"rating": 1 + i, "score": 40.0 - 10.0 * i}}}
    (root / pl.RESULTS_NAME).write_text(json.dumps({"models": {"musiq": {"range": [0, 100]}, "topiq": {"range": [0, 100]}}, "images": images}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["x", str(root), "--jobs", "1", "--top", "3"])
    gal.main()
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None, args=["--no-sandbox"])
        except Exception as e:
            pytest.skip(f"no browser: {str(e)[:60]}")
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto((root / "gallery.html").as_uri())
        page.wait_for_selector(".card")
        assert page.locator(".group").count() == 2 and page.locator(".group").first.locator(".card").count() == 3
        page.locator(".node", has_text="2023").locator("input").first.click()     # untick one year
        page.wait_for_function("document.querySelectorAll('.group').length === 1")
        page.locator("#views button[data-v=compare]").click()
        page.wait_for_selector(".col")
        assert page.locator(".col").count() == 2 and "in all 2 lists: 3" in page.locator(".group h2").first.inner_text()
        assert not errors
        browser.close()
