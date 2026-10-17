"""A long first run must say what it is doing: reading the folder, loading libraries, downloading models."""
import contextlib
import io
import re
import sys

from conftest import make_photo

import imageautorater.pipeline as pl


def run(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["imageautorater-rate", *argv])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        pl.main()
    return out.getvalue()


def test_a_run_explains_itself_from_the_first_second(stubbed, tmp_path, monkeypatch):
    p = tmp_path / "P"
    for n in ("a.png", "b.png", "c.png"):
        make_photo(p, n, "A")
    out = run(monkeypatch, str(p), "--faces-only", "--faces-dir", str(tmp_path / "_faces"))
    lines = out.splitlines()
    assert re.match(r"\[00:0\d\] imageautorater .*nothing is installed unless you agree", lines[0])   # first line, immediately
    for expected in ("Reading ", "Found 3 photos in ", "Checking 3 photos against the cache",
                     "Cache check done: 3 new to this cache, 0 modified, 0 unchanged", "Faces folder: ", "Analysing faces",
                     "Face models ready", "Combining the ratings", "Wrote combined results", "Done in 0h00m"):
        assert expected in out, expected
    assert all(re.match(r"\[\d\d:\d\d\] ", l) for l in lines if l.startswith("[") and "] " in l[:9])
    second = run(monkeypatch, str(p), "--faces-only", "--faces-dir", str(tmp_path / "_faces"))
    assert "0 new to this cache, 0 modified, 3 unchanged" in second               # the cache is explained too


def test_slow_imports_announce_themselves_and_only_once(capsys):
    sys.modules.pop("colorsys", None)
    pl.ensure_imported("colorsys", "Loading a pretend heavy library")
    first = capsys.readouterr().out
    assert "Loading a pretend heavy library ..." in first and "colorsys loaded in " in first
    pl.ensure_imported("colorsys", "Loading a pretend heavy library")
    assert capsys.readouterr().out == ""                                        # already loaded: silent


def test_score_ranges_published_as_text_are_understood():
    """pyiqa lists ranges as text; reading them as numbers failed and every model silently fell back to ranking."""
    assert pl.parse_range("~0, ~100") == [0.0, 100.0]
    assert pl.parse_range("0, 1") == [0.0, 1.0] and pl.parse_range("1, 5") == [1.0, 5.0]
    assert pl.parse_range("-2, 2") == [-2.0, 2.0]
    assert pl.parse_range((0, 10)) == [0.0, 10.0] and pl.parse_range([0.5, 1.5]) == [0.5, 1.5]
    assert pl.parse_range(None) is None and pl.parse_range("unknown") is None and pl.parse_range("5, 5") is None
    stars = pl.to_stars({"a": 20, "b": 90}, {"lower_better": False, "range": pl.parse_range("~0, ~100")}, "absolute")
    assert stars == {"a": 1, "b": 5}                                         # absolute stars, not a ranking


def test_ranges_missing_from_an_old_cache_are_restored(stubbed, tmp_path, monkeypatch):
    p = tmp_path / "P"
    make_photo(p, "a.png", "A"); make_photo(p, "b.png", "A")
    cache = pl.new_cache()
    cache["scores"]["musiq"] = {"a.png": 20.0, "b.png": 90.0}
    cache["models"]["musiq"] = {"lower_better": False, "range": None}      # what the buggy version stored
    cache["files"] = {k: pl.fingerprint(p / k) for k in ("a.png", "b.png")}
    cache["max_side"] = 1024
    pl.save_cache(p / pl.CACHE_NAME, cache)
    monkeypatch.setattr(pl, "config_range", lambda name: [0.0, 100.0])
    monkeypatch.setattr(pl, "ensure_imported", lambda *a, **k: None)
    out = run(monkeypatch, str(p), "--no-iqa", "--no-classify", "--no-faces")
    res = __import__("json").loads((p / pl.RESULTS_NAME).read_text(encoding="utf-8"))["images"]
    assert "Score ranges restored for musiq" in out
    assert {k: v["models"]["musiq"]["rating"] for k, v in res.items()} == {"a.png": 1, "b.png": 5}


def test_folder_walk_reports_progress_on_a_big_folder(tmp_path):
    for i in range(1_250):
        (tmp_path / f"f{i}.txt").write_bytes(b"")
    (tmp_path / "x.jpg").write_bytes(b"")
    seen = []
    found = pl.find_images(tmp_path, True, False, progress=lambda n, f: seen.append((n, f)))
    assert [n for n, _ in seen] == [500, 1000] and len(found) == 1
