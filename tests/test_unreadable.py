"""Damaged photos (e.g. 'broken data stream' on a truncated JPEG) are reported once and skipped, not retried on every run
and not counted against the model. Network or permission errors are NOT remembered: a good photo must never be
blacklisted because the connection to the NAS dropped for a moment."""
import contextlib
import io
import json
import sys

import pytest
from conftest import make_jpeg, make_photo
from PIL import UnidentifiedImageError

import imageautorater.pipeline as pl


def test_damaged_files_are_told_apart_from_file_system_and_network_errors():
    assert pl.is_damaged_image(OSError("broken data stream when reading image file"))
    assert pl.is_damaged_image(OSError("image file is truncated (3 bytes not processed)"))
    assert pl.is_damaged_image(UnidentifiedImageError("cannot identify image file"))
    assert pl.is_damaged_image(ValueError("bad image"))
    assert not pl.is_damaged_image(PermissionError(13, "Access is denied"))
    assert not pl.is_damaged_image(ConnectionResetError(10054, "connection reset"))
    assert not pl.is_damaged_image(FileNotFoundError(2, "no such file"))


@pytest.fixture
def fake_pyiqa(monkeypatch):
    torch = pytest.importorskip("torch")

    class Metric:
        score_range = "0, 1"
        lower_better = False

        def __call__(self, x):
            return torch.tensor([0.5])

    class Pyiqa:
        @staticmethod
        def create_metric(name, device=None):
            return Metric()

    monkeypatch.setitem(sys.modules, "pyiqa", Pyiqa)
    monkeypatch.setattr(pl, "pick_device", lambda arg=None: torch.device("cpu"))


def truncated_jpeg(root, rel):
    good = make_jpeg(root, rel)
    good.write_bytes(good.read_bytes()[:len(good.read_bytes()) // 2])      # cut in half: 'broken data stream'
    return good


def test_a_damaged_photo_is_skipped_remembered_and_does_not_hurt_the_model(fake_pyiqa, tmp_path, monkeypatch, capsys):
    p = tmp_path / "P"
    make_jpeg(p, "a.jpg", seed=1); make_jpeg(p, "b.jpg", seed=2); truncated_jpeg(p, "bad.jpg")
    paths = sorted(p.glob("*.jpg"))
    loads = []
    real = pl.load_image
    monkeypatch.setattr(pl, "load_image", lambda path, ms: loads.append(path.name) or real(path, ms))
    cache = pl.new_cache()
    pl.run_models(p, paths, ["musiq"], "cpu", 1024, cache, p / pl.CACHE_NAME, io_threads=2)
    assert sorted(cache["scores"]["musiq"]) == ["a.jpg", "b.jpg"]
    assert list(cache["unreadable"]) == ["bad.jpg"] and "OSError" in cache["unreadable"]["bad.jpg"]
    assert "giving up" not in capsys.readouterr().err                          # not the model's fault
    loads.clear()
    pl.run_models(p, paths, ["musiq"], "cpu", 1024, cache, p / pl.CACHE_NAME, io_threads=2)
    assert loads == []                                                          # nothing is read again: all done or known damaged


def test_a_network_error_is_not_remembered_so_the_photo_is_tried_again(fake_pyiqa, tmp_path, monkeypatch):
    p = tmp_path / "P"
    make_jpeg(p, "a.jpg", seed=1); make_jpeg(p, "b.jpg", seed=2)
    paths = sorted(p.glob("*.jpg"))
    real = pl.load_image

    def flaky(path, ms):
        if path.name == "b.jpg":
            raise ConnectionResetError(10054, "the connection to the NAS was reset")
        return real(path, ms)

    monkeypatch.setattr(pl, "load_image", flaky)
    cache = pl.new_cache()
    pl.run_models(p, paths, ["musiq"], "cpu", 1024, cache, p / pl.CACHE_NAME, io_threads=1)
    assert sorted(cache["scores"]["musiq"]) == ["a.jpg"] and cache["unreadable"] == {}     # b.jpg is NOT blacklisted
    monkeypatch.setattr(pl, "load_image", real)                                             # the connection is back
    pl.run_models(p, paths, ["musiq"], "cpu", 1024, cache, p / pl.CACHE_NAME, io_threads=1)
    assert sorted(cache["scores"]["musiq"]) == ["a.jpg", "b.jpg"]


def run(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["imageautorater-rate", *argv])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        pl.main()
    return out.getvalue()


def test_summary_results_file_and_a_second_chance_when_the_file_changes(stubbed, tmp_path, monkeypatch):
    p = tmp_path / "P"
    make_photo(p, "ok1.png", "A"); make_photo(p, "ok2.png", "A")
    (p / "bad.jpg").write_bytes(b"\xff\xd8\xff\xe0 this is not a real jpeg")
    args = (str(p), "--faces-only", "--faces-dir", str(tmp_path / "_faces"))
    out = run(monkeypatch, *args)
    assert "NOTE: 1 photo(s) could not be read" in out and "bad.jpg" in out
    res = json.loads((p / pl.RESULTS_NAME).read_text(encoding="utf-8"))
    assert list(res["unreadable"]) == ["bad.jpg"] and sorted(res["images"]) == ["ok1.png", "ok2.png"]
    # the same damaged file is not retried (and the note stays), but once the file is replaced it gets another chance
    (p / "bad.jpg").unlink()
    make_photo(p, "bad.jpg", "B", size=140)          # a valid replacement (PNG content under the old name still decodes)
    out2 = run(monkeypatch, *args)
    res2 = json.loads((p / pl.RESULTS_NAME).read_text(encoding="utf-8"))
    assert res2["unreadable"] == {} and "bad.jpg" in res2["images"]
