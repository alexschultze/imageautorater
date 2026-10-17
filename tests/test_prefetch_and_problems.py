"""Photos are decoded ahead of the model, and anything that did not run is called out at the end."""
import threading
import time

import pytest
from conftest import make_photo

import imageautorater.pipeline as pl


def test_prefetch_keeps_the_order_and_reports_load_errors_instead_of_raising():
    def loader(n):
        if n == 3:
            raise OSError("cannot read photo 3")
        time.sleep(0.01 * (7 - n))                          # later items finish FIRST: order must still be kept
        return n * n

    got = list(pl.prefetched(range(1, 8), loader, workers=4))
    assert [item for item, _, _ in got] == [1, 2, 3, 4, 5, 6, 7]
    assert [v for _, v, _ in got] == [1, 4, None, 16, 25, 36, 49]
    assert isinstance(got[2][2], OSError) and all(e is None for i, (_, _, e) in enumerate(got) if i != 2)


def test_prefetch_really_loads_in_parallel_and_not_further_ahead_than_asked():
    live, peak, started = 0, 0, []
    lock = threading.Lock()

    def loader(n):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
            started.append(n)
        time.sleep(0.05)
        with lock:
            live -= 1
        return n

    gen = pl.prefetched(range(40), loader, workers=4)
    assert next(gen)[0] == 0
    time.sleep(0.3)
    assert len(started) <= 4 * 2 + 2                        # bounded read-ahead: memory stays small
    list(gen)
    assert 2 <= peak <= 4                                   # several photos in flight at once, never more than the workers


def test_prefetch_with_one_worker_behaves_like_the_old_sequential_loop():
    assert [(i, v) for i, v, _ in pl.prefetched(range(5), lambda n: n + 1, workers=1)] == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    assert list(pl.prefetched([], lambda n: n, workers=4)) == []


def run(monkeypatch, *argv):
    import contextlib
    import io
    monkeypatch.setattr("sys.argv", ["imageautorater-rate", *argv])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            pl.main()
        except SystemExit as stop:      # "Nothing was scored." is a normal way for a run to end: keep what was printed
            out.write(f"[exit] {stop.code}\n")
    return out.getvalue()


def test_missing_face_model_is_shouted_at_the_end_not_buried_in_the_download_noise(stubbed, tmp_path, monkeypatch):
    class NoFaces:
        def __init__(self, *a, **k):
            raise ModuleNotFoundError("No module named 'facenet_pytorch'")

    monkeypatch.setattr(pl, "Faces", NoFaces)
    p = tmp_path / "P"
    for n in ("a.png", "b.png"):
        make_photo(p, n, "A")
    out = run(monkeypatch, str(p), "--faces-only", "--faces-dir", str(tmp_path / "_faces"))
    assert "ATTENTION: 1 thing(s) did not run" in out
    assert "faces were NOT detected" in out and "uv pip install --no-deps facenet-pytorch" in out


def test_a_model_that_cannot_load_is_listed_with_the_fix_for_pkg_resources(stubbed, tmp_path, monkeypatch):
    class BrokenPyiqa:
        @staticmethod
        def create_metric(name, device=None):
            raise ModuleNotFoundError("No module named 'pkg_resources'")

    import sys
    monkeypatch.setitem(sys.modules, "pyiqa", BrokenPyiqa)
    monkeypatch.setattr(pl, "pick_device", lambda arg=None: __import__("torch").device("cpu"))
    p = tmp_path / "P"
    make_photo(p, "a.png", "A")
    out = run(monkeypatch, str(p), "--no-faces", "--no-classify", "--models", "clipiqa", "liqe", "--device", "cpu")
    assert "model clipiqa was skipped" in out and "model liqe was skipped" in out
    assert 'uv pip install "setuptools<82"' in out
    assert "Run the same command again after fixing" in out


def test_a_clean_run_has_no_attention_block(stubbed, tmp_path, monkeypatch):
    p = tmp_path / "P"
    make_photo(p, "a.png", "A"); make_photo(p, "b.png", "A")
    out = run(monkeypatch, str(p), "--faces-only", "--faces-dir", str(tmp_path / "_faces"))
    assert "ATTENTION" not in out


def test_setuptools_is_pinned_below_the_release_that_removed_pkg_resources():
    import tomllib
    from pathlib import Path
    deps = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    assert "setuptools<82" in deps
