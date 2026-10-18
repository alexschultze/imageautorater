"""A finished run must never be lost because a file on a NAS or synced folder is locked for a moment.
The real failure this reproduces:
  PermissionError: [WinError 5] Access is denied: '...\\iqa_scores.json.tmp' -> '...\\iqa_scores.json'   (os.replace)"""
import json
import os
import shutil

import pytest
from conftest import make_photo

import imageautorater.fileio as fio
import imageautorater.pipeline as pl


def denied(*a, **k):
    raise PermissionError(13, "Access is denied")


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(fio.time, "sleep", lambda s: None)


def make_tmp(tmp_path, content="new"):
    dst = tmp_path / "iqa_scores.json"
    dst.write_text("old", encoding="utf-8")
    tmp = tmp_path / "iqa_scores.json.tmp"
    tmp.write_text(content, encoding="utf-8")
    return tmp, dst


def test_a_briefly_locked_file_is_retried_until_it_works(tmp_path, monkeypatch):
    tmp, dst = make_tmp(tmp_path)
    real, tries = os.replace, []

    def flaky(a, b):
        tries.append(1)
        if len(tries) < 3:
            denied()
        return real(a, b)

    monkeypatch.setattr(os, "replace", flaky)
    assert fio.replace_file(tmp, dst) == "replaced" and len(tries) == 3
    assert dst.read_text() == "new" and not tmp.exists()


def test_a_share_that_forbids_renaming_over_a_file_is_written_in_place(tmp_path, monkeypatch):
    tmp, dst = make_tmp(tmp_path)
    monkeypatch.setattr(os, "replace", denied)
    assert fio.replace_file(tmp, dst, allow_copy=True) == "copied"
    assert dst.read_text() == "new" and not tmp.exists()


def test_photos_are_never_overwritten_in_place_and_the_new_copy_is_kept(tmp_path, monkeypatch):
    tmp, dst = make_tmp(tmp_path)
    monkeypatch.setattr(os, "replace", denied)
    with pytest.raises(OSError, match="cannot replace"):
        fio.replace_file(tmp, dst, allow_copy=False)
    assert dst.read_text() == "old" and tmp.read_text() == "new"       # the original is untouched


def test_when_nothing_works_the_error_says_why(tmp_path, monkeypatch):
    tmp, dst = make_tmp(tmp_path)
    monkeypatch.setattr(os, "replace", denied)
    monkeypatch.setattr(shutil, "copyfile", denied)
    with pytest.raises(OSError, match="overwriting it in place failed too"):
        fio.replace_file(tmp, dst, allow_copy=True)
    assert tmp.read_text() == "new"                                     # the complete new content is still there


def test_save_cache_never_raises_and_recovers_when_the_lock_goes_away(tmp_path, monkeypatch, capsys):
    path = tmp_path / pl.CACHE_NAME
    cache = pl.new_cache()
    cache["scores"]["musiq"] = {"a.jpg": 61.2}
    monkeypatch.setattr(os, "replace", denied)
    monkeypatch.setattr(shutil, "copyfile", denied)
    assert pl.save_cache(path, cache) is False                          # no crash
    assert "could not save iqa_scores.json" in capsys.readouterr().err and pl._cache_error
    monkeypatch.undo()
    monkeypatch.setattr(fio.time, "sleep", lambda s: None)
    assert pl.save_cache(path, cache) is True and pl._cache_error is None
    assert json.loads(path.read_text(encoding="utf-8"))["scores"]["musiq"] == {"a.jpg": 61.2}


def test_the_cache_is_written_compactly(tmp_path):
    path = tmp_path / pl.CACHE_NAME
    cache = pl.new_cache()
    cache["clip"]["a.jpg"] = [0.123] * 512                             # an embedding: 512 numbers per photo
    pl.save_cache(path, cache)
    assert path.read_text(encoding="utf-8").count("\n") == 0           # indent=1 would put each number on its own line


def test_a_cache_that_could_not_be_saved_is_kept_locally_and_listed(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    pl.PROBLEMS.clear()
    path = tmp_path / "share" / pl.CACHE_NAME
    path.parent.mkdir()
    cache = pl.new_cache()
    cache["scores"]["musiq"] = {"a.jpg": 1.0}
    monkeypatch.setattr(os, "replace", denied)
    monkeypatch.setattr(shutil, "copyfile", denied)
    pl.save_cache(path, cache)
    pl.finalize_cache(path, cache)
    (what, fix), = pl.PROBLEMS
    rec = fio.user_data_dir() / "recovery" / pl.CACHE_NAME
    assert "could not be saved" in what and str(rec) in what and "run the same command again" in fix
    assert json.loads(rec.read_text(encoding="utf-8"))["scores"]["musiq"] == {"a.jpg": 1.0}
    pl.PROBLEMS.clear()


def test_a_leftover_tmp_from_an_unfinished_save_is_used_when_it_is_newer(tmp_path, capsys):
    path = tmp_path / pl.CACHE_NAME
    old, new = pl.new_cache(), pl.new_cache()
    old["scores"]["musiq"] = {"a.jpg": 1.0}
    new["scores"]["musiq"] = {"a.jpg": 1.0, "b.jpg": 2.0}
    path.write_text(json.dumps(old), encoding="utf-8")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(new), encoding="utf-8")
    os.utime(path, (1_700_000_000, 1_700_000_000))                      # the cache itself is older
    assert pl.load_cache(path)["scores"]["musiq"] == {"a.jpg": 1.0, "b.jpg": 2.0}
    assert "Recovered the cache from iqa_scores.json.tmp" in capsys.readouterr().err
    tmp.write_text('{"scores": {"musiq": ', encoding="utf-8")            # a half-written tmp is ignored
    os.utime(path, (1_800_000_000, 1_800_000_000))
    assert pl.load_cache(path)["scores"]["musiq"] == {"a.jpg": 1.0}


def test_progress_is_saved_at_most_once_a_minute(tmp_path, monkeypatch):
    saves = []
    monkeypatch.setattr(pl, "save_cache", lambda path, cache: saves.append(1) or True)
    monkeypatch.setattr(pl, "_last_autosave", 0.0)
    pl.autosave(tmp_path / "c.json", {})
    monkeypatch.setattr(pl, "_last_autosave", pl.time.time())
    pl.autosave(tmp_path / "c.json", {})
    pl.autosave(tmp_path / "c.json", {})
    assert len(saves) == 1


def run_main(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["imageautorater-rate", *argv])
    pl.main()


def test_ctrl_c_saves_what_has_been_scored(stubbed, tmp_path, monkeypatch, capsys):
    p = tmp_path / "P"
    make_photo(p, "a.png", "A")

    def interrupted(folder, paths, models, device, max_side, cache, cache_path, **kw):
        cache["scores"]["musiq"] = {"a.png": 55.0}
        cache["models"]["musiq"] = {"lower_better": False, "range": [0, 100]}
        raise KeyboardInterrupt

    monkeypatch.setattr(pl, "run_models", interrupted)
    with pytest.raises(SystemExit) as stop:
        run_main(monkeypatch, str(p), "--no-faces", "--no-classify", "--models", "musiq", "--device", "cpu")
    assert stop.value.code == 130
    assert json.loads((p / pl.CACHE_NAME).read_text(encoding="utf-8"))["scores"]["musiq"] == {"a.png": 55.0}
    assert "Interrupted. The progress was saved." in capsys.readouterr().err


def test_a_crash_saves_the_progress_before_it_propagates(stubbed, tmp_path, monkeypatch):
    p = tmp_path / "P"
    make_photo(p, "a.png", "A")

    def crashing(folder, paths, models, device, max_side, cache, cache_path, **kw):
        cache["scores"]["musiq"] = {"a.png": 12.0}
        raise RuntimeError("boom")

    monkeypatch.setattr(pl, "run_models", crashing)
    with pytest.raises(RuntimeError, match="boom"):
        run_main(monkeypatch, str(p), "--no-faces", "--no-classify", "--models", "musiq", "--device", "cpu")
    assert json.loads((p / pl.CACHE_NAME).read_text(encoding="utf-8"))["scores"]["musiq"] == {"a.png": 12.0}


def test_a_locked_cache_at_the_end_of_a_real_run_is_reported_not_a_crash(stubbed, tmp_path, monkeypatch, capsys):
    """The user's failure: all scoring done, then 'Access is denied' replacing iqa_scores.json on the NAS."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    p = tmp_path / "P"
    make_photo(p, "a.png", "A"); make_photo(p, "b.png", "A")
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: denied() if str(b).endswith(pl.CACHE_NAME) else real(a, b))
    monkeypatch.setattr(shutil, "copyfile", lambda a, b: denied() if str(b).endswith(pl.CACHE_NAME) else shutil.copy(a, b))
    run_main(monkeypatch, str(p), "--faces-only", "--faces-dir", str(tmp_path / "_f"))     # completes instead of crashing
    out = capsys.readouterr()
    assert "ATTENTION" in out.out and "iqa_scores.json could not be saved" in out.out and "recovery" in out.out
    assert (p / pl.RESULTS_NAME).exists()                                                   # the results were still written
    pl.PROBLEMS.clear()
