"""Where the single ground-truth _faces folder is looked for, and what the photo scan must never pick up."""
import contextlib
import io
import json

import pytest
from conftest import make_jpeg, make_photo, people_of

import imageautorater.pipeline as pl


@pytest.fixture
def where(tmp_path, monkeypatch):
    """An isolated world: no env override, not a source checkout, per-user data folder inside tmp_path.
    Returns the photo folder  tmp/Photos/2025."""
    monkeypatch.delenv("IMAGEAUTORATER_FACES", raising=False)
    monkeypatch.setattr(pl, "__file__", str(tmp_path / "site-packages" / "imageautorater" / "pipeline.py"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    photos = tmp_path / "Photos" / "2025"
    photos.mkdir(parents=True)
    return photos


def user_dir():
    return pl.user_data_faces_dir()


def test_a_custom_folder_wins_over_everything(where, monkeypatch, tmp_path):
    (where / "_faces").mkdir()
    monkeypatch.setenv("IMAGEAUTORATER_FACES", str(tmp_path / "from-env"))
    assert pl.resolve_faces_dir(where, tmp_path / "custom")[:2] == (tmp_path / "custom", "--faces-dir")
    assert pl.resolve_faces_dir(where)[:2] == (tmp_path / "from-env", "$IMAGEAUTORATER_FACES")


def test_inside_the_root_of_the_scanned_folder_is_the_primary_choice(where):
    (where / "_faces").mkdir()                          # in the root of the folder you scan, next to its subfolders
    (where.parent / "_faces").mkdir()                   # another one above it
    user_dir().mkdir(parents=True)                      # and a per-user one
    folder, how, others = pl.resolve_faces_dir(where)
    assert folder == where.resolve() / "_faces" and how == "inside the photo folder"
    assert [h for _, h in others] == ["next to the photo folder", "per-user data folder"]   # reported, not silently ignored


def test_next_to_the_photo_folder_and_further_up_are_found_when_nothing_is_inside(where):
    sub = where / "march" / "trip"
    sub.mkdir(parents=True)
    (where.parent / "_faces").mkdir()                   # Photos/_faces serves Photos/2025 and everything below it
    assert pl.resolve_faces_dir(where)[:2] == (where.resolve().parent / "_faces", "next to the photo folder")
    assert pl.resolve_faces_dir(sub)[:2] == (where.resolve().parent / "_faces", "in a parent folder")
    nearer = where / "march" / "_faces"
    nearer.mkdir()
    assert pl.resolve_faces_dir(sub)[:2] == (nearer.resolve(), "next to the photo folder")   # the nearest one wins


def test_existing_fallbacks_are_found_when_nothing_is_around_the_photos(where):
    user_dir().mkdir(parents=True)
    assert pl.resolve_faces_dir(where)[:2] == (user_dir(), "per-user data folder")


def test_nothing_exists_yet_creates_it_inside_the_photo_folder(where):
    folder, how, others = pl.resolve_faces_dir(where)
    assert folder == where.resolve() / "_faces" and how == "inside the photo folder (new)" and others == []


def test_a_scanned_folder_called__faces_never_selects_itself(where, tmp_path):
    faces_named = tmp_path / "Photos" / "_faces"
    faces_named.mkdir()
    folder, how, _ = pl.resolve_faces_dir(faces_named)
    assert folder != faces_named and faces_named not in folder.parents


def test_photo_scan_skips_hidden_folders_and_every__faces_folder(tmp_path):
    p = tmp_path / "P"
    make_jpeg(p, "real.jpg")
    make_jpeg(p, ".iqa_thumbs/thumb.jpg")               # the gallery's thumbnails
    make_jpeg(p, "_faces/person_0001/crop.jpg")         # the faces folder in the root of the photo folder
    make_jpeg(p, "2025/_faces/Alice/ref.jpg")           # a _faces folder deeper down, even if another one is in use
    make_jpeg(p, "2025/sub/also_real.jpg")
    make_jpeg(p, "custom/person_0001/crop.jpg")         # a custom faces folder inside the photo folder
    found = [x.relative_to(p).as_posix() for x in pl.find_images(p, True, False, exclude=[p / "custom"])]
    assert found == ["2025/sub/also_real.jpg", "real.jpg"]


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["imageautorater-rate", *argv])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        pl.main()
    return out.getvalue()


def test_cli_uses_and_announces_the__faces_folder_in_the_root_of_the_scanned_folder(stubbed, where, monkeypatch):
    for n in ("1.png", "2.png"):
        make_photo(where, n, "A")
    out = run_cli(monkeypatch, str(where), "--faces-only")
    faces = where / "_faces"
    assert (faces / "person_0001").is_dir()
    assert f"Faces folder: {faces.resolve()} (inside the photo folder (new))" in out
    (faces / "person_0001").rename(faces / "Alice")
    make_photo(where, "3.png", "A")
    run_cli(monkeypatch, str(where), "--faces-only")
    res = json.loads((where / pl.RESULTS_NAME).read_text(encoding="utf-8"))
    assert sorted(res["images"]) == ["1.png", "2.png", "3.png"]                 # the crops in _faces were not rated as photos
    assert people_of(res) == {"1.png": ["Alice"], "2.png": ["Alice"], "3.png": ["Alice"]}


def test_cli_one__faces_folder_above_serves_several_photo_folders(stubbed, where, monkeypatch):
    faces = where.parent / "_faces"
    faces.mkdir()
    for n in ("1.png", "2.png"):
        make_photo(where, n, "A")
    run_cli(monkeypatch, str(where), "--faces-only")
    (faces / "person_0001").rename(faces / "Alice")
    other = where.parent / "2024"
    make_photo(other, "x.png", "A"); make_photo(other, "y.png", "C")
    out = run_cli(monkeypatch, str(other), "--faces-only")
    res = json.loads((other / pl.RESULTS_NAME).read_text(encoding="utf-8"))
    assert people_of(res) == {"x.png": ["Alice"], "y.png": [None]}               # a name given once is known everywhere
    assert "(next to the photo folder)" in out


def test_cli_first_run_prompt_names_the_folder_it_looked_in(stubbed, where, monkeypatch):
    make_photo(where, "1.png", "A")
    monkeypatch.setattr("sys.stdin", io.StringIO(""))                           # not a terminal: takes the default answer
    out = run_cli(monkeypatch, str(where))
    assert f"nothing confirmed in {where.resolve() / '_faces'}" in out          # says WHERE it looked, not just "the faces folder"
    assert "using --faces-only" in out


def test_cli_custom_faces_folder_inside_the_photo_folder_is_not_scanned(stubbed, where, monkeypatch):
    for n in ("1.png", "2.png"):
        make_photo(where, n, "A")
    make_jpeg(where, ".iqa_thumbs/t.jpg")
    run_cli(monkeypatch, str(where), "--faces-only", "--faces-dir", str(where / "custom"))
    assert (where / "custom" / "person_0001").is_dir()                          # crops went to the custom folder
    run_cli(monkeypatch, str(where), "--faces-only", "--faces-dir", str(where / "custom"))   # second run
    res = json.loads((where / pl.RESULTS_NAME).read_text(encoding="utf-8"))
    assert sorted(res["images"]) == ["1.png", "2.png"]                          # no crops, no thumbnails as "photos"


def test_cli_warns_when_a_second__faces_folder_exists(stubbed, where, monkeypatch, capsys):
    (where / "_faces").mkdir()
    (where.parent / "_faces").mkdir()
    make_photo(where, "1.png", "A")
    run_cli(monkeypatch, str(where), "--faces-only")
    assert "another _faces folder exists and is NOT used" in capsys.readouterr().err
