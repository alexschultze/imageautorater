"""Where the single ground-truth faces folder is looked for, and what the photo scan must never pick up."""
import contextlib
import io
import json

import pytest
from conftest import make_jpeg, make_photo, people_of

import imageautorater.pipeline as pl


@pytest.fixture
def where(tmp_path, monkeypatch):
    """An isolated world: no env override, not a source checkout, per-user data folder inside tmp_path."""
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
    (where.parent / "_faces").mkdir()
    monkeypatch.setenv("IMAGEAUTORATER_FACES", str(tmp_path / "from-env"))
    assert pl.resolve_faces_dir(where, tmp_path / "custom")[:2] == (tmp_path / "custom", "--faces-dir")
    assert pl.resolve_faces_dir(where)[:2] == (tmp_path / "from-env", "$IMAGEAUTORATER_FACES")


def test_next_to_the_photo_folder_is_the_primary_choice(where):
    user_dir().mkdir(parents=True)                      # a per-user folder exists too ...
    (where.parent / "_faces").mkdir()                    # ... but _faces next to the photos wins
    folder, how, others = pl.resolve_faces_dir(where)
    assert folder == where.resolve().parent / "_faces" and how == "next to the photo folder"
    assert others == [(user_dir(), "per-user data folder")]   # reported, not silently ignored


def test_existing_fallbacks_are_found_when_nothing_is_next_to_the_photos(where):
    user_dir().mkdir(parents=True)
    assert pl.resolve_faces_dir(where)[:2] == (user_dir(), "per-user data folder")


def test_nothing_exists_yet_creates_it_next_to_the_photo_folder(where):
    folder, how, others = pl.resolve_faces_dir(where)
    assert folder == where.resolve().parent / "_faces" and how.endswith("(new)") and others == []


def test_the_scanned_folder_itself_is_never_the_faces_folder(where, tmp_path):
    faces_named = tmp_path / "Photos" / "_faces"        # scanning a folder that is itself called "_faces"
    faces_named.mkdir()
    folder, how, _ = pl.resolve_faces_dir(faces_named)
    assert folder != faces_named and faces_named not in folder.parents
    root = tmp_path.anchor and pl.Path(tmp_path.anchor)  # a drive root has no usable sibling either
    assert pl.resolve_faces_dir(root)[0] != root / "_faces"


def test_photo_scan_skips_hidden_folders_and_the_faces_folder(tmp_path):
    p = tmp_path / "P"
    make_jpeg(p, "real.jpg")
    make_jpeg(p, ".iqa_thumbs/thumb.jpg")               # the gallery's thumbnails
    make_jpeg(p, "_faces/person_0001/crop.jpg")          # a custom faces folder inside the photo folder
    make_jpeg(p, "2025/sub/also_real.jpg")
    found = [x.relative_to(p).as_posix() for x in pl.find_images(p, True, False, exclude=[p / "_faces"])]
    assert found == ["2025/sub/also_real.jpg", "real.jpg"]


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["imageautorater-rate", *argv])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        pl.main()
    return out.getvalue()


def test_cli_puts_the_faces_folder_next_to_the_scanned_folder(stubbed, where, monkeypatch):
    for n in ("1.png", "2.png"):
        make_photo(where, n, "A")
    out = run_cli(monkeypatch, str(where), "--faces-only")
    faces = where.parent / "_faces"
    assert (faces / "person_0001").is_dir()
    assert f"Faces folder: {faces.resolve()} (next to the photo folder (new))" in out
    # a second photo folder beside it shares the same faces folder, so a name given once is known everywhere
    (faces / "person_0001").rename(faces / "Alice")
    other = where.parent / "2024"
    make_photo(other, "x.png", "A"); make_photo(other, "y.png", "C")
    run_cli(monkeypatch, str(other), "--faces-only")
    res = json.loads((other / pl.RESULTS_NAME).read_text(encoding="utf-8"))
    assert people_of(res) == {"x.png": ["Alice"], "y.png": [None]}


def test_cli_custom_faces_folder_inside_the_photo_folder_is_not_scanned(stubbed, where, monkeypatch):
    for n in ("1.png", "2.png"):
        make_photo(where, n, "A")
    make_jpeg(where, ".iqa_thumbs/t.jpg")
    run_cli(monkeypatch, str(where), "--faces-only", "--faces-dir", str(where / "_faces"))
    assert (where / "_faces" / "person_0001").is_dir()                          # crops went to the custom folder
    run_cli(monkeypatch, str(where), "--faces-only", "--faces-dir", str(where / "_faces"))   # second run
    res = json.loads((where / pl.RESULTS_NAME).read_text(encoding="utf-8"))
    assert sorted(res["images"]) == ["1.png", "2.png"]                          # no crops, no thumbnails as "photos"


def test_cli_warns_when_a_second_faces_folder_exists(stubbed, where, monkeypatch, capsys):
    (where.parent / "_faces").mkdir()
    user_dir().mkdir(parents=True)
    make_photo(where, "1.png", "A")
    run_cli(monkeypatch, str(where), "--faces-only")
    assert "another _faces folder exists and is NOT used" in capsys.readouterr().err
