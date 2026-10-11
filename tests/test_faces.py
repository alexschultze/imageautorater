import shutil

from conftest import make_photo, people_of

import imageautorater.pipeline as pl


def groups(faces_dir):
    return sorted(d.name for d in faces_dir.iterdir() if d.is_dir()) if faces_dir.exists() else []


def test_unknown_faces_get_person_folders_and_a_single_face_waits_for_a_second(cli, tmp_path):
    p = tmp_path / "P1"
    make_photo(p, "a1.png", "A"); make_photo(p, "a2.png", "A"); make_photo(p, "b1.png", "B")  # B is seen only once
    cli(p, "--faces-only")
    assert groups(cli.faces_dir) == ["person_0001"]                     # --min-cluster 2: B has no folder yet
    make_photo(p, "b2.png", "B")
    cli(p, "--faces-only")
    assert groups(cli.faces_dir) == ["person_0001", "person_0002"]      # the second B face creates its folder


def test_renamed_folder_names_the_person_in_every_photo_folder(cli, tmp_path):
    p1, p2 = tmp_path / "P1", tmp_path / "P2"
    for n in ("1.png", "2.png"):
        make_photo(p1, n, "A")
    cli(p1, "--faces-only")
    (cli.faces_dir / "person_0001").rename(cli.faces_dir / "Alice")
    _, res = cli(p1, "--faces-only")
    assert people_of(res) == {"1.png": ["Alice"], "2.png": ["Alice"]}
    make_photo(p2, "1.png", "C")   # same file name, a stranger
    make_photo(p2, "2.png", "A")   # Alice again, in a different photo folder
    _, res2 = cli(p2, "--faces-only")
    assert people_of(res2) == {"1.png": [None], "2.png": ["Alice"]}


def test_folder_numbers_are_never_reused_after_renaming(cli, tmp_path):
    p = tmp_path / "P"
    for n, who in (("a1.png", "A"), ("a2.png", "A"), ("b1.png", "B"), ("b2.png", "B")):
        make_photo(p, n, who)
    cli(p, "--faces-only")
    (cli.faces_dir / "person_0002").rename(cli.faces_dir / "Bob")
    for n in ("c1.png", "c2.png"):
        make_photo(p, n, "C")
    cli(p, "--faces-only")
    assert "person_0002" not in groups(cli.faces_dir)
    assert {"person_0001", "person_0003", "Bob"} <= set(groups(cli.faces_dir))


def test_a_replaced_photo_does_not_inherit_the_old_name(cli, tmp_path):
    p = tmp_path / "P"
    make_photo(p, "x.png", "A"); make_photo(p, "y.png", "A")
    cli(p, "--faces-only")
    (cli.faces_dir / "person_0001").rename(cli.faces_dir / "Alice")
    cli(p, "--faces-only")
    make_photo(p, "x.png", "C")        # same name, now a different person
    _, res = cli(p, "--faces-only")
    assert people_of(res) == {"x.png": [None], "y.png": ["Alice"]}


def test_deleting_the_cache_keeps_the_confirmed_names(cli, tmp_path):
    p = tmp_path / "P"
    make_photo(p, "x.png", "A"); make_photo(p, "y.png", "A")
    cli(p, "--faces-only")
    (cli.faces_dir / "person_0001").rename(cli.faces_dir / "Alice")
    cli(p, "--faces-only")
    (p / pl.CACHE_NAME).unlink()
    _, res = cli(p, "--faces-only")
    assert people_of(res) == {"x.png": ["Alice"], "y.png": ["Alice"]}


def test_junk_folder_means_not_a_person(cli, tmp_path):
    p = tmp_path / "P"
    make_photo(p, "x.png", "A"); make_photo(p, "y.png", "A")
    cli(p, "--faces-only")
    (cli.faces_dir / "person_0001").rename(cli.faces_dir / "_junk")
    _, res = cli(p, "--faces-only")
    assert people_of(res) == {"x.png": [None], "y.png": [None]}
    assert not any(d.name.startswith("person_") for d in cli.faces_dir.iterdir() if d.is_dir())


def test_reference_photo_dropped_into_a_named_folder_is_learned(cli, tmp_path):
    p = tmp_path / "P"
    make_photo(p, "x.png", "B"); make_photo(p, "y.png", "A")
    ref = make_photo(tmp_path, "ref.png", "B")
    (cli.faces_dir / "Bob").mkdir(parents=True)
    shutil.move(ref, cli.faces_dir / "Bob" / "bob.png")
    _, res = cli(p, "--faces-only")
    assert people_of(res) == {"x.png": ["Bob"], "y.png": [None]}


def test_moving_a_crop_between_folders_relabels_that_face(cli, tmp_path):
    p = tmp_path / "P"
    for n in ("a1.png", "a2.png", "b1.png", "b2.png"):
        make_photo(p, n, "A" if n.startswith("a") else "B")
    cli(p, "--faces-only")
    folders = {d.name: [f.name for f in d.iterdir()] for d in cli.faces_dir.iterdir() if d.is_dir()}
    a_dir = next(n for n, files in folders.items() if any("__a1" in f for f in files))
    b_dir = next(n for n in folders if n != a_dir)
    (cli.faces_dir / a_dir).rename(cli.faces_dir / "Alice")
    (cli.faces_dir / b_dir).rename(cli.faces_dir / "Bob")
    cli(p, "--faces-only")
    crop = next((cli.faces_dir / "Bob").glob("*__b1*"))
    shutil.move(crop, cli.faces_dir / "Alice" / crop.name)      # the user drags one crop into the other folder...
    _, res = cli(p, "--faces-only")
    assert people_of(res)["b1.png"] == ["Alice"]                # ...which confirms that very face as Alice
    shutil.move(cli.faces_dir / "Alice" / crop.name, cli.faces_dir / "Bob" / crop.name)
    _, res = cli(p, "--faces-only")
    assert people_of(res)["b1.png"] == ["Bob"]
