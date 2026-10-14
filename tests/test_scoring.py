import imageautorater.pipeline as pl


def test_stars_absolute_uses_score_range_and_direction():
    info = {"lower_better": False, "range": [0, 100]}
    assert pl.to_stars({"a": 5, "b": 55, "c": 99}, info, "absolute") == {"a": 1, "b": 3, "c": 5}
    niqe = {"lower_better": True, "range": [0, 20]}
    assert pl.to_stars({"sharp": 1, "blurry": 19}, niqe, "absolute") == {"sharp": 5, "blurry": 1}


def test_stars_percentile_spreads_over_all_five():
    stars = pl.to_stars({str(i): i for i in range(50)}, {"lower_better": False, "range": None}, "percentile")
    assert set(stars.values()) == {1, 2, 3, 4, 5}


def test_rating_run_writes_no_metadata_next_to_the_photos(cli, tmp_path):
    from conftest import make_photo
    p = tmp_path / "P"
    make_photo(p, "a.png", "A"); make_photo(p, "b.png", "A")
    cli(p, "--faces-only")
    cli(p, "--faces-only")
    assert sorted(f.name for f in p.iterdir()) == ["a.png", "b.png", "iqa_results.json", "iqa_scores.json"]  # no .xmp
