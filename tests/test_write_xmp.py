import xml.etree.ElementTree as ET

from conftest import make_photo, write_results

from imageautorater import write_xmp

RDF_LI = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li"
SUBJECT = "{http://purl.org/dc/elements/1.1/}subject"


def keywords(xmp_path):
    return [li.text for li in ET.parse(xmp_path).getroot().iter(RDF_LI)]


def test_sidecar_has_rating_models_and_keywords(tmp_path):
    p = tmp_path / "P"
    make_photo(p, "a.png", "A")
    write_results(p, {"a.png": dict(rating=4, models={"musiq": (4, 61.2), "clipiqa+": (3, 0.5)}, tags=["beach"], people=["Alice"])})
    write_xmp.main([str(p)])
    text = (p / "a.xmp").read_text(encoding="utf-8")
    assert 'xmp:Rating="4"' in text and 'iqa:Rating_musiq="4"' in text and "iqa:Score_clipiqaplus" in text
    assert keywords(p / "a.xmp") == ["Alice", "beach"]


def test_photos_are_never_touched_and_rerun_changes_nothing(tmp_path):
    p = tmp_path / "P"
    img = make_photo(p, "a.png", "A")
    write_results(p, {"a.png": dict(rating=3, models={"musiq": (3, 50.0)}, tags=["night"])})
    before = img.read_bytes()
    write_xmp.main([str(p)])
    first = (p / "a.xmp").stat().st_mtime_ns, (p / "a.xmp").read_bytes()
    write_xmp.main([str(p)])
    assert img.read_bytes() == before
    assert ((p / "a.xmp").stat().st_mtime_ns, (p / "a.xmp").read_bytes()) == first   # unchanged sidecar is not rewritten


def test_existing_sidecar_keeps_foreign_data_and_user_keywords(tmp_path):
    p = tmp_path / "P"
    make_photo(p, "a.png", "A")
    (p / "a.xmp").write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" crs:Exposure2012="+0.5">'
        "<dc:subject><rdf:Bag><rdf:li>holiday</rdf:li></rdf:Bag></dc:subject></rdf:Description></rdf:RDF></x:xmpmeta>",
        encoding="utf-8")
    write_results(p, {"a.png": dict(rating=5, models={"musiq": (5, 90.0)}, tags=["beach"], people=["Alice"])})
    write_xmp.main([str(p)])
    assert 'crs:Exposure2012="+0.5"' in (p / "a.xmp").read_text(encoding="utf-8")
    assert keywords(p / "a.xmp") == ["holiday", "Alice", "beach"]
    write_results(p, {"a.png": dict(rating=5, models={"musiq": (5, 90.0)}, tags=["night"], people=[])})   # new results
    write_xmp.main([str(p)])
    assert keywords(p / "a.xmp") == ["holiday", "night"]          # our old keywords replaced, yours kept


def test_options_dry_run_darktable_naming_and_no_fields(tmp_path):
    p = tmp_path / "P"
    make_photo(p, "a.png", "A")
    write_results(p, {"a.png": dict(rating=4, models={"musiq": (4, 61.2)}, tags=["beach"])})
    write_xmp.main([str(p), "--dry-run"])
    assert not list(p.glob("*.xmp"))
    write_xmp.main([str(p), "--darktable-naming", "--no-model-fields", "--no-keywords"])
    text = (p / "a.png.xmp").read_text(encoding="utf-8")
    assert 'xmp:Rating="4"' in text and "iqa:Rating_" not in text and SUBJECT not in text
    write_xmp.main([str(p), "--no-rating"])
    assert 'xmp:Rating' not in (p / "a.xmp").read_text(encoding="utf-8")


def test_unreadable_sidecar_is_set_aside_not_lost(tmp_path):
    p = tmp_path / "P"
    make_photo(p, "a.png", "A")
    (p / "a.xmp").write_text("<broken", encoding="utf-8")
    write_results(p, {"a.png": dict(rating=2, models={"musiq": (2, 30.0)})})
    write_xmp.main([str(p)])
    assert (p / "a.xmp.bak").read_text(encoding="utf-8") == "<broken"
    assert 'xmp:Rating="2"' in (p / "a.xmp").read_text(encoding="utf-8")
