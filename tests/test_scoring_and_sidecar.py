import xml.etree.ElementTree as ET

import imageautorater.pipeline as pl

RDF_LI = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li"


def test_stars_absolute_uses_score_range_and_direction():
    info = {"lower_better": False, "range": [0, 100]}
    assert pl.to_stars({"a": 5, "b": 55, "c": 99}, info, "absolute") == {"a": 1, "b": 3, "c": 5}
    niqe = {"lower_better": True, "range": [0, 20]}
    assert pl.to_stars({"sharp": 1, "blurry": 19}, niqe, "absolute") == {"sharp": 5, "blurry": 1}


def test_stars_percentile_spreads_over_all_five():
    stars = pl.to_stars({str(i): i for i in range(50)}, {"lower_better": False, "range": None}, "percentile")
    assert set(stars.values()) == {1, 2, 3, 4, 5}


def test_sidecar_keeps_user_data_and_replaces_only_own_keywords(tmp_path):
    xmp = tmp_path / "a.xmp"
    xmp.write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" crs:Exposure2012="+0.5">'
        "<dc:subject><rdf:Bag><rdf:li>holiday</rdf:li></rdf:Bag></dc:subject></rdf:Description></rdf:RDF></x:xmpmeta>",
        encoding="utf-8")
    pl.write_sidecar(xmp, {"musiq": (4, 61.2), "niqe": (2, 9.0)}, True, ["Alice", "beach"])
    text = xmp.read_text(encoding="utf-8")
    assert 'crs:Exposure2012="+0.5"' in text           # foreign namespaces survive
    assert 'xmp:Rating="3"' in text and 'iqa:Rating_musiq="4"' in text
    pl.write_sidecar(xmp, {"musiq": (4, 61.2)}, True, ["Bob"])  # second run: our old keywords are replaced
    items = [li.text for li in ET.parse(xmp).getroot().iter(RDF_LI)]
    assert items == ["holiday", "Bob"]                    # the user's own keyword is untouched


def test_keyword_you_already_had_is_not_removed_later(tmp_path):
    xmp = tmp_path / "a.xmp"
    xmp.write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'
        "<dc:subject><rdf:Bag><rdf:li>Alice</rdf:li></rdf:Bag></dc:subject></rdf:Description></rdf:RDF></x:xmpmeta>",
        encoding="utf-8")
    pl.write_sidecar(xmp, {}, False, ["Alice", "beach"])   # Alice was already there: must not be claimed as ours
    pl.write_sidecar(xmp, {}, False, [])                    # the next run drops only what the tool itself added
    assert [li.text for li in ET.parse(xmp).getroot().iter(RDF_LI)] == ["Alice"]
