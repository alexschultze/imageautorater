import json
import os
import struct
import xml.etree.ElementTree as ET

import pytest
from conftest import make_jpeg, make_photo, write_results
from PIL import Image

import imageautorater.pipeline as pl
from imageautorater import write_exif as we
from imageautorater.metadata import ImageMeta

RDF_LI = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li"


def read_back(path):
    """What other programs see: EXIF via Pillow, XMP via the raw APP1 segment."""
    data = path.read_bytes()
    segs, _ = we.split_jpeg(data)
    xmp = next((raw[4 + len(we.XMP_ID):] for m, raw in segs if m == 0xE1 and raw[4:].startswith(we.XMP_ID)), None)
    with Image.open(path) as im:
        exif = im.getexif()
        info = {"rating": exif.get(0x4746), "percent": exif.get(0x4749), "make": exif.get(0x010F),
                "model": exif.get(0x0110), "orientation": exif.get(0x0112),
                "xp": bytes(exif[0x9C9E]).decode("utf-16-le").rstrip("\0") if 0x9C9E in exif else None,
                "date": exif.get_ifd(0x8769).get(0x9003), "maker": exif.get_ifd(0x8769).get(0x927C)}
    return info, (ET.fromstring(xmp.rstrip(b"\x00")) if xmp else None)


def pixels(path):
    with Image.open(path) as im:
        return im.convert("RGB").tobytes()


META = dict(rating=4, models={"musiq": (4, 61.2), "niqe": (2, 9.0)}, tags=["beach", "night"], people=["Alice"])


def run(p, *args):
    we.main([str(p), "--yes", *args])


def test_jpeg_gets_exif_rating_keywords_and_xmp_and_keeps_everything_else(tmp_path):
    p = tmp_path / "P"
    img = make_jpeg(p, "a.jpg")
    original_pixels, original = pixels(img), img.read_bytes()
    _, scan_before = we.split_jpeg(original)
    write_results(p, {"a.jpg": META})
    run(p)
    info, xmp = read_back(img)
    assert info["rating"] == 4 and info["percent"] == 75
    assert info["xp"] == "Alice;beach;night"
    xml = ET.tostring(xmp, encoding="unicode")
    assert 'xmp:Rating="4"' in xml and 'iqa:Rating_musiq="4"' in xml and [li.text for li in xmp.iter(RDF_LI)] == ["Alice", "beach", "night"]
    # everything that was there survives, maker note byte for byte
    assert (info["make"], info["model"], info["orientation"], info["date"]) == ("TestMake", "TestModel", 6, "2024:01:02 03:04:05")
    assert info["maker"] == b"\x01\x02\x03MAKERNOTE\x00\xff"
    # ... and the maker note did not move inside the EXIF block (camera formats keep offsets into it)
    tiff = lambda data: next(raw[4 + 6:] for m, raw in we.split_jpeg(data)[0] if raw[4:].startswith(we.EXIF_ID))
    assert tiff(original).find(b"\x01\x02\x03MAKERNOTE") == tiff(img.read_bytes()).find(b"\x01\x02\x03MAKERNOTE") != -1
    # the picture itself is untouched: same decoded pixels and the identical compressed scan data
    assert pixels(img) == original_pixels
    assert we.split_jpeg(img.read_bytes())[1] == scan_before


def test_jpeg_without_exif_and_with_other_segments(tmp_path):
    p = tmp_path / "P"
    img = make_jpeg(p, "plain.jpg", with_exif=False)
    before = pixels(img)
    write_results(p, {"plain.jpg": META})
    run(p)
    info, xmp = read_back(img)
    assert info["rating"] == 4 and info["xp"] == "Alice;beach;night" and xmp is not None
    assert pixels(img) == before


def test_progressive_jpeg_with_icc_profile_and_comment(tmp_path):
    p = tmp_path / "P"
    p.mkdir()
    img = p / "prog.jpg"
    import numpy as np
    arr = np.random.default_rng(3).integers(0, 255, (96, 128, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img, "JPEG", progressive=True, optimize=True, icc_profile=b"\x00" * 3000 + b"ICC-PROFILE-BYTES", comment=b"kept")
    original = img.read_bytes()
    before = pixels(img)
    write_results(p, {"prog.jpg": META})
    run(p)
    new = img.read_bytes()
    assert pixels(img) == before and we.split_jpeg(new)[1] == we.split_jpeg(original)[1]
    kept = lambda data: [raw for m, raw in we.split_jpeg(data)[0] if m in (0xE2, 0xFE, 0xDB, 0xC2, 0xC4)]   # ICC, comment, tables, SOF2, huffman
    assert kept(new) == kept(original)
    assert read_back(img)[0]["rating"] == 4


def test_rerun_is_a_no_op_and_files_do_not_grow(tmp_path):
    p = tmp_path / "P"
    img = make_jpeg(p, "a.jpg")
    write_results(p, {"a.jpg": META})
    run(p)
    first = img.read_bytes()
    stamp = img.stat().st_mtime_ns
    run(p)
    assert img.read_bytes() == first and img.stat().st_mtime_ns == stamp
    write_results(p, {"a.jpg": dict(META, rating=5, tags=["beach"])})    # new results: updates in place
    run(p)
    second = img.read_bytes()
    assert read_back(img)[0]["rating"] == 5 and read_back(img)[0]["xp"] == "Alice;beach"
    run(p)
    assert img.read_bytes() == second


def test_your_own_keywords_survive_and_old_auto_keywords_are_replaced(tmp_path):
    p = tmp_path / "P"
    img = make_jpeg(p, "a.jpg")
    # the user already has Windows tags and an XMP subject
    data = img.read_bytes()
    segs, rest = we.split_jpeg(data)
    exif = next(raw for m, raw in segs if raw[4:].startswith(we.EXIF_ID))
    tiff = we.edit_exif(exif[4 + 6:], None, ["holiday"])
    out = [(we._app1(we.EXIF_ID + tiff) if raw is exif else raw) for _, raw in segs]
    img.write_bytes(b"\xff\xd8" + b"".join(out) + rest)
    write_results(p, {"a.jpg": dict(META, tags=["beach"], people=["Alice"])})
    run(p)
    assert read_back(img)[0]["xp"] == "holiday;Alice;beach"
    write_results(p, {"a.jpg": dict(META, tags=["night"], people=[])})
    run(p)
    info, xmp = read_back(img)
    assert info["xp"] == "holiday;night"
    assert [li.text for li in xmp.iter(RDF_LI)] == ["night"]      # only ours in the XMP; "holiday" lives in the Windows tags


def test_big_endian_and_little_endian_exif_blocks(tmp_path):
    for bo in ("<", ">"):
        order = b"II" if bo == "<" else b"MM"
        make = b"Camera\x00"
        ifd = struct.pack(bo + "H", 1) + struct.pack(bo + "HHI", 0x010F, 2, len(make)) + struct.pack(bo + "I", 8 + 2 + 12 + 4) \
            + struct.pack(bo + "I", 0)
        tiff = order + struct.pack(bo + "HI", 42, 8) + ifd + make
        new = we.edit_exif(tiff, 3, ["x", "y"])
        assert we.read_exif_tags(new) == (3, ["x", "y"])
        assert new.endswith(make) or make in new                # the original value bytes are still there, at the same place
        assert new[:8 + len(ifd) + len(make)][8 + len(ifd):] == make
        assert we.edit_exif(new, 3, ["x", "y"]) is None         # nothing to change: no new bytes
        shrunk = we.edit_exif(new, 3, [])                        # keywords removed: tag dropped, in place
        assert we.read_exif_tags(shrunk) == (3, []) and len(shrunk) == len(new)


def test_png_gets_xmp_only_and_pixels_are_identical(tmp_path):
    p = tmp_path / "P"
    img = make_photo(p, "a.png", "A")
    before = pixels(img)
    write_results(p, {"a.png": META})
    run(p)
    assert pixels(img) == before
    chunks = we.split_png(img.read_bytes())
    xmp = next(pl_ for k, pl_, _ in chunks if k == b"iTXt" and pl_.startswith(we.PNG_XMP_KEY))
    assert b'xmp:Rating="4"' in xmp and b"beach" in xmp
    stamp, data = img.stat().st_mtime_ns, img.read_bytes()
    run(p)
    assert img.read_bytes() == data and img.stat().st_mtime_ns == stamp


def test_other_formats_are_never_edited(tmp_path):
    p = tmp_path / "P"
    p.mkdir()
    for name, fmt in (("a.tif", "TIFF"), ("b.webp", "WEBP"), ("c.bmp", "BMP")):
        Image.new("RGB", (20, 20), "red").save(p / name, fmt)
    (p / "d.cr2").write_bytes(b"raw bytes")
    snapshot = {f.name: f.read_bytes() for f in p.iterdir()}
    write_results(p, {n: dict(rating=3, models={"musiq": (3, 50.0)}) for n in snapshot})
    we.main([str(p), "--yes"])
    assert {f.name: f.read_bytes() for f in p.iterdir() if f.name != pl.RESULTS_NAME} == snapshot


def test_dry_run_and_confirmation_and_backup(tmp_path, monkeypatch, capsys):
    p = tmp_path / "P"
    img = make_jpeg(p, "sub/a.jpg")
    original = img.read_bytes()
    write_results(p, {"sub/a.jpg": META})
    we.main([str(p), "--dry-run"])
    assert img.read_bytes() == original
    assert "1 files would change" in capsys.readouterr().out
    with pytest.raises(SystemExit):                      # not interactive and no --yes: refuses to touch files
        we.main([str(p)])
    assert img.read_bytes() == original
    backup = tmp_path / "backup"
    we.main([str(p), "--yes", "--backup", str(backup)])
    assert (backup / "sub" / "a.jpg").read_bytes() == original and img.read_bytes() != original


def test_modified_time_kept_and_pipeline_cache_stays_valid(tmp_path):
    p = tmp_path / "P"
    img = make_jpeg(p, "a.jpg")
    os.utime(img, (1_700_000_000, 1_700_000_000))
    write_results(p, {"a.jpg": META})
    cache = pl.new_cache()
    cache["files"]["a.jpg"] = pl.fingerprint(img)
    cache["scores"]["musiq"] = {"a.jpg": 61.2}
    pl.save_cache(p / pl.CACHE_NAME, cache)
    run(p)
    assert int(img.stat().st_mtime) == 1_700_000_000                       # modified time preserved
    after = json.loads((p / pl.CACHE_NAME).read_text(encoding="utf-8"))
    assert after["files"]["a.jpg"] == pl.fingerprint(img)                  # so the next rating run does not re-score it
    assert after["scores"]["musiq"] == {"a.jpg": 61.2}


def test_damaged_files_are_skipped_not_destroyed(tmp_path):
    p = tmp_path / "P"
    p.mkdir()
    (p / "bad.jpg").write_bytes(b"\xff\xd8\xff\xe1\x00\x10garbage")
    (p / "notajpeg.jpg").write_bytes(b"hello")
    snapshot = {f.name: f.read_bytes() for f in p.iterdir()}
    write_results(p, {n: dict(rating=3, models={"musiq": (3, 50.0)}) for n in snapshot})
    we.main([str(p), "--yes"])
    assert {f.name: f.read_bytes() for f in p.iterdir() if f.name != pl.RESULTS_NAME} == snapshot


def test_options_leave_out_what_you_do_not_want(tmp_path):
    p = tmp_path / "P"
    img = make_jpeg(p, "a.jpg")
    write_results(p, {"a.jpg": META})
    run(p, "--no-rating", "--no-model-fields")
    info, xmp = read_back(img)
    text = ET.tostring(xmp, encoding="unicode")
    assert info["rating"] is None and "xmp:Rating" not in text and "iqa:Rating_" not in text
    assert info["xp"] == "Alice;beach;night"
