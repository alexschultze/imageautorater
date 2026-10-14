"""Write the results of imageautorater-rate INTO the image files (final step; the rating itself writes nothing).

  imageautorater-write-exif "C:\\Photos\\2025" --dry-run    # report what would change
  imageautorater-write-exif "C:\\Photos\\2025"              # asks first, then edits the files
  imageautorater-write-exif "C:\\Photos\\2025" --backup D:\\photo-backup

This MODIFIES YOUR PHOTOS (metadata only):
  JPEG  EXIF Rating, RatingPercent and XPKeywords (the tags Windows shows as Rating and Tags) plus an XMP packet
        (xmp:Rating, dc:subject keywords, the per-model scores) that Lightroom, darktable, digiKam and others read.
  PNG   an XMP packet.
  Anything else (TIFF, WebP, BMP, RAW such as CR2/NEF/ARW/DNG, HEIC) is never touched: use imageautorater-write-xmp,
  which writes sidecar files for any format.

How it stays safe:
  * The compressed picture data is copied byte for byte; nothing is decoded or re-encoded. The scan data of a JPEG,
    the pixel chunks of a PNG, and every existing EXIF/maker-note byte keep their exact content.
  * The EXIF edit only adds or updates its three tags: the IFD is rebuilt at the end of the EXIF block, so no offset
    inside the maker note moves.
  * Keywords you added yourself are kept; only the ones this tool added earlier are replaced.
  * Files are written next to the original and swapped in atomically; the modified time is preserved.
  * Files whose metadata would not change are not rewritten, so re-running is cheap and does not grow the files.
  * --backup DIR copies each original there first. Without --yes the command asks before writing.
"""
import argparse
import os
import shutil
import struct
import sys
import zlib
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from imageautorater.metadata import (ImageMeta, apply_meta, iter_meta, keyword_list, parse_xmp, previous_auto_keywords,
                                     render_xmp)
from imageautorater.pipeline import CACHE_NAME, fingerprint, load_cache, save_cache
from imageautorater.write_xmp import add_common_args, load_results

TAG_RATING, TAG_RATING_PERCENT, TAG_XPKEYWORDS = 0x4746, 0x4749, 0x9C9E
RATING_PERCENT = {1: 1, 2: 25, 3: 50, 4: 75, 5: 99}
XMP_ID = b"http://ns.adobe.com/xap/1.0/\x00"
XMP_EXT_ID = b"http://ns.adobe.com/xmp/extension/\x00"
EXIF_ID = b"Exif\x00\x00"
PNG_SIG = b"\x89PNG\r\n\x1a\n"
PNG_XMP_KEY = b"XML:com.adobe.xmp\x00"
MAX_SEGMENT = 65533  # payload limit of a JPEG segment


class Skip(Exception):
    """This file is left alone (reason in the message)."""


# ---- EXIF (TIFF structure) --------------------------------------------------------------------------------------
def _read_ifd0(tiff: bytes):
    if len(tiff) < 8 or tiff[:2] not in (b"II", b"MM"):
        raise Skip("EXIF block is not a TIFF structure")
    bo = "<" if tiff[:2] == b"II" else ">"
    if struct.unpack(bo + "H", tiff[2:4])[0] != 42:
        raise Skip("EXIF block has a bad TIFF header")
    off = struct.unpack(bo + "I", tiff[4:8])[0]
    if off < 8 or off + 2 > len(tiff):
        raise Skip("EXIF IFD0 offset is out of range")
    n = struct.unpack(bo + "H", tiff[off:off + 2])[0]
    end = off + 2 + 12 * n + 4
    if end > len(tiff):
        raise Skip("EXIF IFD0 is truncated")
    entries = [tiff[off + 2 + 12 * i: off + 14 + 12 * i] for i in range(n)]
    nxt = tiff[off + 2 + 12 * n: end]
    return bo, off, entries, nxt


def _entry_value(tiff: bytes, bo: str, entry: bytes) -> bytes:
    tag, typ, count = struct.unpack(bo + "HHI", entry[:8])
    size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8}.get(typ, 1) * count
    if size <= 4:
        return entry[8:8 + size]
    off = struct.unpack(bo + "I", entry[8:12])[0]
    return tiff[off:off + size]


def read_exif_tags(tiff: "bytes | None"):
    """(rating or None, [keywords]) currently stored in IFD0."""
    if not tiff:
        return None, []
    bo, _, entries, _ = _read_ifd0(tiff)
    rating, kws = None, []
    for e in entries:
        tag = struct.unpack(bo + "H", e[:2])[0]
        if tag == TAG_RATING:
            rating = struct.unpack(bo + "H", _entry_value(tiff, bo, e)[:2])[0]
        elif tag == TAG_XPKEYWORDS:
            text = _entry_value(tiff, bo, e).decode("utf-16-le", "ignore").rstrip("\x00")
            kws = [k.strip() for k in text.split(";") if k.strip()]
    return rating, kws


def edit_exif(tiff: "bytes | None", rating: "int | None", keywords: "list | None") -> "bytes | None":
    """Return a TIFF block with Rating / RatingPercent / XPKeywords set (None for a tag = leave it alone).
    Returns None when nothing would change. All existing bytes are preserved: the rebuilt IFD0 is written in place
    when it is not larger, else appended at the end of the block."""
    if tiff is None:
        tiff = b"II*\x00\x08\x00\x00\x00" + struct.pack("<H", 0) + b"\x00\x00\x00\x00"
    cur_rating, cur_kws = read_exif_tags(tiff)
    want_rating = cur_rating if rating is None else rating
    want_kws = cur_kws if keywords is None else keywords
    if want_rating == cur_rating and want_kws == cur_kws:
        bo, _, entries, _ = _read_ifd0(tiff)
        tags = {struct.unpack(bo + "H", e[:2])[0] for e in entries}
        # RatingPercent is derived from the rating, so equal values mean nothing to do (unless a tag is missing)
        if (want_rating is None or {TAG_RATING, TAG_RATING_PERCENT} <= tags) and (not want_kws or TAG_XPKEYWORDS in tags):
            return None

    bo, ifd_off, entries, nxt = _read_ifd0(tiff)
    ours = {TAG_RATING, TAG_RATING_PERCENT, TAG_XPKEYWORDS}
    kept = [e for e in entries if struct.unpack(bo + "H", e[:2])[0] not in ours]
    buf = bytearray(tiff)

    def pad_even():
        if len(buf) % 2:
            buf.append(0)

    new = []
    if want_rating:
        new.append((TAG_RATING, struct.pack(bo + "HHI", TAG_RATING, 3, 1) + struct.pack(bo + "H", want_rating) + b"\x00\x00"))
        pct = RATING_PERCENT.get(want_rating, 0)
        new.append((TAG_RATING_PERCENT, struct.pack(bo + "HHI", TAG_RATING_PERCENT, 3, 1) + struct.pack(bo + "H", pct) + b"\x00\x00"))
    if want_kws:
        value = ";".join(want_kws).encode("utf-16-le") + b"\x00\x00"
        pad_even()
        voff = len(buf)
        buf += value
        new.append((TAG_XPKEYWORDS, struct.pack(bo + "HHII", TAG_XPKEYWORDS, 1, len(value), voff)))
    allents = sorted([(struct.unpack(bo + "H", e[:2])[0], e) for e in kept] + new, key=lambda t: t[0])
    body = struct.pack(bo + "H", len(allents)) + b"".join(e for _, e in allents) + nxt

    if len(allents) <= len(entries):                      # fits where the old IFD0 was
        buf[ifd_off:ifd_off + len(body)] = body
    else:                                                 # grow: new IFD0 at the end, header points to it
        pad_even()
        buf[4:8] = struct.pack(bo + "I", len(buf))
        buf += body
    out = bytes(buf)
    if len(out) + len(EXIF_ID) > MAX_SEGMENT:
        raise Skip("EXIF block would exceed the 64 KB JPEG segment limit")
    return out


# ---- JPEG -------------------------------------------------------------------------------------------------------
def split_jpeg(data: bytes):
    """-> ([(marker, raw segment bytes)], rest) where rest starts at the scan (SOS) and is never modified."""
    if data[:2] != b"\xff\xd8":
        raise Skip("not a JPEG")
    segs, pos, n = [], 2, len(data)
    while True:
        if pos + 2 > n:
            raise Skip("JPEG ends before the picture data")
        if data[pos] != 0xFF:
            raise Skip("unexpected bytes between JPEG segments")
        marker = data[pos + 1]
        if marker == 0xFF:
            pos += 1
            continue
        if marker == 0xDA:
            return segs, data[pos:]
        if marker == 0xD9:
            raise Skip("JPEG has no picture data")
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            segs.append((marker, data[pos:pos + 2]))
            pos += 2
            continue
        if pos + 4 > n:
            raise Skip("truncated JPEG segment")
        length = int.from_bytes(data[pos + 2:pos + 4], "big")
        if length < 2 or pos + 2 + length > n:
            raise Skip("corrupt JPEG segment")
        segs.append((marker, data[pos:pos + 2 + length]))
        pos += 2 + length


def _app1(payload: bytes) -> bytes:
    if len(payload) > MAX_SEGMENT:
        raise Skip("metadata too large for a JPEG segment")
    return b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload


def edit_jpeg(data: bytes, meta: ImageMeta, rating=True, model_fields=True, keywords=True) -> "bytes | None":
    segs, rest = split_jpeg(data)
    pay = lambda raw: raw[4:]
    i_exif = next((i for i, (m, raw) in enumerate(segs) if m == 0xE1 and pay(raw).startswith(EXIF_ID)), None)
    i_xmp = next((i for i, (m, raw) in enumerate(segs) if m == 0xE1 and pay(raw).startswith(XMP_ID)), None)
    if any(m == 0xE1 and pay(raw).startswith(XMP_EXT_ID) for m, raw in segs):
        raise Skip("has extended XMP (more than 64 KB of XMP); use imageautorater-write-xmp")

    old_xml = pay(segs[i_xmp][1])[len(XMP_ID):].rstrip(b"\x00") if i_xmp is not None else None

    def build(with_models: bool):
        root = parse_xmp(old_xml)
        apply_meta(root, meta, rating=rating, model_fields=with_models, keywords=keywords)
        return root, XMP_ID + render_xmp(root)

    try:
        prev_auto = previous_auto_keywords(parse_xmp(old_xml))
        root, xmp = build(model_fields)
        if len(xmp) > MAX_SEGMENT:      # too many models for one packet: keep the summary only
            root, xmp = build(False)
    except ET.ParseError:
        raise Skip("existing XMP is unreadable")

    tiff = pay(segs[i_exif][1])[len(EXIF_ID):] if i_exif is not None else None
    kws = None
    if keywords and meta.keywords is not None:
        _, user_xp = read_exif_tags(tiff)
        merged, seen = [], set()
        for k in [k for k in user_xp if k not in prev_auto] + keyword_list(root):
            if k.lower() not in seen:
                seen.add(k.lower())
                merged.append(k)
        kws = merged
    new_tiff = edit_exif(tiff, meta.rating if rating else None, kws)

    out_segs = [raw for _, raw in segs]
    exif_changed = new_tiff is not None
    if exif_changed:
        seg = _app1(EXIF_ID + new_tiff)
        if i_exif is not None:
            out_segs[i_exif] = seg
        else:
            at = next((i for i, (m, _) in enumerate(segs) if m != 0xE0), len(segs))  # after the JFIF APP0
            out_segs.insert(at, seg)
            if i_xmp is not None and i_xmp >= at:
                i_xmp += 1
            i_exif = at
    xmp_seg = _app1(xmp)
    if i_xmp is not None:
        out_segs[i_xmp] = xmp_seg
    else:
        at = (i_exif + 1) if i_exif is not None else next((i for i, (m, _) in enumerate(segs) if m != 0xE0), len(segs))
        out_segs.insert(at, xmp_seg)
    out = b"\xff\xd8" + b"".join(out_segs) + rest
    if out == data:
        return None
    check_segs, check_rest = split_jpeg(out)   # the result must parse, and the picture data must be untouched
    if check_rest != rest:
        raise Skip("internal check failed: picture data would change")
    return out


# ---- PNG --------------------------------------------------------------------------------------------------------
def split_png(data: bytes):
    if data[:8] != PNG_SIG:
        raise Skip("not a PNG")
    chunks, pos = [], 8
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        end = pos + 12 + length
        if end > len(data):
            raise Skip("truncated PNG chunk")
        chunks.append((data[pos + 4:pos + 8], data[pos + 8:pos + 8 + length], data[pos:end]))
        pos = end
    if not chunks or chunks[-1][0] != b"IEND":
        raise Skip("PNG has no IEND chunk")
    return chunks


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return len(payload).to_bytes(4, "big") + kind + payload + (zlib.crc32(kind + payload) & 0xFFFFFFFF).to_bytes(4, "big")


def edit_png(data: bytes, meta: ImageMeta, rating=True, model_fields=True, keywords=True) -> "bytes | None":
    chunks = split_png(data)
    i_xmp = next((i for i, (k, p, _) in enumerate(chunks) if k == b"iTXt" and p.startswith(PNG_XMP_KEY)), None)
    old = None
    if i_xmp is not None:
        body = chunks[i_xmp][1][len(PNG_XMP_KEY):]
        if body[:1] != b"\x00":
            raise Skip("PNG XMP is compressed")
        old = body[2:].split(b"\x00", 2)[2].rstrip(b"\x00")   # skip flag/method, language tag, translated keyword
    try:
        root = parse_xmp(old)
    except ET.ParseError:
        raise Skip("existing XMP is unreadable")
    apply_meta(root, meta, rating=rating, model_fields=model_fields, keywords=keywords)
    new_chunk = _png_chunk(b"iTXt", PNG_XMP_KEY + b"\x00\x00\x00\x00" + render_xmp(root))
    raws = [raw for _, _, raw in chunks]
    if i_xmp is not None:
        raws[i_xmp] = new_chunk
    else:
        at = next((i for i, (k, _, _) in enumerate(chunks) if k == b"IDAT"), len(chunks) - 1)
        raws.insert(at, new_chunk)
    out = PNG_SIG + b"".join(raws)
    return None if out == data else out


EDITORS = {".jpg": edit_jpeg, ".jpeg": edit_jpeg, ".png": edit_png}


# ---- command ----------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--backup", type=Path, metavar="DIR", help="copy each original into DIR (same relative path) before changing it")
    ap.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation before modifying files")
    a = ap.parse_args(argv)

    folder = a.folder.resolve()
    results = load_results(folder, a.results)
    opts = dict(rating=not a.no_rating, model_fields=not a.no_model_fields, keywords=not a.no_keywords)

    todo, unsupported, missing = [], Counter(), 0
    for rel, meta in iter_meta(results):
        img = folder / rel
        if not img.is_file():
            missing += 1
        elif img.suffix.lower() in EDITORS:
            todo.append((rel, img, meta))
        else:
            unsupported[img.suffix.lower() or "(none)"] += 1

    def plan():
        """Which files would change (nothing is written)."""
        changing, same, skipped = [], 0, Counter()
        for rel, img, meta in todo:
            try:
                if EDITORS[img.suffix.lower()](img.read_bytes(), meta, **opts) is None:
                    same += 1
                else:
                    changing.append(rel)
            except Skip as s:
                skipped[str(s)] += 1
        return changing, same, skipped

    changing, same, skipped = plan()
    print(f"{len(changing)} files would change, {same} already up to date" + (f", {sum(skipped.values())} skipped" if skipped else ""))
    for reason, n in skipped.items():
        print(f"  skipped {n}: {reason}")
    if unsupported:
        print("  not touched (formats that are never edited in place): "
              + ", ".join(f"{n} {ext}" for ext, n in unsupported.most_common())
              + "  -> use imageautorater-write-xmp for these")
    if missing:
        print(f"  {missing} photos from the results no longer exist")
    if a.dry_run or not changing:
        print("Dry run: nothing was changed." if a.dry_run else "Nothing to do.")
        return
    if not a.yes:
        if not sys.stdin or not sys.stdin.isatty():
            sys.exit("Refusing to modify files without confirmation: re-run with --yes (and consider --backup DIR).")
        print(f"\nThis edits {len(changing)} image files in {folder} in place (metadata only, pictures are not re-encoded).")
        print("Backups: " + (str(a.backup) if a.backup else "none (use --backup DIR to keep the originals)"))
        if input("Continue? [y/N] ").strip().lower() != "y":
            sys.exit("Aborted, nothing changed.")

    cache_path = folder / CACHE_NAME
    cache = load_cache(cache_path) if cache_path.exists() else None
    done, failed = 0, []
    wanted = set(changing)
    for rel, img, meta in todo:
        if rel not in wanted:
            continue
        try:
            data = img.read_bytes()
            new = EDITORS[img.suffix.lower()](data, meta, **opts)
            if new is None:
                continue
            st = img.stat()
            old_fp = fingerprint(img)
            if a.backup:
                dst = a.backup / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(img, dst)
            tmp = img.with_name(img.name + ".tmp")
            tmp.write_bytes(new)
            os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))   # keep the modified time
            os.replace(tmp, img)
            if cache is not None and cache["files"].get(rel) == old_fp:
                cache["files"][rel] = fingerprint(img)   # the pixels did not change: keep its scores valid
            done += 1
        except (Skip, OSError) as e:
            failed.append((rel, str(e)))
    if cache is not None:
        save_cache(cache_path, cache)
    print(f"Updated {done} image files." + (f" {len(failed)} failed:" if failed else ""))
    for rel, why in failed[:10]:
        print(f"  {rel}: {why}", file=sys.stderr)


if __name__ == "__main__":
    main()
