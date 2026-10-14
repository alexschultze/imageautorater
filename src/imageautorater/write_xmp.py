"""Write the results of imageautorater-rate into XMP sidecar files (final step; the rating itself writes nothing).

  imageautorater-write-xmp "C:\\Photos\\2025"             # photo.jpg -> photo.xmp next to it
  imageautorater-write-xmp "C:\\Photos\\2025" --dry-run   # show what would change, write nothing

Your image files are not touched. Existing sidecars are updated in place: everything else in them (develop
settings, your own keywords, ...) is preserved, and only unchanged sidecars are left alone, so re-running is cheap.
To store the data inside the image files instead, use imageautorater-write-exif.
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from imageautorater.metadata import apply_meta, iter_meta, parse_xmp, render_xmp
from imageautorater.pipeline import RESULTS_NAME, backup_file


def sidecar_path(img: Path, darktable_naming: bool = False) -> Path:
    return img.with_name(img.name + ".xmp") if darktable_naming else img.with_suffix(".xmp")


def add_common_args(ap: argparse.ArgumentParser):
    ap.add_argument("folder", type=Path, help="photo folder that holds iqa_results.json")
    ap.add_argument("--results", type=Path, help=f"results file (default: <folder>/{RESULTS_NAME})")
    ap.add_argument("--no-rating", action="store_true", help="do not write the star rating (xmp:Rating)")
    ap.add_argument("--no-keywords", action="store_true", help="do not write keywords (image types and people)")
    ap.add_argument("--no-model-fields", action="store_true",
                    help="do not write the per-model ratings and scores (iqa:Rating_<model>, iqa:Score_<model>)")
    ap.add_argument("--dry-run", action="store_true", help="report what would be written, change nothing")


def load_results(folder: Path, results: "Path | None") -> dict:
    path = results or folder / RESULTS_NAME
    if not path.is_file():
        sys.exit(f"{path} not found. Run imageautorater-rate on this folder first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("images"):
        sys.exit("The results file contains no images.")
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--darktable-naming", action="store_true", help="write IMG.jpg.xmp instead of IMG.xmp")
    a = ap.parse_args(argv)

    folder = a.folder.resolve()
    results = load_results(folder, a.results)
    written = unchanged = missing = 0
    for rel, meta in iter_meta(results):
        img = folder / rel
        if not img.is_file():
            missing += 1
            continue
        xmp = sidecar_path(img, a.darktable_naming)
        old = xmp.read_bytes() if xmp.exists() else None
        try:
            root = parse_xmp(old)
        except ET.ParseError:
            if not a.dry_run:
                print(f"  {xmp.name} was unreadable; kept as {backup_file(xmp).name}", file=sys.stderr)
            old, root = None, parse_xmp(None)
        apply_meta(root, meta, rating=not a.no_rating, model_fields=not a.no_model_fields, keywords=not a.no_keywords)
        new = render_xmp(root)
        if new == old:
            unchanged += 1
            continue
        written += 1
        if not a.dry_run:
            tmp = xmp.with_name(xmp.name + ".tmp")
            tmp.write_bytes(new)
            os.replace(tmp, xmp)
    verb = "would write" if a.dry_run else "wrote"
    print(f"{verb} {written} sidecars, {unchanged} already up to date"
          + (f", {missing} photos from the results no longer exist" if missing else ""))


if __name__ == "__main__":
    main()
