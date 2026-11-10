#!/usr/bin/env python3
"""Copy photos from a list into a subfolder of the photo folder (the command-line twin of the gallery's copy button).

  imageautorater-export "D:\\Photos" selection.txt --to _gallery/selection
  imageautorater-export "D:\\Photos" selection.txt --to _gallery/best --flat --dry-run

The list has one photo per line, relative to the photo folder or as a full path inside it (what the gallery's
"Download list" and "Export list" write). Originals are only read. A file that already exists in the target with the same
size is skipped, so the command can be repeated. Layout: the photo's own folders below the target (default) or --flat
(everything in the target; a name clash gets _2, _3, ...).
"""
import argparse
import shutil
import sys
from pathlib import Path


def read_list(root: Path, list_file: Path) -> list:
    """Relative paths (posix style) of the listed photos; lines that are not inside the photo folder are reported."""
    prefix = str(root).replace("\\", "/").rstrip("/") + "/"
    rels, seen = [], set()
    for line in list_file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip().strip('"')
        if not line or line.startswith("#"):
            continue
        norm = line.replace("\\", "/")
        if norm.lower().startswith(prefix.lower()):
            norm = norm[len(prefix):]
        elif Path(line).is_absolute() or (len(norm) > 1 and norm[1] == ":"):
            print(f"  not inside {root}: {line}", file=sys.stderr)
            continue
        if ".." in norm.split("/"):
            print(f"  ignored (leaves the photo folder): {line}", file=sys.stderr)
            continue
        if norm not in seen:
            seen.add(norm)
            rels.append(norm)
    return rels


def plan_copy(rels: list, flat: bool) -> list:
    """[(source rel, target rel)]; same rules as the gallery page."""
    used, out = set(), []
    for rel in rels:
        dst = rel
        if flat:
            base = rel.rsplit("/", 1)[-1]
            stem, dot, ext = base.rpartition(".")
            stem, ext = (stem, "." + ext) if dot and stem else (base, "")
            dst, k = base, 2
            while dst.lower() in used:
                dst, k = f"{stem}_{k}{ext}", k + 1
        used.add(dst.lower())
        out.append((rel, dst))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="the photo folder")
    ap.add_argument("list", type=Path, help="text file with one photo per line")
    ap.add_argument("--to", default="_gallery/selection", help="subfolder of the photo folder (default _gallery/selection)")
    ap.add_argument("--flat", action="store_true", help="all photos directly in the target folder")
    ap.add_argument("--dry-run", action="store_true", help="only report what would be copied")
    a = ap.parse_args(argv)

    root = a.folder.resolve()
    if not root.is_dir():
        sys.exit(f"{root} is not a folder.")
    parts = [x for x in a.to.replace("\\", "/").split("/") if x]
    if not parts or any(x in (".", "..") for x in parts):
        sys.exit("--to must be a folder name below the photo folder, without '..'.")
    target = root.joinpath(*parts)
    rels = read_list(root, a.list)
    if not rels:
        sys.exit("The list contains no photos.")
    copied = skipped = 0
    failed = []
    for src_rel, dst_rel in plan_copy(rels, a.flat):
        src, dst = root / src_rel, target / dst_rel
        try:
            if not src.is_file():
                raise FileNotFoundError("not found")
            if dst.is_file() and dst.stat().st_size == src.stat().st_size:
                skipped += 1
                continue
            if not a.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            copied += 1
        except OSError as e:
            failed.append((src_rel, str(e)[:80]))
    print(f"{'Would copy' if a.dry_run else 'Copied'} {copied}, already there {skipped}, failed {len(failed)} -> {target}")
    for rel, err in failed[:10]:
        print(f"  failed: {rel}: {err}", file=sys.stderr)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
