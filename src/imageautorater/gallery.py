#!/usr/bin/env python3
"""Build a browsable web page from the results of imageautorater-rate.

  imageautorater-gallery "C:\\Photos"            # writes <folder>\\gallery.html (+ gallery_data\\), opens nothing
  imageautorater-gallery "C:\\Photos" --open     # ... and opens it in your browser

The page shows the best photos per folder (by default per top-level folder, e.g. per year, 50 each). In the page you
choose which folders take part (a folder always includes its subfolders), how to rank (overall, stars, any model, or a
blend of models with your own weights) and you can compare rankings: side by side, or as the change in rank from one
ranking to another. Click a photo for a large view; every photo links to its original file.

It reads <folder>/iqa_results.json (written by imageautorater-rate) and never modifies your photos. Thumbnails of ALL
photos are made once (in parallel) in <folder>/.iqa_thumbs/ and reused; the page data is split into one small file per
top-level folder (gallery_data/) that the page loads when the folder is selected, so it stays fast for big libraries.
"""
import argparse
import hashlib
import json
import os
import sys
import webbrowser
from pathlib import Path
from urllib.parse import quote

from imageautorater.gallery_thumbs import THUMB_DIR_NAME, default_jobs, ensure_thumbs, make_thumb, thumb_file  # noqa: F401
from imageautorater.pipeline import RESULTS_NAME, fingerprint

BROWSER_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}  # shown directly in the large view
PAGE_DIR = Path(__file__).with_name("gallery_page")


def norm_score(score, info, lo_hi):
    """Raw model score -> 0..100, higher = better (direction and range taken from the model's own metadata)."""
    lo, hi = (info.get("range") or lo_hi)
    if hi <= lo:
        return 50.0
    frac = max(0.0, min(1.0, (score - lo) / (hi - lo)))
    return 100.0 * ((1 - frac) if info.get("lower_better") else frac)


def top_key(rel_folder: str) -> str:
    """The data file an image belongs to: its first folder, "" for photos lying directly in the root."""
    return rel_folder.split("/", 1)[0]


def build_data(folder: Path, results: dict, out_dir: Path, thumb_dir: Path, thumbs: bool, thumb_size: int, jobs: int = 1):
    """Returns (items, model names). The per-model values of an item are lists aligned with the model names."""
    models = results.get("models", {})
    # models without a known score range: use the observed min/max over this folder
    observed = {}
    for entry in results["images"].values():
        for m, v in entry.get("models", {}).items():
            lo, hi = observed.get(m, (v["score"], v["score"]))
            observed[m] = (min(lo, v["score"]), max(hi, v["score"]))
    model_names = sorted(observed)

    def url(p: Path) -> str:
        return quote(Path(os.path.relpath(p, out_dir)).as_posix(), safe="/")

    present, missing, todo = [], 0, []
    for rel, entry in sorted(results["images"].items()):
        src = folder / rel
        if not src.is_file():
            missing += 1
            continue
        dst = None
        if thumbs:
            dst = thumb_file(thumb_dir, rel, fingerprint(src), thumb_size)
            if not dst.exists():
                todo.append((rel, src, dst))
        present.append((rel, entry, src, dst))
    failed = ensure_thumbs(todo, thumb_size, jobs) if todo else {}
    for rel, err in failed.items():
        print(f"  no thumbnail for {rel}: {err}", file=sys.stderr)

    items = []
    for rel, entry, src, dst in present:
        browser = src.suffix.lower() in BROWSER_EXT
        thumb = dst if dst is not None and dst.exists() else None
        per_model, norms = [], []
        for m in model_names:
            v = entry.get("models", {}).get(m)
            if not v:
                per_model.append(None)
                continue
            nv = norm_score(v["score"], models.get(m, {}), observed.get(m, (0, 1)))
            per_model.append([v["rating"], v["score"], round(nv, 1)])
            norms.append(nv)
        folder_rel = Path(rel).parent.as_posix()
        items.append({
            "p": rel,
            "f": "" if folder_rel == "." else folder_rel,
            "t": url(thumb) if thumb else url(src) if browser else "",
            "u": url(src),
            "o": round(sum(norms) / len(norms), 1) if norms else None,
            "r": entry.get("rating_mean"),
            "m": per_model,
            "g": list(entry.get("tags", {})),
            "n": sorted({p["name"] for p in entry.get("people", []) if p.get("name")}),
        })
        if browser:
            items[-1]["b"] = 1                       # the original can be shown in the large view
    if missing:
        print(f"{missing} photos listed in the results no longer exist and are left out", file=sys.stderr)
    return items, model_names


def js_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")


def write_page(out: Path, items: list, model_names: list, results: dict, folder: Path, top: int, embed: bool):
    """gallery.html (index + code) and, unless embed, gallery_data/<group>.js loaded by the page on demand."""
    lower = [bool(results.get("models", {}).get(m, {}).get("lower_better")) for m in model_names]
    by_group = {}
    for it in items:
        by_group.setdefault(top_key(it["f"]), []).append(it)
    data_dir = out.with_name(out.stem + "_data")
    if not embed and data_dir.is_dir():
        for old in data_dir.glob("g_*.js"):         # groups that no longer exist
            old.unlink()
    groups, scripts = [], []
    for k in sorted(by_group):
        name = "g_" + hashlib.md5(k.encode("utf-8")).hexdigest()[:10] + ".js"
        body = "(window.__IQA_DATA=window.__IQA_DATA||{})[" + js_json(k) + "]=" + js_json(by_group[k]) + ";\n"
        groups.append({"k": k, "n": len(by_group[k]), "file": quote(f"{data_dir.name}/{name}")})
        if embed:
            scripts.append("<script>" + body + "</script>")
        else:
            data_dir.mkdir(parents=True, exist_ok=True)
            (data_dir / name).write_text(body, encoding="utf-8")
    folders = {}
    for it in items:
        folders[it["f"]] = folders.get(it["f"], 0) + 1
    index = {"title": folder.name or str(folder), "root": str(results.get("folder") or folder), "mode": results.get("mode", ""),
             "generated": results.get("generated", ""), "top": top, "models": model_names, "lower": lower,
             "groups": groups, "folders": folders,
             "people": sorted({n for it in items for n in it["n"]}), "tags": sorted({t for it in items for t in it["g"]})}
    page = (PAGE_DIR / "index.html").read_text(encoding="utf-8")
    for marker, value in (("/*__CSS__*/", (PAGE_DIR / "gallery.css").read_text(encoding="utf-8")),
                          ("/*__INDEX__*/null", js_json(index)),
                          ("<!--__DATA__-->", "\n".join(scripts)),
                          ("/*__LOGIC__*/", (PAGE_DIR / "logic.js").read_text(encoding="utf-8")),
                          ("/*__JS__*/", (PAGE_DIR / "gallery.js").read_text(encoding="utf-8"))):
        page = page.replace(marker, value)
    out.write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="photo folder that holds iqa_results.json")
    ap.add_argument("--results", type=Path, help=f"results file (default: <folder>/{RESULTS_NAME})")
    ap.add_argument("--out", type=Path, help="output page (default: <folder>/gallery.html)")
    ap.add_argument("--top", type=int, default=50, help="photos shown per folder before 'Show more' (default 50)")
    ap.add_argument("--thumb-size", type=int, default=360, help="thumbnail size in pixels (default 360)")
    ap.add_argument("--thumb-dir", type=Path, help=f"folder for the thumbnails (default: <folder>/{THUMB_DIR_NAME})")
    ap.add_argument("--jobs", type=int, default=default_jobs(), help="parallel processes for thumbnails (default %(default)s)")
    ap.add_argument("--thumbs-only", action="store_true", help="only make the thumbnails, do not write the page")
    ap.add_argument("--no-thumbs", action="store_true", help="use the original files as thumbnails (slow for big photos)")
    ap.add_argument("--embed", action="store_true", help="put all data into gallery.html (one file, but large for big libraries)")
    ap.add_argument("--open", action="store_true", help="open the page in the default browser when done")
    a = ap.parse_args()

    folder = a.folder.resolve()
    results_path = a.results or folder / RESULTS_NAME
    if not results_path.is_file():
        sys.exit(f"{results_path} not found. Run imageautorater-rate on this folder first.")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if not results.get("images"):
        sys.exit("The results file contains no images.")
    out = (a.out or folder / "gallery.html").resolve()
    thumb_dir = (a.thumb_dir or folder / THUMB_DIR_NAME).resolve()
    if a.thumbs_only and a.no_thumbs:
        sys.exit("--thumbs-only and --no-thumbs exclude each other.")

    items, model_names = build_data(folder, results, out.parent, thumb_dir, not a.no_thumbs, a.thumb_size, max(1, a.jobs))
    if not items:
        sys.exit("None of the photos in the results exist any more.")
    if a.thumbs_only:
        print(f"{len(items)} photos, thumbnails in {thumb_dir}")
        return
    write_page(out, items, model_names, results, folder, a.top, a.embed)
    print(f"{len(items)} photos, {len(model_names)} models -> {out}")
    if a.open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
