#!/usr/bin/env python3
"""Rate a folder of photos with pyiqa no-reference models, add image-type keywords and recognised people,
and save everything in the photo folder:

  iqa_results.json   per image: stars and raw score for every model, mean stars, keywords, people
  iqa_scores.json    the persistent cache (resumes interrupted runs)

This command does NOT write any metadata into or next to your photos. Do that as a deliberate last step:
  imageautorater-write-xmp   <folder>   XMP sidecar files (photo.xmp), any format, photos untouched
  imageautorater-write-exif  <folder>   metadata inside JPEG/PNG files (EXIF rating + keywords + XMP)
  imageautorater-gallery     <folder>   a web page to browse the ranking

Content analysis (on by default; needs `transformers`, and `facenet-pytorch` for faces):
  * Image type keywords via CLIP zero-shot classification (portrait, landscape, food, ...).
    Edit the vocabulary with --labels labels.txt (one label per line, optional "label: prompt").
  * Person identification with ONE ground truth: a single _faces/ folder (see resolve_faces_dir: by default in the
    root of the photo folder you scan, else next to it or above it, or a custom folder via --faces-dir), shared by every
    photo folder you process. _faces/<Name>/ holds confirmed faces of that person (crops you moved there and/or
    reference photos you dropped in). Unrecognised faces are grouped into _faces/person_NNNN/ for you to
    name by renaming the folder or moving crops. Names then carry over to every photo folder.
  Keywords (recognised names first, then image types) are stored in iqa_results.json; the write commands turn them
  into photo keywords later.

Scores, CLIP embeddings and face embeddings are cached in <folder>/iqa_scores.json so an
interrupted run resumes and re-runs only score new (model, image) pairs.

Examples:
  imageautorater-rate D:\\Photos
  imageautorater-rate D:\\Photos --models musiq topiq_nr clipiqa
  imageautorater-rate D:\\Photos --mode percentile --limit 10
  python -m imageautorater D:\\Photos --faces-only
"""
import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

from imageautorater.fileio import recovery_copy, user_data_dir, write_text_atomic

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
RAW_EXT = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}
CACHE_NAME = "iqa_scores.json"
RESULTS_NAME = "iqa_results.json"

_STAMPS = {}   # str(path) -> "size:mtime" read from the directory listing, so no extra request per file is needed


def find_images(folder: Path, recursive: bool, include_raw: bool, exclude=(), progress=None):
    """Photos in a folder. Never returned: anything inside a hidden (dot) folder such as the gallery's .iqa_thumbs, or in
    any _faces folder, and anything inside one of the `exclude` folders (e.g. a custom faces folder in the photo folder).
    progress(n_entries_looked_at, n_photos_found) is called every 500 entries so a big folder shows life signs.

    Uses os.scandir: the size and modified time of every file come with the directory listing, which on a network share
    (NAS, SMB) is the difference between minutes and many tens of minutes compared with asking per file. Those stamps
    are remembered for fingerprint(), so the cache check needs no second pass over the files."""
    exts = IMG_EXT | (RAW_EXT if include_raw else set())
    skip = set()
    for e in exclude:
        skip.add(os.path.normcase(os.path.abspath(str(e))))
        skip.add(os.path.normcase(str(Path(e).resolve())))
    out, seen, stack = [], 0, [str(folder)]
    while stack:
        top = stack.pop()
        try:
            it = os.scandir(top)
        except OSError as err:
            print(f"WARNING: cannot read {top}: {err}", file=sys.stderr)
            continue
        with it:
            for entry in it:
                seen += 1
                if progress and seen % 500 == 0:
                    progress(seen, len(out))
                try:
                    if entry.is_dir():
                        name = entry.name
                        if (recursive and not name.startswith(".") and name != "_faces"
                                and os.path.normcase(os.path.abspath(entry.path)) not in skip):
                            stack.append(entry.path)
                        continue
                    if not entry.is_file() or os.path.splitext(entry.name)[1].lower() not in exts:
                        continue
                    st = entry.stat()
                except OSError:
                    continue
                path = Path(entry.path)
                _STAMPS[str(path)] = f"{st.st_size}:{int(st.st_mtime)}"
                out.append(path)
    return sorted(out)


def load_image(path: Path, max_side: int):
    from PIL import Image, ImageOps

    if path.suffix.lower() in RAW_EXT:
        try:
            import rawpy
        except ImportError:
            raise RuntimeError("RAW support needs `pip install rawpy`")
        with rawpy.imread(str(path)) as raw:
            img = Image.fromarray(raw.postprocess(use_camera_wb=True, half_size=True))
    else:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def available_models():
    ensure_imported("pyiqa", "Loading pyiqa and PyTorch (nothing is installed; the first start can take a minute or more)")
    import pyiqa

    try:
        return sorted(pyiqa.list_models(metric_mode="NR"))
    except TypeError:  # older pyiqa
        return sorted(m for m in pyiqa.list_models() if "NR" in str(m) or True)


def new_cache():
    # models: name -> {lower_better, range}; scores: model -> {relpath: score}; files: relpath -> "size:mtime"
    # clip: relpath -> image embedding; faces: relpath -> [{box, prob, emb}]; gallery: ref photo -> {fp, emb}
    return {"version": 3, "max_side": None, "models": {}, "scores": {}, "files": {},
            "clip_model": None, "clip": {}, "clip_text": None, "faces": {}, "gallery": {},
            "clusters_next": 1,
            "unreadable": {},   # "<relpath>" -> why the photo is damaged; skipped until the file changes
            # "<relpath>#<face index>" -> name (placed in a named folder) or None (placed in a _folder = not a person)
            "face_labels": {}}


def backup_file(path: Path):
    """Move an existing file aside as <name>.bak (numbered if needed); returns the new path."""
    dst = path.with_name(path.name + ".bak")
    i = 1
    while dst.exists():
        dst = path.with_name(f"{path.name}.bak{i}")
        i += 1
    path.replace(dst)
    return dst


def _read_cache_file(p: Path):
    """The parsed cache file, or None when it is missing or unreadable."""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def load_cache(path: Path):
    """Load the persistent cache. An unreadable cache is backed up, never silently discarded. A leftover .tmp file from a
    save that could not finish (locked file on a share, crash) holds the newest complete data and is used when it is
    newer than the cache itself."""
    cache = new_cache()
    tmp = path.with_name(path.name + ".tmp")
    main, spare = (_read_cache_file(p) if p.exists() else None for p in (path, tmp))
    data = main
    if spare is not None and (main is None or tmp.stat().st_mtime >= path.stat().st_mtime):
        data = spare
        print(f"Recovered the cache from {tmp.name}, left behind by a save that could not finish.", file=sys.stderr)
    if data is None and path.exists():
        print(f"WARNING: could not read {path.name}; moved to {backup_file(path).name}", file=sys.stderr)
    if data is not None:
        for k in cache:
            if k in data:
                cache[k] = data[k]
    n = sum(len(v) for v in cache["scores"].values())
    if n:
        print(f"Loaded cache: {len(cache['scores'])} models, {n} scores")
    return cache


_cache_error = None     # why the last cache save failed (None = it worked)
_last_autosave = 0.0
_ACTIVE = {}            # the cache of the running scan, so a crash or Ctrl+C can still save it


def save_cache(path: Path, cache) -> bool:
    """Save the cache. Never raises: on a network share or synced folder the file can be locked for a while (sync client,
    antivirus, another PC), so the write is retried, then done in place, and if the share still refuses the problem is
    reported while the results stay in memory. Compact JSON: embeddings make the cache large, and a NAS is slow."""
    global _cache_error, _last_autosave
    try:
        write_text_atomic(path, json.dumps(cache, separators=(",", ":")))
    except OSError as err:
        _cache_error = str(err)
        print(f"WARNING: could not save {path.name} ({err}). The results stay in memory; saving is tried again.", file=sys.stderr)
        return False
    _cache_error = None
    _last_autosave = time.time()
    return True


def autosave(path: Path, cache, every=60):
    """Keep progress if the run is interrupted, but at most once a minute: the cache can be tens of megabytes."""
    if time.time() - _last_autosave >= every:
        save_cache(path, cache)


def finalize_cache(path: Path, cache):
    """Last chance at the end of a run when the previous save failed: try once more, else keep a local recovery copy and
    list the problem, so a finished run is never lost."""
    if _cache_error is None:
        return
    if save_cache(path, cache):
        log(f"{path.name} was saved after all")
        return
    err = _cache_error
    try:
        rec = recovery_copy(path.name, json.dumps(cache, separators=(",", ":")))
    except OSError:
        rec = None
    note_problem(f"{path.name} could not be saved to {path.parent} ({err})"
                 + (f"; a copy of everything scored is in {rec}" if rec else ""),
                 "close whatever has the file open (the sync client, antivirus, another PC running this tool), or check the "
                 "share's permissions, then run the same command again; it continues from the saved copy")


def fingerprint(p: Path):
    """Size and modified time (whole seconds) of a file. Uses the stamp find_images already read from the directory
    listing, once; otherwise asks the file system."""
    hint = _STAMPS.pop(str(p), None)
    if hint:
        return hint
    st = p.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


_last_len = 0


def status(msg):
    """Single updating status line on a terminal; plain lines when piped to a file. status(None) ends the line."""
    global _last_len
    if not sys.stdout.isatty():
        if msg:
            print(msg, flush=True)
        return
    if msg is None:
        sys.stdout.write("\r" + " " * _last_len + "\r")
    else:
        msg = msg[:110]
        sys.stdout.write("\r" + msg.ljust(_last_len))
        _last_len = len(msg)
    sys.stdout.flush()


_T0 = time.time()


def log(msg):
    """A progress line stamped with the time since the program started: [mm:ss] message."""
    status(None)
    t = int(time.time() - _T0)
    print(f"[{t // 60:02d}:{t % 60:02d}] {msg}", flush=True)


def ensure_imported(module: str, why: str):
    """Import a heavy module, saying so first: importing PyTorch can take a minute or more (first start after
    installing, antivirus scanning, slow disk) and prints nothing by itself."""
    if module in sys.modules:
        return
    import importlib

    t0 = time.time()
    log(f"{why} ...")
    importlib.import_module(module)
    log(f"{module} loaded in {time.time() - t0:.0f}s")


PROBLEMS = []   # (what did not run, how to fix it); printed prominently at the end of the run


def note_problem(what, fix=None):
    PROBLEMS.append((what, fix))


def report_problems():
    """Print what did not run and how to fix it, framed so it cannot be missed among the download progress output."""
    if not PROBLEMS:
        return
    print("\n" + "=" * 100)
    print(f"ATTENTION: {len(PROBLEMS)} thing(s) did not run, so the results are incomplete:")
    for what, fix in PROBLEMS:
        print(f"  - {what}" + (f"\n      fix: {fix}" if fix else ""))
    print("Run the same command again after fixing; everything already done stays in the cache.")
    print("=" * 100 + "\n")


def is_damaged_image(err) -> bool:
    """True when a load error means the FILE is damaged (truncated, 'broken data stream', not an image), false for
    errors of the file system or network (permission, connection lost, ...), which carry an errno and may pass later.
    Only damaged photos are remembered as unreadable; a dropped connection must never blacklist a good photo."""
    return not (isinstance(err, OSError) and err.errno not in (None, 0))


def mark_unreadable(cache, rel, err):
    cache["unreadable"][rel] = f"{type(err).__name__}: {str(err)[:80]}"


def report_unreadable(cache, rels):
    bad = {k: v for k, v in cache["unreadable"].items() if k in rels}
    if bad:
        first = ", ".join(list(bad)[:5])
        log(f"NOTE: {len(bad)} photo(s) could not be read (damaged or truncated files) and were skipped: {first}"
            + (" ..." if len(bad) > 5 else "") + f". All of them are listed in {RESULTS_NAME} under 'unreadable'; they are "
            "tried again only after the file changes.")


PKG_RESOURCES_FIX = 'uv pip install "setuptools<82"   (setuptools 82+ removed pkg_resources, which the CLIP package needs)'


def model_problem_fix(reason):
    return PKG_RESOURCES_FIX if "pkg_resources" in reason else None


def prefetched(items, loader, workers=4):
    """Yield (item, value, error) in order. While the caller works on one item, a small thread pool already loads the
    next ones (JPEG decoding releases the GIL, and a network share answers several requests in parallel), so the GPU
    does not wait for the disk. An item that fails to load yields its error instead of raising."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    workers = max(1, int(workers))
    it = iter(items)
    pending = deque()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def fill():
            while len(pending) < workers * 2:
                try:
                    item = next(it)
                except StopIteration:
                    return
                pending.append((item, pool.submit(loader, item)))

        fill()
        while pending:
            item, fut = pending.popleft()
            fill()
            try:
                yield item, fut.result(), None
            except Exception as err:
                yield item, None, err


def fmt_eta(elapsed, n, total):
    if n < 2:
        return "ETA --"
    rem = elapsed / (n - 1) * (total - n + 1)
    return f"{elapsed / (n - 1):.1f}s/img  ETA {int(rem // 60)}m{int(rem % 60):02d}s"


def parse_range(value):
    """A model's score range as [low, high], or None when unknown. pyiqa publishes it as text such as '~0, ~100' or
    '1, 5' (older versions as a tuple); without it the absolute star mapping would silently fall back to ranking."""
    if value is None:
        return None
    try:
        nums = re.findall(r"-?\d+(?:\.\d+)?", value) if isinstance(value, str) else list(value)
        lo, hi = float(nums[0]), float(nums[1])
    except (TypeError, ValueError, IndexError):
        return None
    return [lo, hi] if hi > lo else None


def config_range(name):
    """The score range pyiqa lists for a model, read from its built-in table (no weights are loaded)."""
    try:
        from pyiqa.default_model_configs import DEFAULT_CONFIGS
    except Exception:
        return None
    return parse_range(DEFAULT_CONFIGS.get(name, {}).get("score_range"))


def run_models(folder, paths, models, device, max_side, cache, cache_path, io_threads=4):
    ensure_imported("pyiqa", "Loading pyiqa (nothing is installed; this can take a while on the first start)")
    import numpy as np
    import pyiqa
    import torch

    dev = pick_device(device)
    print(f"device: {dev}   models: {len(models)}   images: {len(paths)}")
    rel = {p: str(p.relative_to(folder)).replace("\\", "/") for p in paths}

    for mi, name in enumerate(models, 1):
        done = cache["scores"].setdefault(name, {})
        todo = [p for p in paths if rel[p] not in done and rel[p] not in cache["unreadable"]]
        if not todo and name in cache["models"]:
            print(f"[{mi}/{len(models)}] {name}: cached")
            continue
        log(f"[{mi}/{len(models)}] {name}: loading the model (the first use downloads its weights, cached afterwards)")
        try:
            metric = pyiqa.create_metric(name, device=dev)
        except Exception as e:
            print(f"[{mi}/{len(models)}] {name}: SKIP (cannot load: {str(e)[:120]})", file=sys.stderr)
            note_problem(f"model {name} was skipped: {str(e)[:100]}", model_problem_fix(str(e)))
            if not cache["scores"].get(name):
                cache["scores"].pop(name, None)  # keep any earlier scores for this model
            continue
        rng = parse_range(getattr(metric, "score_range", None))
        cache["models"][name] = {"lower_better": bool(getattr(metric, "lower_better", False)), "range": rng}
        print(f"[{mi}/{len(models)}] {name}: {len(todo)} images  range={rng}  lower_better={cache['models'][name]['lower_better']}")

        failed = 0
        t0 = time.time()
        for n, (p, img, load_err) in enumerate(prefetched(todo, lambda q: load_image(q, max_side), io_threads), 1):
            status(f"[{mi}/{len(models)}] {name}  {n}/{len(todo)}  {fmt_eta(time.time() - t0, n, len(todo))}  {p.name}")
            autosave(cache_path, cache)   # keep progress if interrupted (at most once a minute)
            if load_err and is_damaged_image(load_err):
                mark_unreadable(cache, rel[p], load_err)   # the photo is at fault, not the model: no strike against it
                continue
            try:
                if load_err:
                    raise load_err
                t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255).unsqueeze(0)   # np.array copies: writable
                with torch.no_grad():
                    s = float(metric(t.to(dev)).item())
                if s == s and abs(s) != float("inf"):
                    done[rel[p]] = s
            except Exception as e:
                failed += 1
                if failed == 1:
                    print(f"    {name}: failed on {p.name}: {str(e)[:120]}", file=sys.stderr)
                if failed >= 3 and not done:  # model is broken for these inputs; give up on it
                    print(f"    {name}: giving up", file=sys.stderr)
                    note_problem(f"model {name} failed on the first photos and was given up: {str(e)[:100]}")
                    break
        status(None)
        print(f"[{mi}/{len(models)}] {name}: done, {len(done)}/{len(paths)} scored in {time.time() - t0:.0f}s")
        save_cache(cache_path, cache)

        del metric
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()


def to_stars(scores, info, mode):
    """scores: {key: raw}. Returns {key: 1..5}, higher = better."""
    import numpy as np

    keys = list(scores)
    vals = np.array([scores[k] for k in keys], dtype=float)
    if info["lower_better"]:
        vals = -vals
    rng = info["range"]
    if mode == "absolute" and rng and rng[1] > rng[0]:
        lo, hi = rng
        if info["lower_better"]:
            lo, hi = -hi, -lo
        frac = np.clip((vals - lo) / (hi - lo), 0, 1)
    else:  # percentile mode, or absolute fallback when the model has no known range
        frac = (vals.argsort().argsort() + 0.5) / len(vals)
    stars = np.clip(np.ceil(frac * 5), 1, 5)
    return {k: int(s) for k, s in zip(keys, stars)}


def write_results_json(path: Path, args, per_image: dict, used: list, cache: dict, content: dict = None):
    """Combined per-image results (keyed by relative path): raw score and stars per model, mean/median stars,
    plus image-type tags and identified people when content analysis ran."""
    import statistics
    from datetime import datetime

    content = content or {}
    images = {}
    for key in sorted(set(per_image) | set(content)):
        pm = per_image.get(key)
        entry = images[key] = {}
        entry.update(content.get(key, {}))
        if not pm:
            continue
        stars = [s for s, _ in pm.values()]
        images[key].update({
            "rating": int(round(sum(stars) / len(stars))),
            "rating_mean": round(sum(stars) / len(stars), 3),
            "rating_median": statistics.median(stars),
            "model_count": len(stars),
            "models": {m: {"rating": s, "score": round(sc, 5)} for m, (s, sc) in pm.items()},
        })
    out = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "folder": str(args.folder),
        "mode": args.mode,
        "max_side": args.max_side,
        "models": {m: cache["models"][m] for m in used},  # score range / direction used for stars
        "unreadable": cache.get("unreadable", {}),
        "images": images,
    }
    write_text_atomic(path, json.dumps(out, indent=1))


# --------------------------------------------------------------------------------------
# Content analysis: image-type keywords (CLIP zero-shot) and person identification
# --------------------------------------------------------------------------------------
DEFAULT_LABELS = [
    "portrait", "group photo", "selfie", "children", "baby", "wedding", "party", "crowd",
    "landscape", "mountains", "forest", "beach", "sea", "lake", "river", "waterfall", "desert",
    "snow", "sunset", "sunrise", "night", "sky", "clouds", "flowers", "garden", "field",
    "city", "street", "architecture", "building", "church", "bridge", "interior", "kitchen",
    "food", "drink", "restaurant", "dog", "cat", "bird", "horse", "wildlife", "insect", "pet",
    "car", "motorcycle", "bicycle", "boat", "train", "airplane", "sports", "concert", "festival",
    "macro", "aerial", "fireworks", "art", "museum", "document", "screenshot", "text", "map",
    "product", "furniture", "toy", "travel", "camping", "hiking", "skiing", "swimming",
]


def load_labels(path):
    """label -> CLIP prompt. File lines: 'label' or 'label: custom prompt'; '#' comments."""
    labels = {}
    if path:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            lab, _, prompt = line.partition(":")
            labels[lab.strip()] = prompt.strip() or f"a photo of {lab.strip()}"
    else:
        labels = {lab: f"a photo of {lab}" for lab in DEFAULT_LABELS}
    return labels


def nvidia_gpus():
    """NVIDIA GPUs the driver sees ('name, driver version' per line), found with nvidia-smi, independent of PyTorch.
    Empty when there is no NVIDIA driver or no GPU."""
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


GPU_FIX = ("Fix: run again with --fix-gpu (it installs the CUDA build of PyTorch into this environment, about 2-3 GB), or "
           "yourself:\n       uv pip install --reinstall torch torchvision --torch-backend=auto\n"
           "     (On Windows the default PyTorch from PyPI is CPU-only, and `uv sync` in a source checkout also installs the "
           "CPU build.)")


def hardware_report(torch, dev, gpus):
    """(summary line, warning or None) for the chosen device. The warning explains why an NVIDIA GPU is NOT used."""
    cuda = getattr(torch.version, "cuda", None)
    build = f"CUDA {cuda}" if cuda else ("ROCm " + torch.version.hip if getattr(torch.version, "hip", None) else "CPU-only build")
    name = ""
    if dev.type == "cuda":
        try:
            name = f" ({torch.cuda.get_device_name(dev.index or 0)})"
        except Exception:
            pass
    line = f"PyTorch {torch.__version__} ({build}); device: {dev}{name}"
    if dev.type != "cpu" or not gpus:
        return line, None
    if not cuda:
        why = (f"WARNING: an NVIDIA GPU is present ({gpus[0]}) but this PyTorch is a CPU-only build, so the GPU is NOT used "
               f"and everything runs on the CPU, many times slower.\n  {GPU_FIX}")
    else:
        why = (f"WARNING: an NVIDIA GPU is present ({gpus[0]}) and PyTorch has CUDA support ({cuda}), but "
               "torch.cuda.is_available() is False, so the CPU is used. Usual causes: the NVIDIA driver is too old for this "
               "CUDA version (update the driver; `nvidia-smi` shows the newest CUDA it supports), or the GPU is not visible "
               "in this session (remote desktop, a container or WSL without GPU access).")
    return line, why


def torch_cuda_build():
    """(installed, cuda) for the installed PyTorch, read from its files WITHOUT importing it. `cuda` is the CUDA version
    the build was made for, None for a CPU-only build, "unknown" when it cannot be told. Importing is avoided on purpose:
    a loaded PyTorch locks its DLLs on Windows, which would make replacing it impossible."""
    from importlib import metadata

    try:
        dist = metadata.distribution("torch")
        text = Path(dist.locate_file("torch/version.py")).read_text(encoding="utf-8")
    except metadata.PackageNotFoundError:
        return False, None
    except OSError:
        return True, "unknown"
    m = re.search(r"^cuda\b[^=\n]*=\s*(?:None|'([\d.]+)'|\"([\d.]+)\")", text, re.MULTILINE)
    if not m:
        return True, "unknown"
    return True, (m.group(1) or m.group(2))


CUDA_INDEXES = ["cu132", "cu130", "cu129", "cu128", "cu126", "cu124", "cu121", "cu118"]   # pytorch.org/whl/<name>


def driver_cuda_version():
    """The newest CUDA version the installed NVIDIA driver supports, as reported by nvidia-smi (e.g. '13.3'), or None."""
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    m = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", out)
    return m.group(1) if m else None


def choose_cuda_index(driver_cuda):
    """The newest PyTorch CUDA build the driver can run, e.g. '13.3' -> 'cu132'; cu126 when the driver is unknown."""
    try:
        have = tuple(int(x) for x in driver_cuda.split(".")[:2])
    except (AttributeError, ValueError):
        return "cu126"
    for name in CUDA_INDEXES:
        if (int(name[2:-1]), int(name[-1])) <= have:
            return name
    return CUDA_INDEXES[-1]


def gpu_fix_command(python, uv, driver_cuda):
    """The command that installs the CUDA build of PyTorch into the environment of `python`."""
    if uv:
        return [uv, "pip", "install", "--python", python, "--reinstall", "torch", "torchvision", "--torch-backend=auto"]
    return [python, "-m", "pip", "install", "--upgrade", "--force-reinstall", "torch", "torchvision",
            "--index-url", f"https://download.pytorch.org/whl/{choose_cuda_index(driver_cuda)}"]


def maybe_fix_gpu(choice, device_arg):
    """An NVIDIA GPU together with a CPU-only PyTorch is the usual reason a GPU is not used. Offer to install the CUDA
    build. choice: True = do it without asking (--fix-gpu), False = never offer (--no-fix-gpu), None = ask when running in a
    terminal, otherwise only report. Runs BEFORE PyTorch is imported. Returns True when PyTorch was replaced."""
    import shutil
    import subprocess

    if choice is False or (device_arg and choice is not True):   # --device cpu etc.: the user chose
        return False
    gpus = nvidia_gpus()
    installed, cuda = torch_cuda_build()
    if not gpus or not installed or cuda is not None:    # no NVIDIA GPU / no PyTorch yet / already a CUDA build (or unknown)
        if choice is True and not gpus:
            log("--fix-gpu: no NVIDIA GPU was found, nothing to do")
        return False
    driver = driver_cuda_version()
    cmd = gpu_fix_command(sys.executable, shutil.which("uv"), driver)
    log(f"An NVIDIA GPU is present ({gpus[0]}) but the installed PyTorch is a CPU-only build, so the GPU would NOT be used.")
    log("Fix: " + " ".join(cmd))
    if choice is None:
        if not sys.stdin or not sys.stdin.isatty():
            print("Not interactive: not installing. Run again with --fix-gpu to install it, or --no-fix-gpu to stop this message.")
            return False
        if input("Install the CUDA build of PyTorch now? It downloads about 2-3 GB and replaces torch and torchvision "
                 "in this environment. [Y/n] ").strip().lower() not in ("", "y", "yes"):
            print("Not installing; this run uses the CPU.")
            return False
    log("Installing the CUDA build of PyTorch (this takes a few minutes) ...")
    try:
        done = subprocess.run(cmd)
    except OSError as err:
        log(f"Could not start the installer: {err}")
        return False
    if done.returncode != 0:
        log(f"The installer failed (exit code {done.returncode}); continuing on the CPU. Run the command above yourself to see why.")
        return False
    check = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.is_available(), torch.version.cuda)"],
                           capture_output=True, text=True)
    log(f"PyTorch replaced. Check: cuda available / build = {check.stdout.strip() or check.stderr.strip()[-120:]}")
    return True


_hardware_reported = False


def pick_device(arg):
    """--device if given, otherwise the best available accelerator: CUDA (NVIDIA/ROCm), Intel XPU, Apple MPS, CPU.
    The first call also reports the PyTorch build and the device, and explains when an NVIDIA GPU is not used."""
    global _hardware_reported
    ensure_imported("torch", "Loading PyTorch (nothing is installed; the first start can take a minute or more)")
    import torch

    if arg and arg.startswith("cuda") and not torch.cuda.is_available():
        gpus = nvidia_gpus()
        _, why = hardware_report(torch, torch.device("cpu"), gpus or ["NVIDIA GPU"])
        sys.exit(f"--device {arg} was requested but CUDA is not available in this PyTorch.\n{why or ''}".rstrip())
    if arg:
        dev = torch.device(arg)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        dev = torch.device("xpu")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    if not _hardware_reported:
        _hardware_reported = True
        line, why = hardware_report(torch, dev, nvidia_gpus() if dev.type == "cpu" else [])
        log(line)
        if why:
            print(why, file=sys.stderr, flush=True)
    return dev


# Model presets. musiq/topiq_nr/clipiqa/liqe/nima: a diverse, reasonably light set that works on a CPU. The GPU set adds
# heavier models. "all" is every pyiqa no-reference model except those that are not per-photo quality scorers.
CPU_MODELS = ["musiq", "topiq_nr", "clipiqa", "liqe", "nima"]
GPU_MODELS = CPU_MODELS + ["arniqa", "hyperiqa", "dbcnn"]   # maniqa is left out on purpose: ~4.5 s per photo on an RTX 2060
NOT_PER_IMAGE = {"fid", "fid_dinov2", "sfid", "inception_score", "clipscore"}  # compare image sets / need text
DEFAULT_MAX_SIDE = 1024   # one size for every preset and device, so scores from different runs are comparable


def models_for_preset(preset):
    avail = available_models()
    if preset == "all":
        return [m for m in avail if m not in NOT_PER_IMAGE]
    wanted = CPU_MODELS if preset == "cpu" else GPU_MODELS
    missing = [m for m in wanted if m not in avail]
    if missing:
        print(f"NOTE: not in this pyiqa version, skipped: {', '.join(missing)}", file=sys.stderr)
    return [m for m in wanted if m in avail]


def _tensor(x):
    import torch

    if torch.is_tensor(x):
        return x
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(x, attr, None)
        if v is not None:
            return v
    raise RuntimeError("unexpected CLIP output type")


class Clip:
    def __init__(self, name, dev):
        from transformers import CLIPModel, CLIPProcessor

        self.dev = dev
        self.model = CLIPModel.from_pretrained(name).to(dev).eval()
        self.proc = CLIPProcessor.from_pretrained(name)

    def image(self, img):
        import torch

        with torch.no_grad():
            f = _tensor(self.model.get_image_features(**self.proc(images=img, return_tensors="pt").to(self.dev)))
        return (f / f.norm(dim=-1, keepdim=True))[0].cpu().tolist()

    def text(self, prompts):
        import torch

        with torch.no_grad():
            inp = self.proc(text=prompts, return_tensors="pt", padding=True, truncation=True).to(self.dev)
            f = _tensor(self.model.get_text_features(**inp))
        return (f / f.norm(dim=-1, keepdim=True)).cpu().tolist()


class Faces:
    """MTCNN detector + FaceNet (InceptionResnetV1, VGGFace2) embeddings from facenet-pytorch."""

    def __init__(self, dev):
        from facenet_pytorch import InceptionResnetV1, MTCNN
        from facenet_pytorch.models.utils.detect_face import extract_face

        self.dev = dev
        self.extract = extract_face
        self.mtcnn = MTCNN(keep_all=True, device=dev)
        self.net = InceptionResnetV1(pretrained="vggface2").eval().to(dev)

    def find(self, img, min_face, min_prob=0.9):
        import torch

        boxes, probs = self.mtcnn.detect(img)
        if boxes is None:
            return []
        W, H = img.size
        crops, kept = [], []
        for b, pr in zip(boxes, probs):
            if pr is None or pr < min_prob or min(b[2] - b[0], b[3] - b[1]) < min_face:
                continue
            crops.append((self.extract(img, b, 160, 20).float() - 127.5) / 128.0)
            kept.append((b, float(pr)))
        if not crops:
            return []
        with torch.no_grad():
            e = self.net(torch.stack(crops).to(self.dev))
        e = (e / e.norm(dim=1, keepdim=True)).cpu().tolist()
        return [{"box": [round(b[0] / W, 4), round(b[1] / H, 4), round(b[2] / W, 4), round(b[3] / H, 4)],
                 "prob": round(pr, 3), "emb": [round(x, 3) for x in v]}
                for (b, pr), v in zip(kept, e)]


# ---- single ground truth for faces: the _faces/ folder (resolve_faces_dir) --------------------------------------
#   _faces/<Name>/      confirmed faces of that person: crops you moved there, and/or reference photos you dropped in
#   _faces/_<anything>/ "not a known person" (junk detections); never named
#   _faces/person_NNNN/   every not-yet-known face shows up here, waiting for you to name it (rename the folder)
#   _faces/.crops.json  the face embedding behind every crop and reference photo, i.e. the ground truth itself
# One _faces/ folder serves every photo folder you ever process.
CROP_RE = re.compile(r"^([0-9a-f]{6})_c(\d+)_([0-9a-f]{8})_(\d+)__(.*)\.jpe?g$", re.IGNORECASE)  # <id>_c0007_<hash>_<i>__<stem>.jpg
OLD_CROP_RE = re.compile(r"^c\d+_.*\.jpe?g$", re.IGNORECASE)  # crops from earlier versions
UNNAMED_RE = re.compile(r"^(person|cluster)_(?:[0-9a-f]{6}_)?\d+$", re.IGNORECASE)
RECORDS_FILE = ".crops.json"
SAME_FACE = 0.9  # cosine similarity above which two embeddings count as the same face


def user_data_faces_dir() -> Path:
    """%LOCALAPPDATA%\\imageautorater\\_faces on Windows, ~/.local/share/imageautorater/_faces elsewhere."""
    return user_data_dir() / "_faces"


def resolve_faces_dir(photo_folder: Path, explicit: "Path | None" = None):
    """Find the single ground-truth _faces/ folder for a scan. Returns (folder, how, others) where `others` are other
    _faces folders that exist but are NOT used (so a split ground truth never goes unnoticed).

    Order of preference:
      1. --faces-dir                         a custom folder you name
      2. $IMAGEAUTORATER_FACES               a custom folder for all scans
      3. <photo folder>/_faces               INSIDE the root of the folder you scan (the primary choice), next to its
                                             year/album folders: scanning D:\\Photos uses D:\\Photos\\_faces
      4. <parent>/_faces, then further up    next to the folder you scan, and above it (up to 4 levels), so one _faces
                                             can serve D:\\Photos\\2024, D:\\Photos\\2025, ... from D:\\Photos\\_faces
      5. <repository>/_faces                 when running from a source checkout (git-ignored)
      6. the per-user data folder
    From 3 on, the first folder that already exists is used (nearest first). If none exists yet, it is created inside
    the photo folder (3), or in the per-user data folder (6) when that location is not writable."""
    if explicit:
        return explicit, "--faces-dir", []
    env = os.environ.get("IMAGEAUTORATER_FACES")
    if env:
        return Path(env).expanduser(), "$IMAGEAUTORATER_FACES", []
    folder = photo_folder.resolve()
    candidates = []
    if folder.name != "_faces":   # scanning a _faces folder itself makes no sense as a default
        candidates.append((folder / "_faces", "inside the photo folder"))
    ups = [u for u in folder.parents if u != u.parent][:4]    # not the drive / share root
    for i, up in enumerate(ups):
        candidates.append((up / "_faces", "next to the photo folder" if i == 0 else "in a parent folder"))
    repo = Path(__file__).resolve().parents[2]
    if (repo / "pyproject.toml").is_file() and (repo / "src" / "imageautorater").is_dir():
        candidates.append((repo / "_faces", "in the source checkout"))
    candidates.append((user_data_faces_dir(), "per-user data folder"))
    existing = [c for c in candidates if c[0].is_dir() and c[0] != folder]   # never the folder being scanned itself
    if existing:
        return existing[0][0], existing[0][1], existing[1:]
    if folder.name != "_faces" and os.access(folder, os.W_OK):
        return folder / "_faces", "inside the photo folder (new)", []
    return candidates[-1][0], candidates[-1][1] + " (new)", []


def folder_id(folder: Path) -> str:
    import hashlib

    return hashlib.md5(str(folder.resolve()).lower().encode("utf-8")).hexdigest()[:6]


def rel_hash(rel: str) -> str:
    import hashlib

    return hashlib.md5(rel.encode("utf-8")).hexdigest()[:8]


def safe_stem(rel: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", Path(rel).stem)


def load_records(faces_dir: Path) -> dict:
    p = faces_dir / RECORDS_FILE
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"WARNING: {RECORDS_FILE} was unreadable and was set aside as {backup_file(p).name}; "
                  "confirmations made so far are lost", file=sys.stderr)
    return {}


def save_records(faces_dir: Path, records: dict):
    if faces_dir.is_dir():
        try:
            write_text_atomic(faces_dir / RECORDS_FILE, json.dumps(records))
        except OSError as err:
            print(f"WARNING: could not save {RECORDS_FILE} in {faces_dir} ({err})", file=sys.stderr)
            note_problem(f"{RECORDS_FILE} could not be saved in {faces_dir} ({err}): new face crops are not recorded yet",
                         "close whatever has the file open (sync client, antivirus), then run the same command again")


STATE_FILE = ".state.json"


def person_allocator(faces_dir: Path):
    """Numbers for new person_NNNN folders: one counter for the whole _faces/ folder (kept in .state.json), always
    above any person_NNNN folder that exists, so a number is never reused after you rename a folder."""
    state = faces_dir / STATE_FILE
    n = 1
    try:
        n = int(json.loads(state.read_text(encoding="utf-8"))["next_person"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if faces_dir.is_dir():
        for d in faces_dir.iterdir():
            m = re.fullmatch(r"person_(\d+)", d.name, re.IGNORECASE)
            if m:
                n = max(n, int(m.group(1)) + 1)
    box = [n]

    def alloc():
        box[0] += 1
        return box[0] - 1

    def save():
        faces_dir.mkdir(parents=True, exist_ok=True)
        try:
            write_text_atomic(state, json.dumps({"next_person": box[0]}))
        except OSError as err:
            print(f"WARNING: could not save {STATE_FILE} in {faces_dir} ({err})", file=sys.stderr)

    return alloc, save


def named_persons_exist(faces_dir: Path) -> bool:
    return any(r.get("label") and r.get("emb") for r in load_records(faces_dir).values())


def scan_faces_dir(faces_dir: Path, cache, fid: str, get_faces, min_face):
    """Read the ground-truth folder.

    * A crop (or reference photo) inside _faces/<Name>/ confirms that face as <Name>.
    * Inside _faces/_something/ it means 'not a known person'. Inside person_NNNN it carries no label.
    The embedding of every face is stored in .crops.json, so a confirmation is about the face itself: it does not
    depend on file names, photo order, face indexes, the cache, or which photo folder is being processed. For the
    photo folder being processed, confirmations are bound to today's faces by content (embedding similarity),
    so an edited, renamed or replaced photo can never pick up someone else's label.
    Returns (records, crop count per cluster, folder per cluster, existing crop names, highest cluster id of this
    photo folder, confirmed [(name, embedding)])."""
    import numpy as np

    records = load_records(faces_dir)
    have, folder_of, existing, max_cid = {}, {}, set(), 0
    seen_refs, old_crops, new_refs = set(), 0, 0
    if faces_dir.is_dir():
        for f in sorted(faces_dir.rglob("*")):
            parts = f.relative_to(faces_dir).parts
            if not f.is_file() or len(parts) < 2 or parts[0].startswith("."):
                continue
            top = parts[0]
            unnamed = bool(UNNAMED_RE.match(top))
            label = None if unnamed else ("" if top.startswith("_") else top)
            m = CROP_RE.match(f.name)
            if m:
                rec = records.get(f.name)
                if rec is None:
                    continue  # a crop we hold no record of cannot be verified
                rec["label"] = label
                if rec.get("folder") == fid:
                    cid = int(m.group(2))
                    max_cid = max(max_cid, cid)
                    existing.add(f.name)
                    have[cid] = have.get(cid, 0) + 1
                    folder_of.setdefault(cid, faces_dir / top)
            elif OLD_CROP_RE.match(f.name) and not label:
                old_crops += 1  # old crop in an unnamed or _ folder: nothing to learn from it
            elif f.suffix.lower() in IMG_EXT and label:
                # A photo in _faces/<Name>/ with no crop record: a reference photo you dropped in, or a crop from an
                # earlier version. Its largest face is embedded once and stored, then confirmed as <Name>.
                key = "ref:" + "/".join(parts)
                fp = fingerprint(f)
                rec = records.get(key)
                if not rec or rec.get("fp") != fp:
                    fc = get_faces()
                    if fc is None:
                        continue
                    found = fc.find(load_image(f, 1600), min_face)
                    best = max(found, key=lambda x: (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]), default=None)
                    if best is None:
                        print(f"  reference photo without a usable face: {'/'.join(parts)}", file=sys.stderr)
                    rec = records[key] = {"ref": True, "fp": fp, "emb": best["emb"] if best else None}
                    new_refs += 1
                rec["label"] = label
                seen_refs.add(key)
    for key in [k for k, r in records.items() if r.get("ref") and k not in seen_refs]:
        del records[key]  # reference photo was removed
    if new_refs:
        print(f"Learned {new_refs} new reference face(s) from photos in the named folders of {faces_dir} "
              "(everything in a named folder counts as that person, so remove wrong ones)")
    if old_crops:
        print(f"NOTE: {old_crops} old-version crops in unnamed/_ folders of {faces_dir} are ignored.", file=sys.stderr)

    labels, confirmed = {}, []
    for rec in records.values():
        lab = rec.get("label")
        if lab is None or not rec.get("emb"):
            continue
        if lab:
            confirmed.append((lab, rec["emb"]))
        faces = cache["faces"].get(rec.get("rel")) if rec.get("folder") == fid else None
        if faces:
            e = np.array(rec["emb"], dtype=float)
            e /= np.linalg.norm(e) or 1.0
            sims = [float(np.array(fc["emb"], dtype=float) @ e / (np.linalg.norm(fc["emb"]) or 1.0)) for fc in faces]
            j = int(np.argmax(sims))
            if sims[j] >= SAME_FACE:
                labels[f"{rec['rel']}#{j}"] = lab or None
    cache["face_labels"] = labels
    return records, have, folder_of, existing, max_cid, confirmed


def build_exemplars(confirmed):
    """name -> array of embeddings: every face you confirmed, whichever photo folder it came from."""
    import numpy as np

    ex = {}
    for name, emb in confirmed:
        ex.setdefault(name, []).append(emb)
    return {n: np.array(v, dtype=float) for n, v in ex.items()}


def match_person(emb, exemplars, threshold, margin):
    """Nearest-exemplar match. Returns (name or None, best similarity). A face is only named when the best person
    clears `threshold` AND beats the runner-up person by `margin`; ambiguous faces stay unnamed."""
    import numpy as np

    if not exemplars:
        return None, None
    v = np.array(emb, dtype=float)
    v /= np.linalg.norm(v) or 1.0
    ranked = sorted(((float((E @ v).max()), n) for n, E in exemplars.items()), reverse=True)
    s, n = ranked[0]
    s2 = ranked[1][0] if len(ranked) > 1 else -1.0
    return (n if s >= threshold and s - s2 >= margin else None), s


def identify(faces, rel_key, face_labels, exemplars, threshold, margin):
    """Name the faces of one image: a folder placement you made wins, otherwise nearest-exemplar matching."""
    res = []
    for i, f in enumerate(faces):
        key = f"{rel_key}#{i}"
        if key in face_labels:
            name, sim, src = face_labels[key], None, "folder"
        else:
            name, sim = match_person(f["emb"], exemplars, threshold, margin)
            src = "match" if name else None
        res.append({"name": name, "similarity": None if sim is None else round(sim, 3), "source": src,
                    "cluster": f.get("cluster"), "box": f["box"]})
    return res


def assign_clusters(cache, threshold, exemplars, face_threshold, margin):
    """Group faces that match no known person into unnamed clusters for review. Cluster ids are stable.
    Faces that match a known person, or that you labelled, are not clustered (cluster 0 = matched, -1 = not a
    person). A face sitting in a cluster you have named is re-checked and leaves it unless it truly matches.
    Returns {cluster id: face count} for the unnamed clusters."""
    import numpy as np

    labels = cache["face_labels"]
    named_clusters = set()
    for key, name in labels.items():
        rel, _, i = key.rpartition("#")
        faces = cache["faces"].get(rel)
        if name and faces and int(i) < len(faces) and (faces[int(i)].get("cluster") or 0) > 0:
            named_clusters.add(faces[int(i)]["cluster"])

    sums, counts, pending = {}, {}, []
    for rel in sorted(cache["faces"]):
        for i, f in enumerate(cache["faces"][rel]):
            key, c = f"{rel}#{i}", f.get("cluster")
            if key in labels:
                if labels[key] is None:
                    f["cluster"] = -1
                continue
            name, _ = match_person(f["emb"], exemplars, face_threshold, margin)
            if name:
                f["cluster"] = 0
            elif c is None or c <= 0 or c in named_clusters:
                f["cluster"] = None
                pending.append(f)
            else:
                v = np.array(f["emb"], dtype=float)
                sums[c] = sums.get(c, 0) + v
                counts[c] = counts.get(c, 0) + 1

    nxt = cache["clusters_next"]
    for f in sorted(pending, key=lambda f: -f["prob"]):  # confident faces seed clusters first
        v = np.array(f["emb"], dtype=float)
        v /= np.linalg.norm(v) or 1.0
        cid = None
        if sums:
            ids = list(sums)
            C = np.array([sums[i] / (np.linalg.norm(sums[i]) or 1.0) for i in ids])
            sims = C @ v
            j = int(sims.argmax())
            if sims[j] >= threshold:
                cid = ids[j]
        if cid is None:
            cid, nxt = nxt, nxt + 1
        f["cluster"] = cid
        sums[cid] = sums.get(cid, 0) + v
        counts[cid] = counts.get(cid, 0) + 1
    cache["clusters_next"] = nxt
    return counts


def export_crops(faces_dir: Path, fid, cache, records, by_rel, counts, have, folder_of, existing, per_cluster, min_cluster,
                 max_side):
    """Write sample crops of each *unnamed* cluster into _faces/person_NNNN/ for you to name. Folders you have
    named are never added to, so nothing unconfirmed can end up in them. Folder numbers come from one counter for
    the whole _faces/ folder, so they are unique across photo folders and never reused after you rename a folder."""
    alloc, save_counter = person_allocator(faces_dir)
    cands = {}
    for k, faces in cache["faces"].items():
        if k not in by_rel:
            continue
        for i, f in enumerate(faces):
            cid = f.get("cluster")
            if cid and cid > 0 and counts.get(cid, 0) >= min_cluster:
                area = (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1])
                cands.setdefault(cid, []).append((f["prob"] * area, k, i))
    made = 0
    for cid, lst in sorted(cands.items()):
        if have.get(cid, 0) >= per_cluster:
            continue
        dest = folder_of.get(cid)  # None: this group has no folder yet; one is created with its first crop
        if dest is not None and not UNNAMED_RE.match(dest.name):
            continue  # its crops now live in a folder you named
        for _, k, i in sorted(lst, reverse=True):
            if have.get(cid, 0) >= per_cluster:
                break
            fname = f"{fid}_c{cid:04d}_{rel_hash(k)}_{i}__{safe_stem(k)}.jpg"
            if fname in existing:
                continue
            try:
                img = load_image(by_rel[k], max_side)
                x0, y0, x1, y1 = cache["faces"][k][i]["box"]
                W, H = img.size
                pw, ph = (x1 - x0) * 0.3, (y1 - y0) * 0.3
                crop = img.crop((int(max(0, x0 - pw) * W), int(max(0, y0 - ph) * H),
                                 int(min(1, x1 + pw) * W), int(min(1, y1 + ph) * H)))
                crop.thumbnail((220, 220))
                if dest is None:
                    dest = folder_of[cid] = faces_dir / f"person_{alloc():04d}"
                dest.mkdir(parents=True, exist_ok=True)
                crop.save(dest / fname, quality=90)
                records[fname] = {"folder": fid, "rel": k, "emb": cache["faces"][k][i]["emb"], "cid": cid, "label": None}
                existing.add(fname)
                have[cid] = have.get(cid, 0) + 1
                made += 1
            except Exception as e:
                print(f"    crop failed for {k}: {str(e)[:100]}", file=sys.stderr)
    if made:
        save_counter()
    return made


def pick_tags(emb, labels, text_feats, threshold, max_tags):
    import numpy as np

    logits = 100.0 * (np.array(text_feats, dtype=float) @ np.array(emb, dtype=float))
    p = np.exp(logits - logits.max())
    p /= p.sum()
    order = p.argsort()[::-1][:max_tags]
    names = list(labels)
    chosen = [(names[i], float(p[i])) for n, i in enumerate(order) if n == 0 or p[i] >= threshold]  # always the top label
    return {k: round(v, 3) for k, v in chosen}


def run_content(folder, paths, rel, cache, cache_path, a, faces_dir):
    """Fill cache['clip'] / cache['faces'] for new images; return {relpath: {'tags':..., 'people':...}}.
    Faces are detected and clustered even with no names known; names arrive later through the single ground truth
    in faces_dir (crops moved into named folders, reference photos dropped in)."""
    import hashlib

    dev = pick_device(a.device)
    labels = load_labels(a.labels)  # also needed when not classifying, to keep earlier keywords in step
    state = {"clip": None, "faces": None, "clip_ok": a.classify, "faces_ok": bool(a.faces)}

    def get_clip():
        if state["clip"] is None and state["clip_ok"]:
            try:
                log(f"Loading the CLIP model {a.clip_model} for keywords (the first use downloads it, about 600 MB) ...")
                state["clip"] = Clip(a.clip_model, dev)
                log("CLIP model ready")
            except Exception as e:
                print(f"WARNING: CLIP unavailable, skipping keywords ({str(e)[:150]}). uv pip install transformers", file=sys.stderr)
                note_problem(f"image-type keywords were skipped (CLIP model unavailable: {str(e)[:100]})", "uv pip install transformers")
                state["clip_ok"] = False
        return state["clip"]

    def get_faces():
        if state["faces"] is None and state["faces_ok"]:
            try:
                log("Loading the face detection and recognition models (the first use downloads their weights) ...")
                state["faces"] = Faces(dev)
                log("Face models ready")
            except Exception as e:
                print(f"WARNING: face model unavailable, skipping people ({str(e)[:150]}). "
                      "Install it with: uv pip install --no-deps facenet-pytorch", file=sys.stderr)
                note_problem(f"faces were NOT detected: the face model is unavailable ({str(e)[:100]})",
                             "uv pip install --no-deps facenet-pytorch   (no-deps: it pins an old torch)")
                state["faces_ok"] = False
        return state["faces"]

    if a.classify and cache["clip_model"] != a.clip_model:
        cache["clip"], cache["clip_text"], cache["clip_model"] = {}, None, a.clip_model

    fid = folder_id(folder)

    todo = [p for p in paths if rel[p] not in cache["unreadable"]
            and ((state["clip_ok"] and rel[p] not in cache["clip"]) or (state["faces_ok"] and rel[p] not in cache["faces"]))]
    t0, last_save = time.time(), time.time()
    for n, (p, img, load_err) in enumerate(prefetched(todo, lambda q: load_image(q, a.content_max_side), a.io_threads), 1):
        status(f"[content] {n}/{len(todo)}  {fmt_eta(time.time() - t0, n, len(todo))}  {p.name}")
        if load_err and is_damaged_image(load_err):
            mark_unreadable(cache, rel[p], load_err)
            continue
        try:
            if load_err:
                raise load_err
            need_clip = state["clip_ok"] and rel[p] not in cache["clip"]
            need_faces = state["faces_ok"] and rel[p] not in cache["faces"]
            if need_clip and get_clip():
                cache["clip"][rel[p]] = [round(x, 3) for x in state["clip"].image(img)]
            if need_faces and get_faces():
                cache["faces"][rel[p]] = state["faces"].find(img, a.min_face)
        except Exception as e:
            print(f"    content failed on {p.name}: {str(e)[:120]}", file=sys.stderr)
        if time.time() - last_save > 60:
            save_cache(cache_path, cache)
            last_save = time.time()
    status(None)
    if todo:
        print(f"[content] analysed {len(todo)} images in {time.time() - t0:.0f}s")
        save_cache(cache_path, cache)

    exemplars = {}
    if state["faces_ok"]:
        # the _faces/ folder is the single ground truth; every other face is matched against it
        records, have, folder_of, existing, max_cid, confirmed = scan_faces_dir(faces_dir, cache, fid, get_faces, a.min_face)
        cache["clusters_next"] = max(cache["clusters_next"], max_cid + 1)  # never reuse an id that has crops on disk
        exemplars = build_exemplars(confirmed)
        counts = assign_clusters(cache, a.cluster_threshold, exemplars, a.face_threshold, a.face_margin)
        by_rel = {rel[p]: p for p in paths}
        made = export_crops(faces_dir, fid, cache, records, by_rel, counts, have, folder_of, existing,
                            a.crops_per_cluster, a.min_cluster, a.content_max_side)
        save_records(faces_dir, records)
        save_cache(cache_path, cache)
        review = sum(1 for n in counts.values() if n >= a.min_cluster)
        print(f"People: {len(exemplars)} known ({len(confirmed)} confirmed faces in {faces_dir}); "
              f"{review} unnamed groups to review" + (f" ({made} new crops)" if made else ""))
        if review:
            print(f"  To name people: rename the person_NNNN folders in {faces_dir} to the person's name "
                  "(or move crops into an existing person's folder), then re-run.")

    # Keywords come from the cached CLIP data, so a run that does not classify (e.g. --faces-only) still reports the
    # same keywords and does not strip them from the sidecars. No model is loaded unless the vocabulary changed.
    key = hashlib.md5(json.dumps([a.clip_model, labels], sort_keys=True).encode()).hexdigest()
    if state["clip_ok"] and (not cache["clip_text"] or cache["clip_text"]["key"] != key):
        if get_clip():
            cache["clip_text"] = {"key": key, "feats": [[round(x, 3) for x in v] for v in state["clip"].text(list(labels.values()))]}
            save_cache(cache_path, cache)
    text_feats = cache["clip_text"]["feats"] if cache["clip_text"] and cache["clip_text"]["key"] == key else None

    content = {}
    for p in paths:
        k = rel[p]
        entry = {}
        if text_feats and k in cache["clip"]:
            entry["tags"] = pick_tags(cache["clip"][k], labels, text_feats, a.tag_threshold, a.max_tags)
        if state["faces_ok"] and k in cache["faces"]:
            entry["people"] = identify(cache["faces"][k], k, cache["face_labels"], exemplars, a.face_threshold, a.face_margin)
        if entry:
            content[k] = entry
    return content


def first_run_without_persons(cache, faces_dir):
    """True when no person is known yet (nothing confirmed in the _faces folder) and this photo folder has no
    analysed faces."""
    if cache["faces"]:
        return False  # faces already detected on an earlier run: just carry on
    return not named_persons_exist(faces_dir)


def ask_faces_first(faces_dir):
    """Returns True for faces-only (the default), False to proceed without face recognition."""
    print(f"\nNo persons are known yet (nothing confirmed in {faces_dir}).")
    print("  [F] Run --faces-only first: detect and group the faces, then you name them (default)")
    print("  [P] Proceed without face recognition (quality scores and keywords only)")
    if not sys.stdin or not sys.stdin.isatty():
        print("Not interactive: using --faces-only. Pass --no-faces to skip faces instead.")
        return True
    while True:
        ans = input("Choice [F/p]: ").strip().lower()
        if ans in ("", "f"):
            return True
        if ans == "p":
            return False
        if ans == "q":
            sys.exit("Aborted.")
        print("Please answer F or P (Q to quit).")


def _main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--preset", choices=["auto", "cpu", "gpu", "all"], default="auto",
                    help=f"model set when --models is not given. auto (default): gpu if a GPU is available, else cpu. "
                         f"cpu = {' '.join(CPU_MODELS)}; gpu = cpu + {' '.join(GPU_MODELS[len(CPU_MODELS):])}; "
                         f"all = every NR model (very slow without a GPU). All presets score at --max-side {DEFAULT_MAX_SIDE}")
    ap.add_argument("--models", nargs="+", help="only score with these pyiqa models (overrides --preset); "
                    "ratings always combine every model already in the cache")
    ap.add_argument("--exclude", nargs="+", default=[], help="models to skip")
    ap.add_argument("--mode", choices=["absolute", "percentile"], default="absolute",
                    help="absolute = map onto each model's score range (default); percentile = rank within folder")
    ap.add_argument("--recursive", "-r", action=argparse.BooleanOptionalAction, default=True,
                    help="descend into subfolders (default); --no-recursive for the top folder only")
    ap.add_argument("--limit", type=int, metavar="N", help="only process the first N images (sorted by path), e.g. for a quick test")
    ap.add_argument("--raw", action="store_true", help="also rate RAW files (needs rawpy)")
    ap.add_argument("--device", help="cuda / cpu (default: auto)")
    ap.add_argument("--io-threads", type=int, default=4, metavar="N",
                    help="photos decoded in parallel in the background while a model works (default 4; the GPU no longer "
                         "waits for the disk, which helps most on a network share)")
    ap.add_argument("--fix-gpu", action=argparse.BooleanOptionalAction, default=None,
                    help="with an NVIDIA GPU but a CPU-only PyTorch: install the CUDA build into this environment before "
                         "running. Default: ask when run in a terminal; --no-fix-gpu never offers it")
    ap.add_argument("--max-side", type=int, default=None,
                    help=f"downscale the long side before scoring; 0 = full size (default {DEFAULT_MAX_SIDE} for every preset and device)")
    ap.add_argument("--json-out", type=Path, help=f"combined results JSON (default: <folder>/{RESULTS_NAME})")
    ap.add_argument("--fresh", action="store_true", help="start a new score cache (the old one is kept as .bak)")
    ap.add_argument("--iqa", action=argparse.BooleanOptionalAction, default=True,
                    help="quality scoring with pyiqa (default on; --no-iqa for keywords/people only)")
    ap.add_argument("--classify", action=argparse.BooleanOptionalAction, default=True,
                    help="image-type keywords via CLIP (default on)")
    ap.add_argument("--labels", type=Path, help="keyword vocabulary file: one 'label' or 'label: prompt' per line")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32",
                    help="HuggingFace CLIP model (e.g. openai/clip-vit-large-patch14 is better but slower)")
    ap.add_argument("--tag-threshold", type=float, default=0.12, help="min probability for extra keywords (top label is always kept)")
    ap.add_argument("--max-tags", type=int, default=4, help="max image-type keywords per photo")
    ap.add_argument("--faces", action=argparse.BooleanOptionalAction, default=None,
                    help="face detection + person identification. Unset: on, but the first run with no known persons "
                         "asks whether to do --faces-only or proceed without faces")
    ap.add_argument("--faces-dir", type=Path,
                    help="a custom ground-truth folder for faces. Default: a '_faces' folder in the ROOT of the photo folder "
                         "(D:\\Photos -> D:\\Photos\\_faces), else next to it or further up, then $IMAGEAUTORATER_FACES, a "
                         "source checkout's _faces/, the per-user data folder; the first that exists wins (see README)")
    ap.add_argument("--face-threshold", type=float, default=0.55, help="min cosine similarity to accept a match; raise if you get wrong names")
    ap.add_argument("--face-margin", type=float, default=0.05,
                    help="a face is only named if the best person beats the runner-up by this much (avoids look-alike mix-ups)")
    ap.add_argument("--min-face", type=int, default=48, help="ignore faces smaller than this many pixels")
    ap.add_argument("--faces-only", action="store_true", help="shortcut for --no-iqa --no-classify: only detect, cluster and name faces")
    ap.add_argument("--reset-faces", action="store_true",
                    help="forget this photo folder's detected faces and groups in its cache (quality scores and keywords "
                         "are kept; the _faces/ ground truth is never touched)")
    ap.add_argument("--cluster-threshold", type=float, default=0.6,
                    help="cosine similarity to join an existing face cluster; raise to split more, lower to merge more")
    ap.add_argument("--crops-per-cluster", type=int, default=8, help="sample crops exported per cluster")
    ap.add_argument("--min-cluster", type=int, default=2,
                    help="only create a person_NNNN folder for groups with at least this many faces (default 2; 1 = every unknown face shows up)")
    ap.add_argument("--content-max-side", type=int, default=1600, help="image size used for keywords/faces")
    a = ap.parse_args()
    if a.faces_only:
        a.iqa = a.classify = False
        a.faces = True

    from imageautorater import __version__

    global _T0
    _T0 = time.time()   # the [mm:ss] stamps count from here
    PROBLEMS.clear()
    global _cache_error
    _cache_error = None
    _ACTIVE.clear()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # Windows without developer mode: harmless, very noisy
    import warnings
    warnings.filterwarnings("ignore", message=".*torch.jit.load.*")
    warnings.filterwarnings("ignore", message=".*Corrupt EXIF data.*")   # a damaged EXIF block; the picture still loads
    log(f"imageautorater {__version__}: nothing is installed unless you agree to the GPU fix. Normal reasons to wait: reading "
        "a big folder, loading PyTorch, and (first run only) downloading model weights, which are cached in your user profile.")
    if a.iqa or a.classify or a.faces is not False:
        maybe_fix_gpu(a.fix_gpu, a.device)   # before anything imports PyTorch
    if not a.folder.is_dir():
        sys.exit(f"Not a folder: {a.folder}")
    faces_dir, faces_how, faces_others = resolve_faces_dir(a.folder, a.faces_dir)   # one ground truth for all scans

    def walking(seen, found):
        if sys.stdout.isatty() or seen % 5000 == 0:   # a redirected log gets a line every 5000 entries, not 10x as many
            status(f"[reading] {seen} entries looked at, {found} photos found so far ...")

    log(f"Reading {a.folder}" + (" (including subfolders)" if a.recursive else "") + " to list the photos ...")
    t_scan = time.time()
    paths = find_images(a.folder, a.recursive, a.raw, exclude=[faces_dir], progress=walking)
    status(None)
    if not paths:
        sys.exit("No images found.")
    log(f"Found {len(paths)} photos in {time.time() - t_scan:.0f}s")
    if a.limit:
        paths = paths[:a.limit]
        log(f"--limit: using the first {len(paths)} photos (sorted by path)")

    models = []
    if a.iqa:
        dev = pick_device(a.device)
        preset = a.preset if a.preset != "auto" else ("cpu" if dev.type == "cpu" else "gpu")
        if a.max_side is None:
            a.max_side = DEFAULT_MAX_SIDE
        models = a.models or models_for_preset(preset)
        models = [m for m in models if m not in set(a.exclude)]
        print(f"Preset: {preset} ({'--models given' if a.models else 'auto-selected for ' + str(dev)}), "
              f"{len(models)} models, max-side {a.max_side}")
    elif a.max_side is None:
        a.max_side = DEFAULT_MAX_SIDE
    cache_path = a.folder / CACHE_NAME
    if a.fresh and cache_path.exists():
        print(f"--fresh: previous cache kept as {backup_file(cache_path).name}")
    cache = load_cache(cache_path)
    _ACTIVE.update(path=cache_path, cache=cache)
    if a.reset_faces:
        cache["faces"], cache["face_labels"], cache["clusters_next"] = {}, {}, 1
        print("--reset-faces: detected faces and groups cleared from this photo folder's cache")

    # the cache is appended to, never rebuilt: only new/changed files and new models get scored
    rel = {p: str(p.relative_to(a.folder)).replace("\\", "/") for p in paths}
    changed = new_files = 0
    log(f"Checking {len(paths)} photos against the cache (size and modified time of each file) ...")
    for n, p in enumerate(paths, 1):
        if n % 200 == 0 and (sys.stdout.isatty() or n % 5000 == 0):
            status(f"[checking] {n}/{len(paths)} photos")
        fp = fingerprint(p)
        old = cache["files"].get(rel[p])
        if old is None:
            new_files += 1
        if old is not None and old != fp:  # image was modified: its old scores no longer apply
            changed += 1
            for sc in cache["scores"].values():
                sc.pop(rel[p], None)
            cache["clip"].pop(rel[p], None)
            cache["faces"].pop(rel[p], None)
            cache["unreadable"].pop(rel[p], None)
            for key in [k for k in cache["face_labels"] if k.rpartition("#")[0] == rel[p]]:
                del cache["face_labels"][key]  # face indexes of a modified image no longer line up
        cache["files"][rel[p]] = fp
    status(None)
    log(f"Cache check done: {new_files} new to this cache, {changed} modified, {len(paths) - new_files - changed} unchanged"
        + (" (first run for this folder: every photo is analysed, which takes a long time; progress per model follows)"
           if new_files == len(paths) else ""))

    if a.faces is not False:
        log(f"Faces folder: {faces_dir} ({faces_how})")
    if a.faces is None:  # not decided on the command line
        a.faces = True
        if first_run_without_persons(cache, faces_dir):
            if ask_faces_first(faces_dir):
                a.iqa = a.classify = False  # faces-only
            else:
                a.faces = False
    if a.faces:
        for other, how in faces_others:
            print(f"  NOTE: another _faces folder exists and is NOT used: {other} ({how}). "
                  "Move its named folders into the one above if they hold people you have named.", file=sys.stderr)

    if a.iqa:
        if cache["max_side"] is not None and cache["max_side"] != a.max_side and cache["scores"]:
            print(f"WARNING: cache was scored at --max-side {cache['max_side']}, now {a.max_side}; "
                  "new scores are mixed with old ones (use --fresh to start over)", file=sys.stderr)
        cache["max_side"] = a.max_side
        log(f"Scoring quality: {len(models)} models ({', '.join(models)}) on {len(paths)} photos. "
            "Progress and time left are shown per model; you can stop with Ctrl+C and continue later.")
        run_models(a.folder, paths, models, a.device, a.max_side, cache, cache_path, io_threads=a.io_threads)
        save_cache(cache_path, cache)

    # content analysis: keywords + people
    content = {}
    if a.classify or a.faces:
        log("Analysing " + " and ".join(x for x, on in (("image types (CLIP keywords)", a.classify), ("faces", a.faces)) if on) + " ...")
        content = run_content(a.folder, paths, rel, cache, cache_path, a, faces_dir)

    # Combine every model in the cache (honouring --exclude), not just this run's --models,
    # so a partial run never shrinks the ratings built up earlier.
    used = [m for m in cache["scores"] if cache["scores"][m] and m in cache["models"] and m not in set(a.exclude)]
    finalize_cache(cache_path, cache)
    if not used and not content:
        report_problems()   # the reasons come first: they explain why nothing was scored
        sys.exit("Nothing was scored.")
    unknown = [m for m in used if cache["models"][m].get("range") is None]
    if unknown:   # ranges stored by an earlier version could not be read: refill them from pyiqa's model table
        ensure_imported("pyiqa", "Looking up the score ranges of the models (pyiqa)")
        for m in unknown:
            cache["models"][m]["range"] = config_range(m)
        fixed = [m for m in unknown if cache["models"][m]["range"]]
        if fixed:
            log(f"Score ranges restored for {', '.join(fixed)}; star ratings use them now (absolute mode)")
        save_cache(cache_path, cache)
    log(f"Combining the ratings of {len(used)} models ...")
    per_image = {}  # relpath -> {model: (stars, score)}
    for m in used:
        sc = cache["scores"][m]
        stars = to_stars(sc, cache["models"][m], a.mode)
        for k, s in sc.items():
            per_image.setdefault(k, {})[m] = (stars[k], s)

    json_path = a.json_out or (a.folder / RESULTS_NAME)
    try:
        write_results_json(json_path, a, per_image, used, cache, content)
    except OSError as err:   # the share refuses the file: keep the results locally instead of losing them
        rec = user_data_dir() / "recovery" / RESULTS_NAME
        rec.parent.mkdir(parents=True, exist_ok=True)
        write_results_json(rec, a, per_image, used, cache, content)
        note_problem(f"{json_path.name} could not be written to {json_path.parent} ({err}); it is in {rec}",
                     "close whatever has the file open, then run the same command again")
        json_path = rec
    finalize_cache(cache_path, cache)
    report_unreadable(cache, set(rel.values()))
    log(f"Wrote combined results ({len(set(per_image) | set(content))} images): {json_path}")

    named = sum(1 for e in content.values() for pr in e.get("people", []) if pr.get("name"))
    took = int(time.time() - _T0)
    log(f"Done in {took // 3600}h{took % 3600 // 60:02d}m{took % 60:02d}s: {len(set(per_image) | set(content))} images, {len(used)} models"
        + (f", {named} faces named" if content else "") + ".")
    report_problems()
    print("Nothing was written into or next to your photos. Next steps:")
    print(f'  imageautorater-write-xmp  "{a.folder}"    (XMP sidecar files)')
    print(f'  imageautorater-write-exif "{a.folder}"    (inside JPEG/PNG files; asks first, --backup DIR available)')
    print(f'  imageautorater-gallery    "{a.folder}"    (browse the ranking)')


def main():
    try:
        _main()
    except KeyboardInterrupt:
        _save_on_exit("Interrupted")
        print("Run the same command again to continue; everything saved so far is kept.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        _save_on_exit("The run stopped with an error")
        raise


def _save_on_exit(why):
    """Ctrl+C or a crash must not throw away what has been scored: write the cache once more."""
    if _ACTIVE:
        status(None)
        ok = save_cache(_ACTIVE["path"], _ACTIVE["cache"])
        print(f"\n{why}. " + ("The progress was saved." if ok else "The progress could NOT be saved (see above)."), file=sys.stderr)


if __name__ == "__main__":
    main()
