"""Thumbnails for the gallery: one flat folder, made in parallel, only the missing ones, safe to interrupt."""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from imageautorater.pipeline import status

THUMB_DIR_NAME = ".iqa_thumbs"


def default_jobs() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def thumb_file(thumb_dir: Path, rel: str, fp: str, size: int) -> Path:
    """Name = path hash + file fingerprint + size, so an edited photo gets a fresh thumbnail and sizes can coexist."""
    from imageautorater.pipeline import rel_hash

    return thumb_dir / f"{rel_hash(rel)}_{rel_hash(fp)}_{size}.jpg"


def make_thumb(src: Path, dst: Path, size: int):
    from imageautorater.pipeline import load_image

    img = load_image(src, size)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")           # an interrupted run never leaves a half-written thumbnail
    img.save(tmp, "JPEG", quality=82, optimize=True)
    os.replace(tmp, dst)


def _job(args):
    src, dst, size = args
    try:
        make_thumb(Path(src), Path(dst), size)
        return None
    except Exception as e:
        return str(e)[:80] or type(e).__name__


def ensure_thumbs(todo: list, size: int, workers: int = 1) -> dict:
    """todo: [(rel, src, dst)] of thumbnails that do not exist yet. Returns {rel: error} for the ones that failed."""
    failed = {}
    if not todo:
        return failed
    t0 = time.time()
    args = [(str(src), str(dst), size) for _, src, dst in todo]
    workers = max(1, min(workers, len(todo)))

    def report(n):
        el = int(time.time() - t0)
        status(f"[{el // 60:02d}:{el % 60:02d}] thumbnails {n}/{len(todo)}")

    if workers == 1:
        results = map(_job, args)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_job, args, chunksize=4)
    try:
        for n, err in enumerate(results, 1):
            if err:
                failed[todo[n - 1][0]] = err
            if n % (20 if sys.stdout.isatty() else 500) == 0 or n == len(todo):
                report(n)
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
        status(None)
    return failed
