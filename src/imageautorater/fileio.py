"""Writing files safely on network shares (NAS / SMB) and in synced folders (Nextcloud, OneDrive, ...).

The usual safe way to update a file is to write a temporary file and rename it over the original. On a share that is
fragile: the rename fails with "Access is denied" (Windows error 5 / 32) while another program holds the target open
(the sync client, an antivirus scanner, the search indexer, another PC), or on shares that allow writing a file but not
renaming over it. Both are often temporary. So: retry with back-off, then (where allowed) overwrite in place, and as a
last resort keep a recovery copy somewhere local, never losing the finished work.
"""
import os
import shutil
import time
from pathlib import Path

RETRY_DELAYS = (0.25, 0.5, 1.0, 2.0, 4.0)   # seconds to wait before attempts 2..6


def user_data_dir() -> Path:
    """%LOCALAPPDATA%\\imageautorater on Windows, ~/.local/share/imageautorater elsewhere."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "imageautorater"


def replace_file(tmp, dst, allow_copy=False, sleep=None):
    """Move `tmp` over `dst`. Returns "replaced" or, after the in-place fallback, "copied". Raises OSError when it
    cannot be done; `tmp` is then left in place (it holds the complete new content).

    allow_copy=True adds the fallback "overwrite dst in place", which works on shares that forbid renaming over a file.
    It is NOT atomic, so use it only for files that can be rebuilt (caches, sidecars), never for photos."""
    last = None
    for delay in (0,) + RETRY_DELAYS:
        if delay:
            (sleep or time.sleep)(delay)   # looked up when called, so it can be replaced (tests) and is never stale
        try:
            os.replace(tmp, dst)
            return "replaced"
        except OSError as err:
            last = err
    if allow_copy:
        try:
            shutil.copyfile(tmp, dst)
        except OSError as err2:
            raise OSError(f"cannot update {dst}: {last}; overwriting it in place failed too: {err2}") from err2
        try:
            os.remove(tmp)
        except OSError:
            pass
        return "copied"
    raise OSError(f"cannot replace {dst}: {last}") from last


def write_text_atomic(path, text, allow_copy=True, encoding="utf-8"):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding=encoding)
    return replace_file(tmp, path, allow_copy=allow_copy)


def write_bytes_atomic(path, data, allow_copy=True):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    return replace_file(tmp, path, allow_copy=allow_copy)


def recovery_copy(name, text) -> Path:
    """Keep `text` in the per-user data folder (always local and writable) when the real target cannot be written."""
    folder = user_data_dir() / "recovery"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    target.write_text(text, encoding="utf-8")
    return target
