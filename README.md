# imageautorater

Rates a folder of photos and saves the results in a JSON file. **The rating itself writes nothing into or next to your photos.** Storing the results in the photos is a separate, deliberate last step, and a third command builds a web page to browse the ranking.

- **Quality:** scored with [pyiqa](https://github.com/chaofengc/IQA-PyTorch) no-reference models, giving 1-5 stars per model.
- **Image type keywords:** CLIP zero-shot classification (portrait, landscape, food, night, ...).
- **People:** faces are detected and grouped, then named by you.
- **Gallery:** `imageautorater-gallery` shows the best photos per folder, sortable by any model.

Commands, in the order you use them:

| Command | Does | Touches your photos? |
|---|---|---|
| `imageautorater-rate` (also `python -m imageautorater`) | scores, keywords, faces; writes `iqa_results.json` | no (read only) |
| `imageautorater-write-xmp` | writes the results into XMP sidecar files (`photo.xmp`) | no |
| `imageautorater-write-exif` | writes the results inside JPEG and PNG files | **yes, metadata only** |
| `imageautorater-gallery` | builds a web page of the ranking | no |

## Install

### 1. One-time setup of a fresh Windows 10 (nothing installed yet)

Open a normal **PowerShell** window (not as administrator; the Git installer asks for approval itself) and run:

```
winget --version
winget install -e --id Git.Git
winget install -e --id astral-sh.uv
winget install -e --id Microsoft.VCRedist.2015+.x64
```

Then **close the window and open a new one**, so the new programs are found. You do not need to install Python: uv fetches the right version itself.

- `Git.Git` is needed to install from the repository, `astral-sh.uv` is the installer, and the Visual C++ runtime is required by PyTorch (without it you get "DLL load failed" errors on a fresh Windows).
- If `winget` is not recognised: update Windows 10 to 22H2, or install **App Installer** from the Microsoft Store, then try again. Windows 10 itself no longer receives regular updates, so keep the machine patched and offline-safe as you see fit.
- You need about 5 GB of free disk space for the program, plus the model downloads on the first run.

### 2. Install the program

Choose a folder that is **not** synced by a cloud client (the environment holds thousands of files), for example `C:\imageautorater`:

```
mkdir C:\imageautorater
cd C:\imageautorater
uv venv --python 3.12
uv pip install "imageautorater @ git+https://github.com/alexschultze/imageautorater" --torch-backend=auto
.venv\Scripts\imageautorater-rate --help
```

If that prints the help text, everything works. A first scan (and the model downloads) then looks like this:

```
.venv\Scripts\imageautorater-rate "D:\Photos\2025" --limit 10
```

`--torch-backend=auto` picks the PyTorch build for your hardware (NVIDIA CUDA, Intel XPU, otherwise CPU). The commands are in the environment: `.venv\Scripts\imageautorater-rate` and so on (or activate it once per window with `.venv\Scripts\activate`, then use the bare command names).

- **No GitHub account or sign-in is needed**: the repository is public.
- **A specific version:** add a tag, branch or commit after the URL: `...imageautorater@v0.1.0`.
- **Update:** repeat the same `uv pip install` command with `--upgrade`.
- **GPU:** an NVIDIA GPU is used automatically; if PyTorch was installed as a CPU-only build, the first run offers to replace it (see *Troubleshooting*).
- **Person identification (required for faces)**: `uv pip install --no-deps facenet-pytorch`. Without it **no faces are detected**: the run continues without them and says so in the ATTENTION block at the end. It has to be installed without dependencies because it pins an old torch; everything else it needs is already installed.
- **RAW files** (optional): `uv pip install rawpy`.

### Development

```
git clone https://github.com/alexschultze/imageautorater
cd imageautorater
uv venv
uv pip install -e ".[faces]" --group dev --torch-backend=auto    # or =cpu on a machine without a GPU
# (the faces extra pins an old torch; if it complains, install it as in the Install section: uv pip install --no-deps facenet-pytorch)
uv run pytest           # fast; uses stand-ins for the models, downloads nothing
uv run imageautorater-rate --help
uv build                # wheel in dist/
```

Layout:

```
pyproject.toml          package metadata, dependencies, entry points, uv settings
uv.lock                 locked development environment
src/imageautorater/
    pipeline.py         rating, keywords, faces (imageautorater-rate); writes only JSON and the faces folder
    metadata.py         XMP building/merging shared by the two writers (no torch)
    write_xmp.py        XMP sidecar files (imageautorater-write-xmp)
    write_exif.py       metadata inside JPEG/PNG files (imageautorater-write-exif)
    gallery.py          web page builder (imageautorater-gallery)
tests/                  pytest
.github/workflows/      ci.yml (tests on push/PR), release.yml (tag v* -> wheel on a GitHub release)
PLAN.md                 plan for the GUI and distribution
_faces/                 your face ground truth in a source checkout (git-ignored, never commit)
```

Releases: bump `version` in `pyproject.toml`, tag `vX.Y.Z`, push the tag. The release gets the wheel attached; installing from the git tag (see above) needs nothing else.

## Run

```
imageautorater-rate "C:\Photos\2025"
```

Subfolders are included by default (`--no-recursive` for the top folder only). Useful options:

| Option | Effect |
|---|---|
| `--preset auto\|cpu\|gpu\|all` | Model set. `auto` (default) picks `gpu` if a GPU is found (CUDA, Intel XPU, Apple MPS), otherwise `cpu` |
| `--models musiq topiq_nr` | Score with exactly these models (overrides `--preset`) |
| `--limit 10` | Only process the first 10 images (quick test) |
| `--max-side 1024` | Downscale the long side before scoring (default 1024 for every preset and device; `0` = full size) |
| `--mode percentile` | Rank stars within the folder instead of using each model's score range |
| `--faces-only` | Only detect, group and name faces |
| `--no-iqa` / `--no-classify` / `--no-faces` | Turn off one part |
| `--labels labels.txt` | Own keyword vocabulary, one `label` or `label: prompt` per line |
| `--device cpu` | Force the device (`cuda`, `xpu` if available) |

`imageautorater-rate --help` lists everything.

## Output of the rating

Written in the photo folder, never next to the individual photos:

- **`iqa_results.json`**: per image the stars and raw score of every model, the mean stars, the keywords (recognised people first, then image types) and the people found. This is what the other commands read.
- **`iqa_scores.json`**: the persistent cache (scores, CLIP and face embeddings).
- **`_faces\`**: crops to name people, by default in a `_faces` folder in the root of the photo folder (see *Naming people*).

## Saving the results into your photos

Run one of these after the rating. Both read `iqa_results.json`, only change what has changed, and can be re-run any time; both have `--dry-run`, `--no-rating`, `--no-keywords` and `--no-model-fields`.

**`imageautorater-write-xmp <folder>`**: sidecar files, the photos stay untouched. `photo.jpg` gets `photo.xmp` (`--darktable-naming` for `photo.jpg.xmp`). Works for every format, including RAW. Existing sidecars are updated in place: develop settings and your own keywords are kept, and unchanged sidecars are not rewritten.

**`imageautorater-write-exif <folder>`**: writes inside the files. **This modifies your photos**, so it asks before it starts (`--yes` skips the question, `--backup DIR` copies each original first, `--dry-run` only reports).

- **JPEG:** EXIF `Rating`, `RatingPercent` and `XPKeywords` (what Windows shows as rating and tags) plus an embedded XMP packet with `xmp:Rating`, the keywords and the per-model scores (what Lightroom, darktable and digiKam read).
- **PNG:** the XMP packet.
- **Everything else** (TIFF, WebP, BMP, HEIC and all RAW formats) is never edited in place. The command lists them; use `imageautorater-write-xmp` for those.

How the files stay safe: the compressed picture data is copied byte for byte (nothing is decoded or re-encoded), the existing EXIF and the camera's maker note keep their exact bytes and positions (only the three tags are added or updated), keywords you added yourself are kept while the ones this tool added earlier are replaced, each file is swapped in atomically with its modified time preserved, files that would not change are not rewritten, and a damaged or unusual file is skipped, never half-written. The pipeline's cache is updated so the next `imageautorater-rate` does not think the photos were edited and re-score them.

Rating, keywords and people only reach the photos through these two commands, so you can review `iqa_results.json` or the gallery first.

## Naming people

There is **one ground truth for faces**: a single `_faces\` folder, shared by every photo folder that finds it, so a person named once is recognised everywhere. The scan reports which folder it uses (`Faces folder: ... (how it was found)`). Where it looks, in this order:

1. **`--faces-dir <folder>`**: a custom folder you name.
2. **`IMAGEAUTORATER_FACES`**: a custom folder for all scans, set as an environment variable.
3. **`_faces` inside the folder you scan (the primary choice)**: in its root, next to its year or album folders. Scanning `D:\Photos` uses `D:\Photos\_faces`. If nothing exists yet, it is created there.
4. **`_faces` next to the scanned folder, then further up** (up to 4 levels): scanning `D:\Photos\2025` finds `D:\Photos\_faces`, so one `_faces` serves `2024`, `2025`, ... and everything below them. The nearest one wins.
5. **`_faces` in the repository**, when you run from a source checkout (git-ignored).
6. **The per-user data folder**: `%LOCALAPPDATA%\imageautorater\_faces` on Windows, `~/.local/share/imageautorater/_faces` elsewhere. Also used when the scanned folder cannot be written to.

For 3 to 6 the first folder that already exists is used, and any other `_faces` folder that exists is reported as *not used*, so a split ground truth never goes unnoticed. The scan prints `Faces folder: ... (how it was found)`, and the first-run question names the folder it looked in. Photos are never searched in a `_faces` folder (at any depth) or in a hidden folder such as the gallery's `.iqa_thumbs`, so the crops are never rated as photos. A custom faces folder inside the photo folder is excluded too.

```
_faces\
  Alice\                  confirmed faces of Alice: crops you moved here, and/or reference photos you dropped in
  Bob\
  _junk\                  any folder starting with "_" = not a known person (never named)
  person_0001\            a not-yet-known face (or group of the same face) waiting for a name
  .crops.json             the face embedding behind every crop / reference photo (the ground truth itself)
```

1. The first run with no known people asks whether to run `--faces-only` (default) or continue without faces.
2. Faces that are not recognised yet show up in `_faces\` as `person_0001`, `person_0002`, ... (similar faces are grouped, with a few sample crops each; a group needs at least 2 faces, see `--min-cluster`). Numbers come from one counter for the whole folder, so they are unique across photo folders and never reused after you rename.
3. Rename a folder to the person's name (`Alice`). Drag crops between folders to merge groups of the same person. You can also drop ordinary photos of a person into `_faces\<Name>\`.
4. Run again. Names are written as keywords and into the JSON.

**The faces you leave in a named folder are the ground truth, face by face.** Every other face is compared with those confirmed faces and named only if it is similar enough and clearly closer to one person than to any other (`--face-threshold`, `--face-margin`).

- Wrong face in a named folder? Move that crop to the right person's folder, or to a `_` folder. Run again; it is re-labelled and the other faces re-sorted accordingly.
- Named folders are never added to automatically, so nothing unconfirmed ends up in them. New, unmatched faces appear in new `person_NNNN` folders, once at least 2 faces of the same person are seen (`--min-cluster 1` shows every single unknown face). Faces that are not people (background, statues) go to a `_junk` folder.
- Deleting a crop does not un-label it; move it instead. More confirmed faces per person give better matching (`--crops-per-cluster`).
- Look-alikes (siblings, parent and child) get their own folders; matching then separates them.
- Confirmations are tied to the face itself (via `.crops.json`), not to file names, photo order, the cache or a photo folder. They survive re-sorting, renaming or editing photos and deleting `iqa_scores.json`, and a replaced photo never inherits an old name. Back up `_faces\` (it is small) and don't delete `.crops.json`.
- Other options: `--faces-dir` uses a custom ground-truth folder, `--cluster-threshold` controls how unnamed faces are grouped, `--reset-faces` forgets one photo folder's detected faces and groups (never the ground truth).

## Presets

| Preset | Models |
|---|---|
| `cpu` | musiq, topiq_nr, clipiqa, liqe, nima | 1024 |
| `gpu` | the `cpu` set + arniqa, hyperiqa, dbcnn |
| `all` | every pyiqa no-reference model that scores single photos (about 69), very slow without a GPU |

With no options the script uses `gpu` when it finds a GPU and `cpu` otherwise (force it with `--device cpu`). Ratings always combine every model already in the cache, so you can start with `cpu` and later add the other models with `--preset gpu`. Every preset scores at the same size (`--max-side`, default 1024), so scores from different runs, machines and presets are comparable; if you change `--max-side` the script warns that old and new scores were made at different sizes.

## Gallery: browse the ranking

`imageautorater-gallery` turns the results of a photo folder into one web page:

```
imageautorater-gallery "C:\Photos\2025" --open
```

It reads `iqa_results.json` and writes `gallery.html` next to it (open it with a double click; it needs no server). Small thumbnails are cached in `.iqa_thumbs\` inside the photo folder, so the page is fast even with large originals. Photos are never modified.

- **Per folder:** photos are grouped by subfolder, best first, showing the top 50 of each (`--top`). **Show more** reveals the next 50. Pick one folder, or "All folders, merged" for one list that keeps loading as you scroll.
- **Sorting:** overall (mean of all models, on a 0-100 scale), mean stars, any single model by its own score, or file name; best first or worst first. Models where lower is better (such as NIQE) are handled.
- **Filters:** minimum stars, person, keyword, file name.
- **Large view:** click a photo for a big picture with every model's score and stars; arrow keys move through the list, Esc closes.
- Options: `--out` for another output file, `--no-thumbs` to use the originals, `--thumb-size`, `--results` for another results file.

Run `imageautorater-gallery` again after each `imageautorater-rate` run to refresh the page.

## Cache and resuming

Everything is cached, including scores, CLIP embeddings and face embeddings. An interrupted run resumes where it stopped. Re-runs only analyse new or modified photos and new models. Ratings always combine every model in the cache. `--fresh` starts a new cache and keeps the old one as `.bak`.

## Troubleshooting

**The GPU is not used ("device: cpu" on a machine with an NVIDIA card).** On Windows the default PyTorch from PyPI is a CPU-only build, so a plain install ends up on the CPU even with an NVIDIA GPU (`uv sync` in a source checkout installs the CPU build on purpose, too). `imageautorater-rate` checks this itself at the start of every run, before PyTorch is loaded: if `nvidia-smi` shows an NVIDIA GPU but the installed PyTorch is CPU-only, it says so and offers the fix:

```
An NVIDIA GPU is present (NVIDIA GeForce RTX 2060, 610.74) but the installed PyTorch is a CPU-only build, so the GPU would NOT be used.
Install the CUDA build of PyTorch now? It downloads about 2-3 GB and replaces torch and torchvision in this environment. [Y/n]
```

- **Yes** installs it with `uv pip install --reinstall torch torchvision --torch-backend=auto` into the environment you run from (or, without uv, with `pip` and the PyTorch index that matches your driver's CUDA version), checks it in a fresh process, and carries on with the GPU. It only ever runs before PyTorch is loaded, because a loaded PyTorch locks its files on Windows.
- **Nothing is installed without your consent.** Without a terminal (a scheduled job, a pipe) it only reports; add `--fix-gpu` to install without being asked, or `--no-fix-gpu` to stop the question. `--device cpu` also skips it, and machines without an NVIDIA GPU (Intel, AMD, none) are never touched.
- If the installer fails, the run continues on the CPU and prints the command to run yourself.

By hand: `uv pip install --reinstall torch torchvision --torch-backend=auto`, then check `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`: you want a CUDA version and `True`. If a CUDA build still says `False`, update the NVIDIA driver (`nvidia-smi` must work and show the GPU). After the fix the `auto` preset picks the 8-model GPU set, at the same image size as on the CPU. Results already in the cache are kept.

**Something did not run (the ATTENTION block).** A run that could not do everything finishes with a framed list of what was skipped and the command that fixes it, so it is not lost among the download progress. Typical entries:

- `model clipiqa / liqe was skipped ... No module named 'pkg_resources'`: the CLIP package behind these two models needs `pkg_resources`, which `setuptools` 82 and newer removed. This package now pins `setuptools<82`; on an older install run `uv pip install "setuptools<82"`.
- `faces were NOT detected ... facenet_pytorch`: install it with `uv pip install --no-deps facenet-pytorch` (see Install).
- A model that fails on its first photos is given up and listed.

Everything already scored stays in the cache, so after fixing, run the same command again and only the missing parts are done. In the log, a model reported as `cached` was scored by an earlier run and is not computed again.

**Damaged photos (`broken data stream when reading image file`).** A truncated or corrupt JPEG cannot be decoded. The photo is skipped for every model, remembered in the cache and listed in `iqa_results.json` under `unreadable` (the log shows a NOTE with the first few). It is not tried again on later runs until the file changes, and it does not count against a model. Only damaged files are remembered: file system and network errors (permission, connection lost) are not, so a dropped connection never blacklists a good photo. A "Corrupt EXIF data" warning only means a damaged metadata block; the picture still loads and is scored.

**"Access is denied" when saving on a NAS or synced folder.** Every file the program writes (the cache `iqa_scores.json`, the results, sidecars) is written next to the target and then renamed over it. On a network share that rename can fail while another program has the target open (a sync client, an antivirus scanner, the search indexer, another PC) or because the share allows writing but not renaming over a file. The program therefore:

- retries a few times with growing pauses (most locks are brief), then overwrites the file in place (caches and sidecars only; **photos are never overwritten in place**),
- never stops a finished run because of it: if the share still refuses, the results stay in memory, the save is retried, and at the end a local copy is kept in `%LOCALAPPDATA%\imageautorater\recovery\` and listed in the ATTENTION block,
- saves progress at most once a minute (the cache can be tens of megabytes) and also on Ctrl+C or a crash,
- uses a leftover `iqa_scores.json.tmp` from an unfinished save automatically at the next start when it is newer than the cache, so **do not delete that file**.

If it keeps happening, find what holds the file (Resource Monitor, Associated Handles), exclude `iqa_*.json` from the antivirus or sync client, or check that your account may delete and rename files on the share.

**It is slow.** Every model reads and decodes every photo, and decoding a large JPEG (about 24 megapixels) takes roughly half a second of CPU, which is often more than the model itself on a GPU. The next photos are decoded in the background while a model works (`--io-threads N`, default 4; more helps on a PC with many cores and on a network share). Fewer models help most. `maniqa` is deliberately not in any preset: on a 6 GB GPU it took about 4.5 s per photo, more than all the other models together (add it yourself with `--models` if you want it). Start with `--limit 50` to measure, then multiply.

**A big folder on a network share (NAS) is slow to start.** The folder is read with directory listings, which is fast even for tens of thousands of photos (35,000 photos in about 10 seconds on a NAS). The progress lines `[mm:ss]` show every phase.

## Notes

- The first run downloads model weights (CLIP about 600 MB, plus each pyiqa model).
- On a CPU, expect seconds per photo per model. Use `--models`, and `--max-side 768` to go faster still.
- Face embeddings are biometric data. They are stored in `iqa_scores.json`, so keep that file private. If the photo folder syncs to Nextcloud, the file syncs too.
- If Windows blocks torch DLLs ("Application Control policy"), run it in WSL2.
