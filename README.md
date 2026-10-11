# imageautorater

Rates a folder of photos and writes the results as XMP sidecar files and a combined JSON file. Your image files are only read, never modified. A second command builds a web page to browse the ranking.

- **Quality:** scored with [pyiqa](https://github.com/chaofengc/IQA-PyTorch) no-reference models, giving 1-5 stars per model.
- **Image type keywords:** CLIP zero-shot classification (portrait, landscape, food, night, ...).
- **People:** faces are detected and grouped, then named by you.
- **Gallery:** `imageautorater-gallery` shows the best photos per folder, sortable by any model.

Commands: `imageautorater-rate` (also `python -m imageautorater`) and `imageautorater-gallery`.

## Install

Needs [uv](https://docs.astral.sh/uv/) (`winget install astral-sh.uv`). Python 3.12 or newer; uv fetches it if needed. Installation and the first model downloads need internet.

```
uv venv
uv pip install "imageautorater @ <wheel from the GitHub release>" --torch-backend=auto
```

`--torch-backend=auto` picks the PyTorch build for your hardware (NVIDIA CUDA, Intel XPU, otherwise CPU). The repository is private, so install from the wheel attached to a release (`gh release download`), not from `git+https`. For person identification add the `faces` extra together with the override file from the same release:

```
uv pip install "imageautorater[faces] @ <wheel>" --override overrides.txt --torch-backend=auto
```

(`facenet-pytorch` pins an old torch; `overrides.txt` replaces those pins.)

### Development

```
uv sync --group dev     # CPU torch, see pyproject.toml
uv run pytest           # fast; uses stand-ins for the models, downloads nothing
uv run imageautorater-rate --help
uv build                # wheel in dist/
```

Layout:

```
pyproject.toml          package metadata, dependencies, entry points, uv settings
overrides.txt           pins replaced when installing the `faces` extra
src/imageautorater/
    pipeline.py         rating, keywords, faces (imageautorater-rate)
    gallery.py          web page builder (imageautorater-gallery)
tests/                  pytest
.github/workflows/      ci.yml (tests on push/PR), release.yml (tag v* -> wheel on a GitHub release)
PLAN.md                 plan for the GUI and distribution
faces/                  your face ground truth in a source checkout (git-ignored, never commit)
```

Releases: bump `version` in `pyproject.toml`, tag `vX.Y.Z`, push the tag.

## Run

```
imageautorater-rate "C:\Photos\2025"
```

Subfolders are included by default (`--no-recursive` for the top folder only). Useful options:

| Option | Effect |
|---|---|
| `--dry-run` | Do all the analysis, write no `.xmp` files |
| `--preset auto\|cpu\|gpu\|all` | Model set. `auto` (default) picks `gpu` if a GPU is found (CUDA, Intel XPU, Apple MPS), otherwise `cpu` |
| `--models musiq topiq_nr` | Score with exactly these models (overrides `--preset`) |
| `--limit 10` | Only process the first 10 images (quick test) |
| `--max-side 1024` | Downscale before scoring (default: 1024 for the `cpu` preset, 2048 otherwise) |
| `--mode percentile` | Rank stars within the folder instead of using each model's score range |
| `--faces-only` | Only detect, group and name faces |
| `--no-iqa` / `--no-classify` / `--no-faces` | Turn off one part |
| `--labels labels.txt` | Own keyword vocabulary, one `label` or `label: prompt` per line |
| `--device cpu` | Force the device (`cuda`, `xpu` if available) |

`imageautorater-rate --help` lists everything.

## Output

- **`<photo>.xmp`** next to each image (or `<photo>.jpg.xmp` with `--darktable-naming`):
  - `xmp:Rating`: the rounded mean of the per-model ratings
  - `iqa:Rating_<model>` and `iqa:Score_<model>` for each model, plus `iqa:RatingMean`
  - `dc:subject`: image-type keywords and recognised names

  Existing sidecars are updated in place. Your own keywords and settings are kept. Use `--no-rating` to leave `xmp:Rating` alone.
- **`iqa_results.json`** in the photo folder: all scores, ratings, keywords and people per image.
- **`iqa_scores.json`** in the photo folder: the persistent cache.

## Naming people

There is **one ground truth for faces**: a single `faces\` folder, shared by every photo folder you ever process, so a person named once is recognised everywhere. Its location is, in order: `--faces-dir`, the `IMAGEAUTORATER_FACES` environment variable, `faces\` in the repository root when running from a source checkout (git-ignored), otherwise the per-user data folder (`%LOCALAPPDATA%\imageautorater\faces` on Windows, `~/.local/share/imageautorater/faces` elsewhere).

```
faces\
  Alice\                  confirmed faces of Alice: crops you moved here, and/or reference photos you dropped in
  Bob\
  _junk\                  any folder starting with "_" = not a known person (never named)
  person_0001\            a not-yet-known face (or group of the same face) waiting for a name
  .crops.json             the face embedding behind every crop / reference photo (the ground truth itself)
```

1. The first run with no known people asks whether to run `--faces-only` (default) or continue without faces.
2. Faces that are not recognised yet show up in `faces\` as `person_0001`, `person_0002`, ... (similar faces are grouped, with a few sample crops each; a group needs at least 2 faces, see `--min-cluster`). Numbers come from one counter for the whole folder, so they are unique across photo folders and never reused after you rename.
3. Rename a folder to the person's name (`Alice`). Drag crops between folders to merge groups of the same person. You can also drop ordinary photos of a person into `faces\<Name>\`.
4. Run again. Names are written as keywords and into the JSON.

**The faces you leave in a named folder are the ground truth, face by face.** Every other face is compared with those confirmed faces and named only if it is similar enough and clearly closer to one person than to any other (`--face-threshold`, `--face-margin`).

- Wrong face in a named folder? Move that crop to the right person's folder, or to a `_` folder. Run again; it is re-labelled and the other faces re-sorted accordingly.
- Named folders are never added to automatically, so nothing unconfirmed ends up in them. New, unmatched faces appear in new `person_NNNN` folders, once at least 2 faces of the same person are seen (`--min-cluster 1` shows every single unknown face). Faces that are not people (background, statues) go to a `_junk` folder.
- Deleting a crop does not un-label it; move it instead. More confirmed faces per person give better matching (`--crops-per-cluster`).
- Look-alikes (siblings, parent and child) get their own folders; matching then separates them.
- Confirmations are tied to the face itself (via `.crops.json`), not to file names, photo order, the cache or a photo folder. They survive re-sorting, renaming or editing photos and deleting `iqa_scores.json`, and a replaced photo never inherits an old name. Back up `faces\` (it is small) and don't delete `.crops.json`.
- Other options: `--faces-dir` uses another ground-truth folder, `--cluster-threshold` controls how unnamed faces are grouped, `--reset-faces` forgets one photo folder's detected faces and groups (never the ground truth).

## Presets

| Preset | Models | `--max-side` |
|---|---|---|
| `cpu` | musiq, topiq_nr, clipiqa, liqe, nima | 1024 |
| `gpu` | the `cpu` set + arniqa, hyperiqa, maniqa, dbcnn | 2048 |
| `all` | every pyiqa no-reference model that scores single photos (about 69), very slow without a GPU | 2048 |

With no options the script uses `gpu` when it finds a GPU and `cpu` otherwise (force it with `--device cpu`). Ratings always combine every model already in the cache, so you can start with `cpu` and later add the other models with `--preset gpu --max-side 1024`. The `--max-side` keeps the image size the same as the first run; without it the script warns that old and new scores were made at different sizes.

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

## Notes

- The first run downloads model weights (CLIP about 600 MB, plus each pyiqa model).
- On a CPU, expect seconds per photo per model. Use `--models` and `--max-side 1024`.
- Face embeddings are biometric data. They are stored in `iqa_scores.json`, so keep that file private. If the photo folder syncs to Nextcloud, the file syncs too.
- If Windows blocks torch DLLs ("Application Control policy"), run it in WSL2.
