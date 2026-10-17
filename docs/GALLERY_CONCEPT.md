# Concept: gallery v2 (stage 2, web frontend)

Status: proposal for approval. Decisions already taken by the owner:

| Topic | Decision |
|---|---|
| Relation to existing code | **Extend `imageautorater-gallery`** (same command, same `gallery.py`), no new tool |
| Comparing rankings | **All three**: side-by-side lists, custom blend, rank-change view |
| Delivery | **Static HTML**, no server; opens by double click |
| Input | **One root `iqa_results.json`** (rate was run on the root, e.g. `D:\Photos`) |

## 1. Goal

From the root results, answer one question quickly: **"which are the 50 best photos of each year (primary subfolder), and
how stable is that answer across ranking methods?"** Everything else (filters, large view) already exists and stays.

## 2. What exists today (gallery.py)

Reads `iqa_results.json`, makes 360 px thumbs in `.iqa_thumbs/`, embeds all items as JSON in one `gallery.html`, shows top 50
per folder with "Show more", one sort key at a time (overall / stars / one model / name), filters (stars, person, keyword, name),
lightbox with per-model scores.

Gaps against the goal: single ranking at a time; folder choice is a flat dropdown (one or all); no multi-select of years;
thumbs 360 px; everything in one HTML file (a 35,000-photo library means a 10+ MB page); no way to build a ranking of your own.

## 3. Features

### 3.1 Selector (which photos take part)
- Folder **tree with checkboxes**, built from the relative paths. Depth 1 = years. Check any mix of years/albums.
- **Group level** control: rank *per group at depth N* (default 1 = per year), top N each (default 50), or "all selected, merged".
- Counts next to each node (photos, with/without scores). Selection, ranking and filters are kept in the URL hash
  (`#sel=2023,2024&rank=blend:...&n=50`) so a view can be bookmarked; localStorage only as fallback.

### 3.2 Rankings ("rankers")
A ranker = a function photo -> number (higher is better). Built in:
- `overall` (mean of the 0-100 normalised model scores), `stars` (mean stars), each single model (lower-is-better models flipped),
  `name`.
- **Custom blend**: user ticks models and gives weights (sliders, 0-10). Score = weighted mean of the 0-100 values.
  Option "normalise within selection" (percentile rank of each model inside the chosen photos) because models differ in spread;
  default **on**, since the scale of musiq vs topiq differs and unequal spread silently gives one model all the weight.
- Blends can be saved under a name (localStorage, plus "copy as URL").
- Ties broken by mean stars, then file name (as today), so lists are deterministic.

### 3.3 Comparison (the three modes)
1. **Side by side**: 2-4 columns, each a ranker, each showing the top N of the same selection/group. A photo present in several
   columns gets a coloured marker (in all / in some); header shows overlap (e.g. "38 of 50 shared"). Clicking a photo
   highlights it in every column with its rank there. Group mode: one comparison block per year.
2. **Custom blend**: described above; it is a ranker like any other, so it can be one of the columns.
3. **Rank change**: pick ranker A and B. Table/grid of the union of both top-N: rank A, rank B, delta, sorted by |delta|, or
   "only in A" / "only in B" filters. Answers "which photos does model X love that the others do not?".

### 3.4 Browse and pick
- Existing grid, lightbox (arrows, Esc), per-model table, chips. Lightbox gets rank per ranker.
- **Open original**: link on every card/lightbox to the original file (`file://`, relative to the page, so it works for local
  disks and mapped network shares); RAW/TIFF/HEIC just open in the OS's default app. "Copy path" button next to it.
- **Export** the current top-N list(s) as `.txt` (full paths) and `.csv` (rank, path, per-ranker scores), generated in the
  browser (download), for copying the 50 per year into a folder.

## 4. Thumbnails

- **One dedicated folder**, default `<root>/.iqa_thumbs` (existing name; hidden, so it is never scanned as photos); override
  with `--thumb-dir`. Flat, name = `<path-hash>_<fingerprint-hash>_<size>.jpg` (existing scheme, so edited photos get fresh
  thumbs and old ones are reused).
- **Max side 512 px** (default, was 360), JPEG q80-82, `--thumb-size` to change. Sizes coexist (size is in the name).
- **All images in the results get a thumbnail**, not only the visible top 50, so any ranker or filter is instant. Generated in
  parallel (`ProcessPoolExecutor`, `--jobs`, default = CPU count - 1), only missing ones, resumable, progress line `[mm:ss]`,
  failures listed at the end and shown as a placeholder. `--thumbs-only` builds them without writing the page (can run
  overnight on the NAS box); `--no-thumbs` keeps the old fallback.
- Thumbs live in a folder, not in the HTML, so the page stays small and the cache survives re-rating.

## 5. Architecture

```
iqa_results.json (root)  ──►  gallery.py (build)  ──►  <out>/gallery.html        shell + JS + CSS, small, static
                                   │                    <out>/gallery_data/        index.js + one data file per top folder
                                   └──────────────►    <thumb-dir>/*.jpg          512 px thumbnails
```

- **Package layout** (all inside the existing package, no new dependency):
  - `gallery.py`: CLI, data build, thumbnails (as today, extended), writes files.
  - `gallery_thumbs.py`: thumbnail pool (parallel, resumable) split out of `gallery.py` because it grows.
  - `gallery_page/index.html`, `gallery.js`, `gallery.css`: the page sources as real files (package data, included in the
    wheel by hatch) instead of the 250-line string `PAGE`; the build inlines them into **one `gallery.html`** so the page is
    still a single double-click file plus the data/thumbs next to it.
- **Data files**: `gallery_data/index.js` (folders, models, `lower`, counts) and `gallery_data/<folder>.js`, each
  `window.__IQA_DATA["2023"] = [...]`. They are loaded with `<script src>` on demand when a folder is ticked: this works from
  `file://` where `fetch()` of local files does not. Item record unchanged from today (`p,f,t,s,o,r,m,g,n`) plus `u`
  (original URL). Estimated size ~250 bytes/photo (35,000 photos ≈ 9 MB total, but only selected years are loaded).
- **Client logic** (plain JS, no framework, no CDN, works offline): state object -> `rankers` -> `selection` -> views
  (grid / compare / delta). Ranking and overlap are pure functions in one block, so they can be tested with Node.
- **Compatibility**: `--top`, `--thumb-size`, `--no-thumbs`, `--out`, `--results`, `--open` keep working; `--legacy-single-file`
  is not needed (data stays embedded if `--embed` is given, for sending the page as one file, thumbs excluded).

## 6. Testing
- pytest (existing style, stubbed, no downloads): data build for nested years, thumb size cap 512 and reuse, parallel
  thumbs with a corrupt file, data file split, URL escaping for odd file names, `norm_score` unchanged.
- JS: ranking/blend/overlap/delta functions extracted to `gallery_logic.js`, tested by a small pytest that runs `node` if
  available (skipped otherwise); plus one Playwright smoke test (Chromium is available in CI images) that loads the generated
  page from `file://`, ticks two years, checks 50 cards per year and an overlap number.

## 7. Implementation order (one run)
1. Split thumbs module, 512 px, parallel + `--thumbs-only`, `--thumb-dir`.
2. Page sources as files, build inliner, per-folder data files.
3. Folder tree selector, group level, URL state.
4. Rankers + blend + saved blends.
5. Side-by-side, rank-change view, export, open/copy original.
6. Tests, README section "Gallery", version bump to 0.2.0.

## 8. Assumptions I will use unless you object
1. Thumbnail folder stays `.iqa_thumbs` in the root (hidden, not scanned); a visible `_thumbs` is a one-flag change.
2. Blend defaults to percentile normalisation within the selection.
3. "Year" = first path component; photos directly in the root form a group "(top folder)".
4. Originals are linked by `file://` relative to the page, so the page must stay in (or below the same drive as) the photo root;
   `gallery.html` is written to `<root>/gallery.html` as today.
5. No writing back (no star changes, no file moves/copies) from the page: export lists only.
