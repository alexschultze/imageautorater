#!/usr/bin/env python3
"""Build a browsable web page from the results of imageautorater-rate.

  imageautorater-gallery "C:\\Photos\\2025"            # writes <folder>\\gallery.html and opens nothing
  imageautorater-gallery "C:\\Photos\\2025" --open     # ... and opens it in your browser

The page shows the photos grouped by (sub)folder. Within each folder the best photos come first and only the top 50
are shown; "Show more" (or scrolling, in the merged view) reveals the rest. You can sort by the overall score or by
any single model, flip the order, and filter by folder, person, keyword and minimum stars. Click a photo for a large
view with every model's score.

It reads <folder>/iqa_results.json (written by imageautorater-rate) and never modifies your photos. Small thumbnails are
cached in <folder>/.iqa_thumbs/ so the page opens quickly even when the originals are large.
"""
import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path
from urllib.parse import quote

from imageautorater.pipeline import RESULTS_NAME, fingerprint, load_image, rel_hash

BROWSER_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}  # shown directly in the large view


def norm_score(score, info, lo_hi):
    """Raw model score -> 0..100, higher = better (direction and range taken from the model's own metadata)."""
    lo, hi = (info.get("range") or lo_hi)
    if hi <= lo:
        return 50.0
    frac = max(0.0, min(1.0, (score - lo) / (hi - lo)))
    return 100.0 * ((1 - frac) if info.get("lower_better") else frac)


def make_thumb(src: Path, dst: Path, size: int):
    img = load_image(src, size)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=82, optimize=True)


def build_data(folder: Path, results: dict, out_dir: Path, thumbs: bool, thumb_size: int):
    models = results.get("models", {})
    # models without a known score range: use the observed min/max over this folder
    observed = {}
    for entry in results["images"].values():
        for m, v in entry.get("models", {}).items():
            lo, hi = observed.get(m, (v["score"], v["score"]))
            observed[m] = (min(lo, v["score"]), max(hi, v["score"]))

    items, missing = [], 0
    todo = sorted(results["images"].items())
    for n, (rel, entry) in enumerate(todo, 1):
        src = folder / rel
        if not src.is_file():
            missing += 1
            continue
        if thumbs:
            # name = path hash + file fingerprint, so an edited photo gets a fresh thumbnail
            thumb = folder / ".iqa_thumbs" / f"{rel_hash(rel)}_{rel_hash(fingerprint(src))}_{thumb_size}.jpg"
            if not thumb.exists():
                sys.stdout.write(f"\rthumbnails {n}/{len(todo)}  {src.name[:50]:50}")
                sys.stdout.flush()
                try:
                    make_thumb(src, thumb, thumb_size)
                except Exception as e:
                    print(f"\n  no thumbnail for {rel}: {str(e)[:80]}", file=sys.stderr)
                    thumb = None
            thumb_path = thumb if thumb and thumb.exists() else None
        else:
            thumb_path = None

        def url(p: Path) -> str:
            return quote(Path(os.path.relpath(p, out_dir)).as_posix(), safe="/")

        per_model = {}
        norms = []
        for m, v in entry.get("models", {}).items():
            nv = norm_score(v["score"], models.get(m, {}), observed.get(m, (0, 1)))
            per_model[m] = [v["rating"], v["score"], round(nv, 1)]
            norms.append(nv)
        people = sorted({p["name"] for p in entry.get("people", []) if p.get("name")})
        items.append({
            "p": rel,
            "f": "" if Path(rel).parent.as_posix() == "." else Path(rel).parent.as_posix(),
            "t": url(thumb_path) if thumb_path else url(src) if src.suffix.lower() in BROWSER_EXT else "",
            "s": url(src) if src.suffix.lower() in BROWSER_EXT else (url(thumb_path) if thumb_path else ""),
            "o": round(sum(norms) / len(norms), 1) if norms else None,
            "r": entry.get("rating_mean"),
            "m": per_model,
            "g": list(entry.get("tags", {})),
            "n": people,
        })
    if thumbs:
        print()
    if missing:
        print(f"{missing} photos listed in the results no longer exist and are left out", file=sys.stderr)
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="photo folder that holds iqa_results.json")
    ap.add_argument("--results", type=Path, help=f"results file (default: <folder>/{RESULTS_NAME})")
    ap.add_argument("--out", type=Path, help="output page (default: <folder>/gallery.html)")
    ap.add_argument("--top", type=int, default=50, help="photos shown per folder before 'Show more' (default 50)")
    ap.add_argument("--thumb-size", type=int, default=360, help="thumbnail size in pixels (default 360)")
    ap.add_argument("--no-thumbs", action="store_true", help="use the original files as thumbnails (slow for big photos)")
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

    items = build_data(folder, results, out.parent, not a.no_thumbs, a.thumb_size)
    if not items:
        sys.exit("None of the photos in the results exist any more.")
    model_names = sorted({m for it in items for m in it["m"]})
    lower = {m: bool(results.get("models", {}).get(m, {}).get("lower_better")) for m in model_names}
    data = {"items": items, "models": model_names, "lower": lower, "top": a.top, "title": folder.name or str(folder),
            "mode": results.get("mode", ""), "generated": results.get("generated", "")}
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    out.write_text(PAGE.replace("/*__DATA__*/null", payload), encoding="utf-8")
    print(f"{len(items)} photos, {len(model_names)} models -> {out}")
    if a.open:
        webbrowser.open(out.as_uri())


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Photo ranking</title>
<style>
:root{--bg:#f6f6f4;--panel:#ffffff;--ink:#1d1d1b;--mute:#6b6b66;--line:#dcdcd6;--accent:#2f6fdb;--chip:#ecece6;--star:#d98a00}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--panel:#1f1f1d;--ink:#ecece8;--mute:#9a9a92;--line:#34342f;--accent:#6aa0ff;--chip:#2b2b28;--star:#f0b341}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:16px;margin:0 0 8px;font-weight:600}
h1 small{color:var(--mute);font-weight:400;margin-left:8px}
.bar{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center}
label{display:flex;align-items:center;gap:6px;color:var(--mute);white-space:nowrap}
select,input,button{font:inherit;color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button{cursor:pointer}button:hover{border-color:var(--accent)}
main{padding:16px;max-width:1800px;margin:0 auto}
.group{margin-bottom:28px}
.group h2{font-size:15px;margin:0 0 10px;display:flex;gap:10px;align-items:baseline}
.group h2 span{color:var(--mute);font-weight:400;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden;cursor:pointer;display:flex;flex-direction:column}
.card:hover{border-color:var(--accent)}
.thumb{position:relative;aspect-ratio:4/3;background:var(--chip)}
.thumb img{width:100%;height:100%;object-fit:cover;display:block}
.rank{position:absolute;left:6px;top:6px;background:rgba(0,0,0,.65);color:#fff;border-radius:4px;padding:1px 6px;font-size:12px}
.score{position:absolute;right:6px;top:6px;background:rgba(0,0,0,.65);color:#fff;border-radius:4px;padding:1px 6px;font-size:12px;font-variant-numeric:tabular-nums}
.meta{padding:6px 8px 8px;display:flex;flex-direction:column;gap:4px;min-height:58px}
.name{font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.stars{color:var(--star);font-size:12px;letter-spacing:1px}
.chips{display:flex;flex-wrap:wrap;gap:4px}
.chip{background:var(--chip);color:var(--mute);border-radius:10px;padding:0 7px;font-size:11px}
.chip.person{color:var(--ink);font-weight:600}
.more{margin:14px auto 0;display:block}
.empty{color:var(--mute);padding:40px;text-align:center}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.88);display:none;z-index:10;grid-template-columns:1fr 320px}
#lb.on{display:grid}
#lbimg{display:flex;align-items:center;justify-content:center;padding:16px;min-width:0;min-height:0}
#lbimg img{max-width:100%;max-height:100%;object-fit:contain}
#lbside{background:var(--panel);overflow:auto;padding:14px;border-left:1px solid var(--line)}
#lbside h3{margin:0 0 4px;font-size:14px;word-break:break-all}
#lbside table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px;font-variant-numeric:tabular-nums}
#lbside td,#lbside th{padding:3px 4px;border-bottom:1px solid var(--line);text-align:right}
#lbside td:first-child,#lbside th:first-child{text-align:left}
#lbx{position:absolute;right:340px;top:10px;background:rgba(0,0,0,.6);color:#fff;border:0;font-size:20px;padding:2px 10px}
@media (max-width:760px){#lb{grid-template-columns:1fr}#lbside{display:none}#lbx{right:10px}}
</style>
</head>
<body>
<header>
  <h1 id="title">Photo ranking<small id="count"></small></h1>
  <div class="bar">
    <label>Folder <select id="folder"></select></label>
    <label>Sort by <select id="sort"></select></label>
    <label>Order <select id="dir"><option value="-1">best first</option><option value="1">worst first</option></select></label>
    <label>Min stars <select id="minstars"><option value="0">any</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
    <label>Person <select id="person"></select></label>
    <label>Keyword <select id="tag"></select></label>
    <label>Search <input id="q" type="search" placeholder="file name" size="14"></label>
  </div>
</header>
<main id="main"></main>
<div id="lb"><div id="lbimg"></div><div id="lbside"></div><button id="lbx" title="Close (Esc)">&times;</button></div>
<script id="data" type="application/json">/*__DATA__*/null</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const $ = id => document.getElementById(id);
const PAGE = D.top || 50;
const GROUPED = '\u0000grouped', MERGED = '\u0000merged';
const folderLabel = f => f === '' ? '(top folder)' : f;
document.title = 'Photo ranking \u2013 ' + D.title;
$('title').firstChild.textContent = D.title + ' ';

// ---- controls ---------------------------------------------------------------------------------------------
function opt(sel, value, text){ const o = document.createElement('option'); o.value = value; o.textContent = text; sel.appendChild(o); }
const folders = [...new Set(D.items.map(i => i.f))].sort();
opt($('folder'), GROUPED, 'All folders, grouped'); opt($('folder'), MERGED, 'All folders, merged');
folders.forEach(f => opt($('folder'), f, folderLabel(f) + ' (' + D.items.filter(i => i.f === f).length + ')'));
opt($('sort'), 'overall', 'Overall (mean of all models)');
opt($('sort'), 'rating', 'Mean stars');
D.models.forEach(m => opt($('sort'), 'm:' + m, m));
opt($('sort'), 'name', 'File name');
const people = [...new Set(D.items.flatMap(i => i.n))].sort(), tags = [...new Set(D.items.flatMap(i => i.g))].sort();
opt($('person'), '', 'anyone'); people.forEach(p => opt($('person'), p, p));
opt($('tag'), '', 'any'); tags.forEach(t => opt($('tag'), t, t));
if (folders.length === 1) $('folder').value = MERGED;   // a single folder needs no grouping

// value to sort by, higher = better. A single model sorts by its raw score (flipped for lower-is-better models like
// NIQE) so that scores outside the model's nominal range are still told apart; "overall" uses the 0-100 values.
const val = (it, key) => {
  if (key === 'overall') return it.o;
  if (key === 'rating') return it.r;
  if (key.startsWith('m:')) { const m = key.slice(2), v = it.m[m]; return v ? (D.lower[m] ? -v[1] : v[1]) : null; }
  return null;
};
const rawLabel = (it, key) => {
  if (key === 'overall') return it.o == null ? '' : it.o.toFixed(1);
  if (key === 'rating') return it.r == null ? '' : '\u2605 ' + it.r.toFixed(1);
  if (key.startsWith('m:')) { const m = it.m[key.slice(2)]; return m ? m[1].toFixed(2) : ''; }
  return '';
};

function sorted(list){
  const key = $('sort').value, dir = +$('dir').value;   // dir -1 = best first (highest score first), 1 = worst first
  const a = list.slice();
  if (key === 'name') a.sort((x, y) => -dir * x.p.localeCompare(y.p, undefined, {numeric: true}));   // best first = A to Z
  else a.sort((x, y) => {
    const vx = val(x, key), vy = val(y, key);
    if (vx == null && vy == null) return x.p.localeCompare(y.p);
    if (vx == null) return 1; if (vy == null) return -1;      // photos without this score always last
    return dir * (vx - vy) || dir * ((x.r || 0) - (y.r || 0)) || x.p.localeCompare(y.p);
  });
  return a;
}
function filtered(){
  const min = +$('minstars').value, person = $('person').value, tag = $('tag').value, q = $('q').value.trim().toLowerCase();
  return D.items.filter(i => (!min || (i.r != null && i.r >= min - 0.5)) && (!person || i.n.includes(person)) &&
    (!tag || i.g.includes(tag)) && (!q || i.p.toLowerCase().includes(q)));
}

// ---- cards ------------------------------------------------------------------------------------------------
function card(it, rank, list){
  const key = $('sort').value;
  const c = document.createElement('div'); c.className = 'card'; c.title = it.p;
  const t = document.createElement('div'); t.className = 'thumb';
  if (it.t) { const im = document.createElement('img'); im.loading = 'lazy'; im.decoding = 'async'; im.src = it.t; im.alt = ''; t.appendChild(im); }
  const r = document.createElement('span'); r.className = 'rank'; r.textContent = '#' + rank; t.appendChild(r);
  const s = rawLabel(it, key); if (s) { const e = document.createElement('span'); e.className = 'score'; e.textContent = s; t.appendChild(e); }
  c.appendChild(t);
  const m = document.createElement('div'); m.className = 'meta';
  const n = document.createElement('div'); n.className = 'name'; n.textContent = it.p.split('/').pop(); m.appendChild(n);
  if (it.r != null) { const st = document.createElement('div'); st.className = 'stars'; const k = Math.round(it.r); st.textContent = '\u2605'.repeat(k) + '\u2606'.repeat(5 - k); m.appendChild(st); }
  const chips = document.createElement('div'); chips.className = 'chips';
  it.n.forEach(p => { const e = document.createElement('span'); e.className = 'chip person'; e.textContent = p; chips.appendChild(e); });
  it.g.slice(0, 3).forEach(g => { const e = document.createElement('span'); e.className = 'chip'; e.textContent = g; chips.appendChild(e); });
  m.appendChild(chips); c.appendChild(m);
  c.addEventListener('click', () => openLb(list, rank - 1));
  return c;
}

// a section = one grid showing PAGE more cards each time it is asked to grow
function section(parent, list, observe){
  const grid = document.createElement('div'); grid.className = 'grid'; parent.appendChild(grid);
  const btn = document.createElement('button'); btn.className = 'more'; parent.appendChild(btn);
  let shown = 0;
  const grow = () => {
    const end = Math.min(list.length, shown + PAGE);
    for (; shown < end; shown++) grid.appendChild(card(list[shown], shown + 1, list));
    btn.style.display = shown < list.length ? '' : 'none';
    btn.textContent = 'Show ' + Math.min(PAGE, list.length - shown) + ' more (' + (list.length - shown) + ' left)';
  };
  btn.addEventListener('click', grow);
  grow();
  if (observe) {   // merged / single-folder view: keep loading the next page while the user scrolls near the end
    autoGrow = () => { if (btn.style.display !== 'none' && innerHeight + scrollY > document.documentElement.scrollHeight - 900) { grow(); setTimeout(() => autoGrow && autoGrow(), 0); } };
    autoGrow();
  }
}
let autoGrow = null;
addEventListener('scroll', () => autoGrow && autoGrow(), {passive: true});

function render(){
  autoGrow = null;
  const main = $('main'); main.textContent = '';
  const all = filtered(), sel = $('folder').value;
  $('count').textContent = all.length + ' of ' + D.items.length + ' photos' + (D.mode ? ' \u00b7 ' + D.mode + ' scoring' : '');
  if (!all.length) { const e = document.createElement('div'); e.className = 'empty'; e.textContent = 'No photos match these filters.'; main.appendChild(e); return; }
  if (sel === GROUPED) {
    folders.forEach(f => {
      const part = all.filter(i => i.f === f); if (!part.length) return;
      const g = document.createElement('div'); g.className = 'group';
      const h = document.createElement('h2'); h.textContent = folderLabel(f);
      const sp = document.createElement('span'); sp.textContent = part.length + ' photos'; h.appendChild(sp);
      g.appendChild(h); main.appendChild(g); section(g, sorted(part), false);   // grouped: explicit "Show more" per folder
    });
  } else {
    const part = sel === MERGED ? all : all.filter(i => i.f === sel);
    const g = document.createElement('div'); g.className = 'group'; main.appendChild(g);
    section(g, sorted(part), true);
  }
}

// ---- large view -------------------------------------------------------------------------------------------
let lbList = [], lbPos = 0;
function openLb(list, pos){ lbList = list; lbPos = pos; showLb(); $('lb').classList.add('on'); }
function showLb(){
  const it = lbList[lbPos];
  const box = $('lbimg'); box.textContent = '';
  const im = document.createElement('img'); im.src = it.s || it.t; box.appendChild(im);
  const side = $('lbside'); side.textContent = '';
  const h = document.createElement('h3'); h.textContent = it.p; side.appendChild(h);
  const info = document.createElement('div'); info.style.color = 'var(--mute)';
  info.textContent = '#' + (lbPos + 1) + ' of ' + lbList.length + (it.o != null ? ' \u00b7 overall ' + it.o.toFixed(1) : '') + (it.r != null ? ' \u00b7 \u2605 ' + it.r.toFixed(1) : '');
  side.appendChild(info);
  if (it.n.length || it.g.length) {
    const c = document.createElement('div'); c.className = 'chips'; c.style.marginTop = '8px';
    it.n.forEach(p => { const e = document.createElement('span'); e.className = 'chip person'; e.textContent = p; c.appendChild(e); });
    it.g.forEach(g => { const e = document.createElement('span'); e.className = 'chip'; e.textContent = g; c.appendChild(e); });
    side.appendChild(c);
  }
  const tb = document.createElement('table');
  tb.innerHTML = '<tr><th>Model</th><th>Stars</th><th>Score</th><th>0-100</th></tr>';
  Object.entries(it.m).sort((a, b) => b[1][2] - a[1][2]).forEach(([m, v]) => {
    const tr = document.createElement('tr');
    [m, v[0], v[1].toFixed(2), v[2].toFixed(0)].forEach(x => { const td = document.createElement('td'); td.textContent = x; tr.appendChild(td); });
    tb.appendChild(tr);
  });
  side.appendChild(tb);
}
function step(d){ const n = lbPos + d; if (n >= 0 && n < lbList.length) { lbPos = n; showLb(); } }
$('lbx').addEventListener('click', () => $('lb').classList.remove('on'));
$('lbimg').addEventListener('click', e => { if (e.target.id === 'lbimg') $('lb').classList.remove('on'); });
document.addEventListener('keydown', e => {
  if (!$('lb').classList.contains('on')) return;
  if (e.key === 'Escape') $('lb').classList.remove('on');
  else if (e.key === 'ArrowRight') step(1);
  else if (e.key === 'ArrowLeft') step(-1);
});

['folder', 'sort', 'dir', 'minstars', 'person', 'tag'].forEach(id => $(id).addEventListener('change', render));
$('q').addEventListener('input', render);
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
