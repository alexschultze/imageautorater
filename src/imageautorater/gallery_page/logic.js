// Pure functions of the gallery (no DOM), so that they can be tested with node.
// Item record: {p path, f folder, o overall 0-100, r mean stars, m [[stars, score, 0-100] | null per model], g keywords, n people}
(function (root) {
  'use strict';
  const L = {};

  // ---- folders ------------------------------------------------------------------------------------------
  L.parts = f => (f === '' ? [] : f.split('/'));
  L.depthOf = f => L.parts(f).length;
  L.parentOf = f => (f.indexOf('/') < 0 ? null : f.slice(0, f.lastIndexOf('/')));
  // the group a folder belongs to when ranking per group at `depth` (0 = one group for everything)
  L.groupKey = (f, depth) => (depth ? L.parts(f).slice(0, depth).join('/') : '');
  L.natural = (a, b) => a.localeCompare(b, undefined, {numeric: true});

  // direct: {folder path: number of photos directly in it}. Every ancestor becomes a node, "" is the top folder itself.
  L.buildTree = function (direct) {
    const set = new Set();
    Object.keys(direct).forEach(p => {
      set.add(p);
      for (let q = L.parentOf(p); q !== null; q = L.parentOf(q)) set.add(q);
    });
    const paths = [...set].sort(L.natural);
    const children = {'': []}, total = {};
    paths.forEach(p => { if (p !== '') children[p] = children[p] || []; });
    paths.forEach(p => {
      if (p === '') return;
      const par = L.parentOf(p);
      (par === null ? children[''] : children[par]).push(p);
    });
    const sum = p => {
      if (total[p] !== undefined) return total[p];
      let t = direct[p] || 0;
      if (p !== '') children[p].forEach(c => { t += sum(c); });
      return (total[p] = t);
    };
    paths.forEach(sum);
    // "" only holds the photos lying directly in the root; it is not the parent of the other folders
    return {paths, children, direct, total, byPrefix: p => paths.filter(q => q === p || q.startsWith(p + '/'))};
  };
  // selecting a folder selects everything below it
  L.descendants = (tree, p) => (p === '' ? [''] : tree.byPrefix(p));
  L.setTick = function (tree, sel, p, on) {
    L.descendants(tree, p).forEach(q => (on ? sel.add(q) : sel.delete(q)));
    return sel;
  };
  L.tickState = function (tree, sel, p) {
    const ds = L.descendants(tree, p).filter(q => tree.total[q] > 0 || q === p);
    const n = ds.filter(q => sel.has(q)).length;
    return n === 0 ? 'none' : n === ds.length ? 'all' : 'some';
  };
  // compact, unambiguous form for the URL: "+folder" ticks it with its subfolders, "-folder" unticks it with its subfolders
  L.encodeSel = function (tree, sel) {
    const ops = [];
    tree.paths.forEach(p => {
      if (p === '') { if (sel.has('')) ops.push('+.'); return; }
      const par = L.parentOf(p), parTicked = par !== null && sel.has(par);
      if (sel.has(p) && !parTicked) ops.push('+' + p);
      else if (!sel.has(p) && parTicked) ops.push('-' + p);
    });
    return ops.join('|');
  };
  L.decodeSel = function (tree, text) {
    const sel = new Set();
    const ops = text.split('|').filter(Boolean).map(o => [o[0], o.slice(1)]);
    ops.sort((a, b) => L.depthOf(a[1] === '.' ? '' : a[1]) - L.depthOf(b[1] === '.' ? '' : b[1]));
    ops.forEach(([sign, p]) => L.setTick(tree, sel, p === '.' ? '' : p, sign === '+'));
    return sel;
  };

  // ---- scores and rankings --------------------------------------------------------------------------------
  // spec: {type: 'overall' | 'stars' | 'name' | 'model' (m: index) | 'blend' (w: {index: weight}, pct: bool)}
  // A scorer maps an item to a number, higher = better, or null when the item has no such score.
  // lower[i] = true for models where a lower raw score is better (such as NIQE).
  L.makeScorer = function (spec, items, lower) {
    if (spec.type === 'overall') return it => (it.o == null ? null : it.o);
    if (spec.type === 'stars') return it => (it.r == null ? null : it.r);
    if (spec.type === 'model') {
      const i = spec.m;      // raw score (not the clipped 0-100 value), flipped, so scores outside the nominal range still differ
      return it => { const v = it.m[i]; return v ? (lower[i] ? -v[1] : v[1]) : null; };
    }
    if (spec.type === 'blend') {
      const ws = Object.entries(spec.w || {}).map(([i, w]) => [+i, +w]).filter(([, w]) => w > 0);
      if (!ws.length) return () => null;
      // percentile of a model's 0-100 value among the given photos (0-100), so that a model with a narrow spread
      // does not count for less than one with a wide spread
      const sorted = {};
      if (spec.pct) ws.forEach(([i]) => { sorted[i] = items.filter(it => it.m[i]).map(it => it.m[i][2]).sort((a, b) => a - b); });
      const pctile = (arr, v) => {
        let lo = 0, hi = arr.length;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < v) lo = mid + 1; else hi = mid; }
        const less = lo;
        hi = arr.length;
        let lo2 = lo;
        while (lo2 < hi) { const mid = (lo2 + hi) >> 1; if (arr[mid] <= v) lo2 = mid + 1; else hi = mid; }
        return arr.length ? 100 * (less + 0.5 * (lo2 - less)) / arr.length : 0;
      };
      return it => {
        let s = 0, w = 0;
        ws.forEach(([i, wt]) => {
          const v = it.m[i];
          if (!v) return;
          s += wt * (spec.pct ? pctile(sorted[i], v[2]) : v[2]);
          w += wt;
        });
        return w ? s / w : null;
      };
    }
    return () => null;
  };

  // Best first (dir -1) or worst first (dir 1). Ties: more mean stars, then file name; no score = always last.
  L.rank = function (items, spec, lower, dir) {
    dir = dir || -1;
    if (spec.type === 'name') return items.slice().sort((x, y) => -dir * L.natural(x.p, y.p));
    const sc = L.makeScorer(spec, items, lower);
    const keyed = items.map(it => ({it, s: sc(it)}));
    keyed.sort((x, y) => {
      if (x.s == null && y.s == null) return L.natural(x.it.p, y.it.p);
      if (x.s == null) return 1;
      if (y.s == null) return -1;
      return dir * (x.s - y.s) || dir * ((x.it.r || 0) - (y.it.r || 0)) || L.natural(x.it.p, y.it.p);
    });
    return keyed.map(k => k.it);
  };
  L.scoreLabel = function (spec, it, lower) {
    if (spec.type === 'overall') return it.o == null ? '' : it.o.toFixed(1);
    if (spec.type === 'stars') return it.r == null ? '' : '★ ' + it.r.toFixed(1);
    if (spec.type === 'model') { const v = it.m[spec.m]; return v ? v[1].toFixed(2) : ''; }
    return '';
  };

  // ---- comparing --------------------------------------------------------------------------------------------
  // tops: array of arrays of items (the top N of each ranker). Returns {path: how many lists contain it}.
  L.membership = function (tops) {
    const m = new Map();
    tops.forEach(list => list.forEach(it => m.set(it.p, (m.get(it.p) || 0) + 1)));
    return m;
  };
  L.sharedCount = (a, b) => { const s = new Set(a.map(i => i.p)); return b.filter(i => s.has(i.p)).length; };
  L.allShared = function (tops) {
    if (!tops.length) return 0;
    const m = L.membership(tops);
    let n = 0;
    m.forEach(c => { if (c === tops.length) n++; });
    return n;
  };
  L.rankMap = list => { const m = new Map(); list.forEach((it, i) => m.set(it.p, i + 1)); return m; };
  // photos in the top n of A or B, with their rank in each full ranking; d = rankA - rankB (> 0: better in B)
  L.delta = function (fullA, fullB, n) {
    const ra = L.rankMap(fullA), rb = L.rankMap(fullB), seen = new Set(), rows = [];
    fullA.slice(0, n).concat(fullB.slice(0, n)).forEach(it => {
      if (seen.has(it.p)) return;
      seen.add(it.p);
      const a = ra.get(it.p), b = rb.get(it.p);
      rows.push({it, ra: a, rb: b, d: a - b, inA: a <= n, inB: b <= n});
    });
    rows.sort((x, y) => Math.abs(y.d) - Math.abs(x.d) || x.ra - y.ra || L.natural(x.it.p, y.it.p));
    return rows;
  };

  // ---- export ---------------------------------------------------------------------------------------------
  L.csvCell = v => { const s = v == null ? '' : String(v); return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
  L.fullPath = function (rootDir, rel) {
    if (!rootDir) return rel;
    const sep = rootDir.indexOf('\\') >= 0 && rootDir.indexOf('/') < 0 ? '\\' : '/';
    return rootDir.replace(/[\\/]+$/, '') + sep + (sep === '\\' ? rel.replace(/\//g, '\\') : rel);
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = L;
  else root.IQA = L;
})(typeof window !== 'undefined' ? window : globalThis);
