const D = JSON.parse(document.getElementById('index').textContent);
const L = window.IQA;
const $ = id => document.getElementById(id);
window.__IQA_DATA = window.__IQA_DATA || {};
const tree = L.buildTree(D.folders);
const maxDepth = Math.max(0, ...tree.paths.map(L.depthOf));
const folderLabel = f => (f === '' ? '(top folder)' : f);
const topKey = f => L.parts(f)[0] || '';
document.title = 'Photo ranking – ' + D.title;
$('title').firstChild.textContent = D.title + ' ';

// ---- state (kept in the URL hash, so a view can be bookmarked) --------------------------------------------------------
const S = {
  view: 'list', rank: 'overall', dir: -1, cmp: ['overall', 'stars'], da: 'overall', db: 'stars', dshow: 'all',
  n: D.top || 50, depth: maxDepth >= 1 ? 1 : 0, minstars: 0, person: '', tag: '', q: '',
  sel: new Set(tree.paths), blend: {w: {}, pct: true}, open: new Set(),
};
let saved = {};   // name -> {w: {model name: weight}, pct}
try { saved = JSON.parse(localStorage.getItem('iqa_blends') || '{}') || {}; } catch (e) { saved = {}; }
const persistBlends = () => { try { localStorage.setItem('iqa_blends', JSON.stringify(saved)); } catch (e) { /* private window */ } };

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  const num = (k, d) => (h.has(k) && !isNaN(+h.get(k)) ? +h.get(k) : d);
  if (['list', 'compare', 'delta'].includes(h.get('v'))) S.view = h.get('v');
  if (h.get('r')) S.rank = h.get('r');
  S.dir = num('o', S.dir) === 1 ? 1 : -1;
  if (h.get('c')) S.cmp = h.get('c').split(',').filter(Boolean).slice(0, 4);
  S.da = h.get('da') || S.da; S.db = h.get('db') || S.db; S.dshow = h.get('ds') || S.dshow;
  S.n = Math.max(1, num('n', S.n)); S.depth = Math.min(maxDepth, Math.max(0, num('g', S.depth)));
  S.minstars = num('ms', 0); S.person = h.get('pe') || ''; S.tag = h.get('tg') || ''; S.q = h.get('q') || '';
  if (h.has('s')) S.sel = L.decodeSel(tree, h.get('s'));
  if (h.has('bw')) {
    S.blend.w = {};
    h.get('bw').split(',').forEach(p => { const [m, w] = p.split(':'), i = D.models.indexOf(m); if (i >= 0 && +w > 0) S.blend.w[i] = +w; });
  }
  if (h.has('bp')) S.blend.pct = h.get('bp') === '1';
}
function writeHash() {
  const h = new URLSearchParams();
  h.set('v', S.view); h.set('r', S.rank); h.set('o', S.dir); h.set('c', S.cmp.join(','));
  h.set('da', S.da); h.set('db', S.db); h.set('ds', S.dshow); h.set('n', S.n); h.set('g', S.depth);
  if (S.minstars) h.set('ms', S.minstars);
  if (S.person) h.set('pe', S.person);
  if (S.tag) h.set('tg', S.tag);
  if (S.q) h.set('q', S.q);
  if (tree.paths.some(p => !S.sel.has(p))) h.set('s', L.encodeSel(tree, S.sel));
  const bw = Object.entries(S.blend.w).map(([i, w]) => D.models[i] + ':' + w).join(',');
  if (bw) h.set('bw', bw);
  h.set('bp', S.blend.pct ? 1 : 0);
  try { history.replaceState(null, '', '#' + h.toString()); } catch (e) { /* file:// in some browsers */ }
}

// ---- rankers ----------------------------------------------------------------------------------------------------
const rankerIds = () => ['overall', 'stars', ...D.models.map((_, i) => 'm:' + i), 'blend', ...Object.keys(saved).sort().map(n => 'sb:' + n), 'name'];
function labelOf(id) {
  if (id === 'overall') return 'Overall (mean of all models)';
  if (id === 'stars') return 'Mean stars';
  if (id === 'name') return 'File name';
  if (id === 'blend') return 'Custom blend';
  if (id.startsWith('m:')) return D.models[+id.slice(2)] || id;
  if (id.startsWith('sb:')) return 'Blend: ' + id.slice(3);
  return id;
}
function specOf(id) {
  if (id === 'overall' || id === 'stars' || id === 'name') return {type: id};
  if (id.startsWith('m:')) return {type: 'model', m: +id.slice(2)};
  if (id === 'blend') return {type: 'blend', w: S.blend.w, pct: S.blend.pct};
  if (id.startsWith('sb:') && saved[id.slice(3)]) {
    const b = saved[id.slice(3)], w = {};
    Object.entries(b.w).forEach(([m, v]) => { const i = D.models.indexOf(m); if (i >= 0) w[i] = v; });
    return {type: 'blend', w, pct: b.pct};
  }
  return {type: 'overall'};
}
const validId = id => rankerIds().includes(id);
function opt(sel, value, text) { const o = document.createElement('option'); o.value = value; o.textContent = text; sel.appendChild(o); return o; }
function fillRankers(sel, value, none) {
  sel.textContent = '';
  if (none) opt(sel, '', '– none –');
  rankerIds().forEach(id => opt(sel, id, labelOf(id)));
  sel.value = value;
}

// ---- controls ---------------------------------------------------------------------------------------------------
function buildControls() {
  fillRankers($('rank'), validId(S.rank) ? S.rank : 'overall');
  $('dir').value = S.dir; $('topn').value = S.n; $('minstars').value = S.minstars; $('q').value = S.q;
  const dp = $('depth'); dp.textContent = '';
  opt(dp, 0, 'all selected, merged');
  for (let d = 1; d <= maxDepth; d++) opt(dp, d, d === 1 ? 'top-level folder (year)' : 'folder level ' + d);
  dp.value = S.depth;
  $('person').textContent = ''; opt($('person'), '', 'anyone'); D.people.forEach(p => opt($('person'), p, p)); $('person').value = S.person;
  $('tag').textContent = ''; opt($('tag'), '', 'any'); D.tags.forEach(t => opt($('tag'), t, t)); $('tag').value = S.tag;
  buildCompareControls();
  buildBlend();
}
function labeled(text, el) { const l = document.createElement('label'); l.append(text + ' ', el); return l; }
function buildCompareControls() {
  const c = $('cmpctl'); c.textContent = '';
  for (let k = 0; k < 4; k++) {
    const s = document.createElement('select');
    fillRankers(s, S.cmp[k] && validId(S.cmp[k]) ? S.cmp[k] : '', k >= 2);
    s.addEventListener('change', () => { S.cmp[k] = s.value; S.cmp = Array.from(S.cmp).filter((x, i) => x || i < 2).slice(0, 4); buildCompareControls(); render(); });
    c.appendChild(labeled('ABCD'[k], s));
  }
  const d = $('dltctl'); d.textContent = '';
  const a = document.createElement('select'), b = document.createElement('select'), sh = document.createElement('select');
  fillRankers(a, validId(S.da) ? S.da : 'overall'); fillRankers(b, validId(S.db) ? S.db : 'stars');
  [['all', 'all'], ['a', 'only in top of A'], ['b', 'only in top of B']].forEach(([v, t]) => opt(sh, v, t));
  sh.value = S.dshow;
  a.addEventListener('change', () => { S.da = a.value; render(); });
  b.addEventListener('change', () => { S.db = b.value; render(); });
  sh.addEventListener('change', () => { S.dshow = sh.value; render(); });
  d.append(labeled('From A', a), labeled('to B', b), labeled('Show', sh));
}
function setView(v) {
  S.view = v;
  document.querySelectorAll('#views button').forEach(b => b.classList.toggle('on', b.dataset.v === v));
  $('rankctl').hidden = v !== 'list'; $('cmpctl').hidden = v !== 'compare'; $('dltctl').hidden = v !== 'delta';
}

// ---- folder tree ------------------------------------------------------------------------------------------------
function buildTreeUi() {
  const root = $('tree'); root.textContent = '';
  const node = (p, parent, label) => {
    const kids = p === '' ? [] : tree.children[p];
    const wrap = document.createElement('div');
    const row = document.createElement('div'); row.className = 'node';
    const tw = document.createElement('span'); tw.className = 'tw';
    tw.textContent = kids.length ? (S.open.has(p) ? '▾' : '▸') : '';
    const cb = document.createElement('input'); cb.type = 'checkbox';
    const st = L.tickState(tree, S.sel, p);
    cb.checked = st === 'all'; cb.indeterminate = st === 'some';
    const nm = document.createElement('span'); nm.className = 'nm'; nm.textContent = label || p.split('/').pop(); nm.title = folderLabel(p);
    const ct = document.createElement('span'); ct.className = 'ct'; ct.textContent = p === '' ? D.folders[''] : tree.total[p];
    row.append(tw, cb, nm, ct); wrap.appendChild(row);
    const toggleOpen = () => { if (S.open.has(p)) S.open.delete(p); else S.open.add(p); buildTreeUi(); };
    tw.addEventListener('click', toggleOpen);
    nm.addEventListener('click', () => (kids.length ? toggleOpen() : cb.click()));
    cb.addEventListener('change', () => { L.setTick(tree, S.sel, p, L.tickState(tree, S.sel, p) !== 'all'); buildTreeUi(); render(); });
    if (kids.length && S.open.has(p)) {
      const k = document.createElement('div'); k.className = 'kids';
      kids.forEach(c => node(c, k)); wrap.appendChild(k);
    }
    parent.appendChild(wrap);
  };
  if (D.folders['']) node('', root, '(top folder)');
  tree.children[''].forEach(p => node(p, root));
}

// ---- blend editor -----------------------------------------------------------------------------------------------
let blendTimer = null;
function buildBlend() {
  const box = $('blend'); box.textContent = '';
  D.models.forEach((m, i) => {
    const row = document.createElement('div'); row.className = 'row';
    const r = document.createElement('input'); r.type = 'range'; r.min = 0; r.max = 10; r.step = 1; r.value = S.blend.w[i] || 0;
    const v = document.createElement('span'); v.textContent = r.value;
    r.addEventListener('input', () => {
      v.textContent = r.value;
      if (+r.value > 0) S.blend.w[i] = +r.value; else delete S.blend.w[i];
      clearTimeout(blendTimer); blendTimer = setTimeout(() => { if (usesBlend()) render(); }, 200);
    });
    const n = document.createElement('span'); n.textContent = m; n.title = D.lower[i] ? m + ' (lower raw score is better; handled)' : m;
    row.append(n, r, v); box.appendChild(row);
  });
  const pc = document.createElement('input'); pc.type = 'checkbox'; pc.checked = S.blend.pct;
  pc.addEventListener('change', () => { S.blend.pct = pc.checked; if (usesBlend()) render(); });
  const pl = document.createElement('label'); pl.append(pc, 'normalise within the selected photos');
  pl.title = 'Each model is converted to a percentile among the selected photos first, so no model counts for more only because its scores are spread wider.';
  box.appendChild(pl);
  const bar = document.createElement('div'); bar.className = 'bar small';
  const use = document.createElement('button'); use.textContent = 'Rank by blend';
  use.addEventListener('click', () => {
    if (S.view === 'list') { S.rank = 'blend'; $('rank').value = 'blend'; }
    else if (S.view === 'compare') { S.cmp[S.cmp.length < 4 && !S.cmp.includes('blend') ? S.cmp.length : 1] = 'blend'; buildCompareControls(); }
    else { S.db = 'blend'; buildCompareControls(); }
    render();
  });
  const sv = document.createElement('button'); sv.textContent = 'Save as…';
  sv.addEventListener('click', () => {
    const name = (prompt('Name for this blend') || '').trim();
    if (!name) return;
    const w = {}; Object.entries(S.blend.w).forEach(([i, x]) => { w[D.models[i]] = x; });
    saved[name] = {w, pct: S.blend.pct}; persistBlends(); refreshRankers();
  });
  const del = document.createElement('button'); del.textContent = 'Delete saved…';
  del.addEventListener('click', () => {
    const name = (prompt('Delete which saved blend?\n' + Object.keys(saved).join(', ')) || '').trim();
    if (saved[name]) { delete saved[name]; persistBlends(); refreshRankers(); }
  });
  bar.append(use, sv, del); box.appendChild(bar);
}
const usesBlend = () => {
  const b = id => id === 'blend';
  return S.view === 'list' ? b(S.rank) : S.view === 'compare' ? S.cmp.some(b) : b(S.da) || b(S.db);
};
function refreshRankers() { fillRankers($('rank'), validId(S.rank) ? S.rank : 'overall'); buildCompareControls(); render(); }

// ---- data -------------------------------------------------------------------------------------------------------
const loading = {};
function loadGroup(g) {
  if (window.__IQA_DATA[g.k]) return Promise.resolve();
  if (!loading[g.k]) {
    loading[g.k] = new Promise((res, rej) => {
      const s = document.createElement('script'); s.src = g.file; s.onload = res; s.onerror = () => rej(new Error('cannot load ' + g.file));
      document.head.appendChild(s);
    });
  }
  return loading[g.k];
}
const neededGroups = () => D.groups.filter(g => [...S.sel].some(f => topKey(f) === g.k));
function selectedItems() {
  const min = +S.minstars, q = S.q.trim().toLowerCase(), out = [];
  neededGroups().forEach(g => (window.__IQA_DATA[g.k] || []).forEach(it => {
    if (!S.sel.has(it.f)) return;
    if (min && !(it.r != null && it.r >= min - 0.5)) return;
    if (S.person && !it.n.includes(S.person)) return;
    if (S.tag && !it.g.includes(S.tag)) return;
    if (q && !it.p.toLowerCase().includes(q)) return;
    out.push(it);
  }));
  return out;
}

// ---- cards ------------------------------------------------------------------------------------------------------
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
function card(it, rank, o) {
  const c = el('div', 'card' + (o.mark ? ' ' + o.mark : '')); c.title = it.p; c.dataset.p = it.p;
  const t = el('div', 'thumb');
  if (it.t) { const im = document.createElement('img'); im.loading = 'lazy'; im.decoding = 'async'; im.src = it.t; im.alt = ''; t.appendChild(im); }
  t.appendChild(el('span', 'rank', '#' + rank));
  if (o.score) t.appendChild(el('span', 'score', o.score));
  if (o.mv) t.appendChild(el('span', 'mv ' + (o.mv.d > 0 ? 'up' : o.mv.d < 0 ? 'down' : ''), o.mv.text));
  else if (o.mark) t.appendChild(el('span', 'mark ' + o.mark, o.mark === 'all' ? 'in all' : 'in some'));
  const open = el('a', 'open', 'open'); open.href = it.u; open.target = '_blank'; open.rel = 'noopener'; open.title = 'Open the original file';
  open.addEventListener('click', e => e.stopPropagation());
  t.appendChild(open);
  c.appendChild(t);
  const m = el('div', 'meta');
  m.appendChild(el('div', 'name', it.p.split('/').pop()));
  if (!o.compact) {
    if (it.r != null) { const k = Math.round(it.r); m.appendChild(el('div', 'stars', '★'.repeat(k) + '☆'.repeat(5 - k))); }
    const chips = el('div', 'chips');
    it.n.forEach(p => chips.appendChild(el('span', 'chip person', p)));
    it.g.slice(0, 3).forEach(g => chips.appendChild(el('span', 'chip', g)));
    m.appendChild(chips);
  }
  c.appendChild(m);
  c.addEventListener('click', () => openLb(o.list, rank - 1, o.ranks));
  return c;
}
let autoGrow = null;
// a grid that shows `page` more cards each time it is asked to grow (button, or scrolling when `observe`)
function section(parent, count, make, observe) {
  const grid = el('div', 'grid'); parent.appendChild(grid);
  const btn = el('button', 'more'); parent.appendChild(btn);
  let shown = 0;
  const grow = () => {
    const end = Math.min(count, shown + S.n);
    for (; shown < end; shown++) grid.appendChild(make(shown));
    btn.style.display = shown < count ? '' : 'none';
    btn.textContent = 'Show ' + Math.min(S.n, count - shown) + ' more (' + (count - shown) + ' left)';
  };
  btn.addEventListener('click', grow);
  grow();
  if (observe) {
    autoGrow = () => { if (btn.style.display !== 'none' && innerHeight + scrollY > document.documentElement.scrollHeight - 900) { grow(); setTimeout(() => autoGrow && autoGrow(), 0); } };
    autoGrow();
  }
}
addEventListener('scroll', () => autoGrow && autoGrow(), {passive: true});

// ---- views ------------------------------------------------------------------------------------------------------
let shown = [];   // what is on screen, for the export: [{group, ranker, items}]
function groupsOf(items) {
  const m = new Map();
  items.forEach(it => { const k = L.groupKey(it.f, S.depth); if (!m.has(k)) m.set(k, []); m.get(k).push(it); });
  return [...m.entries()].sort((a, b) => L.natural(a[0], b[0]));
}
const groupTitle = k => (k === '' ? (S.depth === 0 ? 'All selected folders' : '(top folder)') : k);
function groupHead(main, key, n, extra) {
  const g = el('div', 'group'), h = el('h2', null, groupTitle(key));
  h.appendChild(el('span', null, n + ' photos' + (extra ? ' · ' + extra : '')));
  g.appendChild(h); main.appendChild(g);
  return g;
}

function viewList(main, groups) {
  const id = validId(S.rank) ? S.rank : 'overall', spec = specOf(id);
  groups.forEach(([key, items]) => {
    const g = groupHead(main, key, items.length), list = L.rank(items, spec, D.lower, S.dir);
    shown.push({group: groupTitle(key), ranker: labelOf(id), items: list.slice(0, S.n)});
    section(g, list.length, i => card(list[i], i + 1, {list, score: L.scoreLabel(spec, list[i], D.lower), ranks: it => [[labelOf(id), list.indexOf(it) + 1]]}), groups.length === 1);
  });
}

function viewCompare(main, groups) {
  const ids = S.cmp.filter(validId);
  if (ids.length < 2) { main.appendChild(el('div', 'empty', 'Pick at least two rankings (A and B).')); return; }
  const lg = el('p', 'legend', 'Top ' + S.n + ' of each ranking. Marker colour: ');
  [['all', 'in every list'], ['some', 'in some lists']].forEach(([c, t]) => { const i = el('i'); i.style.background = 'var(--' + c + ')'; lg.append(i, t); });
  main.appendChild(lg);
  groups.forEach(([key, items]) => {
    const fulls = ids.map(id => L.rank(items, specOf(id), D.lower, -1)), tops = fulls.map(f => f.slice(0, S.n));
    const mem = L.membership(tops), maps = fulls.map(L.rankMap);
    const g = groupHead(main, key, items.length, 'in all ' + ids.length + ' lists: ' + L.allShared(tops));
    const cols = el('div', 'cols'); cols.style.gridTemplateColumns = 'repeat(' + ids.length + ',minmax(0,1fr))'; g.appendChild(cols);
    const ranksOf = it => ids.map((id, k) => [labelOf(id), maps[k].get(it.p)]);
    ids.forEach((id, k) => {
      const col = el('div', 'col'); col.appendChild(el('h3', null, 'ABCD'[k] + ' · ' + labelOf(id)));
      col.appendChild(el('div', 'stat', ids.map((o, j) => j === k ? null : 'shared with ' + 'ABCD'[j] + ': ' + L.sharedCount(tops[k], tops[j])).filter(Boolean).join(' · ')));
      const grid = el('div', 'grid'); col.appendChild(grid);
      tops[k].forEach((it, i) => {
        const n = mem.get(it.p);
        grid.appendChild(card(it, i + 1, {list: tops[k], compact: true, ranks: ranksOf, mark: ids.length > 1 ? (n === ids.length ? 'all' : n > 1 ? 'some' : '') : ''}));
      });
      cols.appendChild(col);
      shown.push({group: groupTitle(key), ranker: labelOf(id), items: tops[k]});
    });
    g.addEventListener('mouseover', e => {
      const c = e.target.closest && e.target.closest('.card'); if (!c) return;
      g.querySelectorAll('.card.hl').forEach(x => x.classList.remove('hl'));
      g.querySelectorAll('.card[data-p="' + CSS.escape(c.dataset.p) + '"]').forEach(x => x.classList.add('hl'));
    });
    g.addEventListener('mouseleave', () => g.querySelectorAll('.card.hl').forEach(x => x.classList.remove('hl')));
  });
}

function viewDelta(main, groups) {
  const ia = validId(S.da) ? S.da : 'overall', ib = validId(S.db) ? S.db : 'stars';
  main.appendChild(el('p', 'legend', 'Photos in the top ' + S.n + ' of A or B, biggest change in rank first. Green: ranked higher by B than by A, red: lower.'));
  groups.forEach(([key, items]) => {
    const fa = L.rank(items, specOf(ia), D.lower, -1), fb = L.rank(items, specOf(ib), D.lower, -1);
    let rows = L.delta(fa, fb, S.n);
    const both = rows.filter(r => r.inA && r.inB).length;
    if (S.dshow === 'a') rows = rows.filter(r => r.inA && !r.inB);
    if (S.dshow === 'b') rows = rows.filter(r => r.inB && !r.inA);
    const g = groupHead(main, key, items.length, 'in both tops: ' + both + ' of ' + S.n);
    const list = rows.map(r => r.it);
    shown.push({group: groupTitle(key), ranker: labelOf(ia) + ' -> ' + labelOf(ib), items: list});
    if (!rows.length) { g.appendChild(el('div', 'empty', 'Nothing to show for this choice.')); return; }
    const ranks = it => { const r = rows.find(x => x.it === it); return [['A ' + labelOf(ia), r.ra], ['B ' + labelOf(ib), r.rb]]; };
    section(g, rows.length, i => {
      const r = rows[i];
      return card(r.it, r.ra, {list, ranks, compact: true, mv: {d: r.d, text: 'B #' + r.rb + ' (' + (r.d > 0 ? '+' : '') + r.d + ')'}});
    }, false);
  });
}

let token = 0;
async function render() {
  autoGrow = null; shown = [];
  writeHash();
  const my = ++token, main = $('main');
  main.textContent = '';
  const need = neededGroups();
  if (!need.length) { $('count').textContent = ''; main.appendChild(el('div', 'empty', 'No folder selected.')); return; }
  if (need.some(g => !window.__IQA_DATA[g.k])) main.appendChild(el('div', 'busy', 'Loading…'));
  try { await Promise.all(need.map(loadGroup)); } catch (e) {
    if (my === token) { main.textContent = ''; main.appendChild(el('div', 'empty', e.message + '. Keep the gallery_data folder next to this page.')); }
    return;
  }
  if (my !== token) return;
  main.textContent = '';
  const items = selectedItems();
  $('count').textContent = items.length + ' photos selected' + (D.mode ? ' · ' + D.mode + ' scoring' : '');
  document.documentElement.style.setProperty('--hh', document.querySelector('header').offsetHeight + 'px');
  if (!items.length) { main.appendChild(el('div', 'empty', 'No photos match these filters.')); return; }
  const groups = groupsOf(items);
  if (S.view === 'list') viewList(main, groups);
  else if (S.view === 'compare') viewCompare(main, groups);
  else viewDelta(main, groups);
}

// ---- large view -------------------------------------------------------------------------------------------------
let lbList = [], lbPos = 0, lbRanks = null;
function openLb(list, pos, ranks) { lbList = list; lbPos = pos; lbRanks = ranks; showLb(); $('lb').classList.add('on'); }
function copyText(text, btn) {
  const done = () => { const o = btn.textContent; btn.textContent = 'Copied'; setTimeout(() => { btn.textContent = o; }, 1200); };
  const fallback = () => {
    const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); done(); } catch (e) { prompt('Copy the path:', text); }
    ta.remove();
  };
  try { navigator.clipboard.writeText(text).then(done, fallback); } catch (e) { fallback(); }
}
function showLb() {
  const it = lbList[lbPos];
  const box = $('lbimg'); box.textContent = '';
  const im = document.createElement('img'); im.src = it.b ? it.u : it.t || it.u; box.appendChild(im);
  const side = $('lbside'); side.textContent = '';
  side.appendChild(el('h3', null, it.p));
  side.appendChild(Object.assign(el('div', null, '#' + (lbPos + 1) + ' of ' + lbList.length + (it.o != null ? ' · overall ' + it.o.toFixed(1) : '') + (it.r != null ? ' · ★ ' + it.r.toFixed(1) : '')), {style: 'color:var(--mute)'}));
  const acts = el('div', 'acts');
  const a = el('a', null, 'Open original'); a.href = it.u; a.target = '_blank'; a.rel = 'noopener';
  const cp = el('button', null, 'Copy path'); cp.addEventListener('click', () => copyText(L.fullPath(D.root, it.p), cp));
  acts.append(a, cp); side.appendChild(acts);
  if (lbRanks) {
    const tb = el('table'); tb.innerHTML = '<tr><th>Ranking</th><th>Rank</th></tr>';
    lbRanks(it).forEach(([name, r]) => { const tr = el('tr'); tr.append(el('td', null, name), el('td', null, r == null ? '' : '#' + r)); tb.appendChild(tr); });
    side.appendChild(tb);
  }
  if (it.n.length || it.g.length) {
    const c = el('div', 'chips'); c.style.marginTop = '8px';
    it.n.forEach(p => c.appendChild(el('span', 'chip person', p)));
    it.g.forEach(g => c.appendChild(el('span', 'chip', g)));
    side.appendChild(c);
  }
  const tb = el('table'); tb.innerHTML = '<tr><th>Model</th><th>Stars</th><th>Score</th><th>0-100</th></tr>';
  D.models.map((m, i) => [m, it.m[i]]).filter(x => x[1]).sort((x, y) => y[1][2] - x[1][2]).forEach(([m, v]) => {
    const tr = el('tr'); [m, v[0], v[1].toFixed(2), v[2].toFixed(0)].forEach(x => tr.appendChild(el('td', null, x))); tb.appendChild(tr);
  });
  side.appendChild(tb);
}
function step(d) { const n = lbPos + d; if (n >= 0 && n < lbList.length) { lbPos = n; showLb(); } }
$('lbx').addEventListener('click', () => $('lb').classList.remove('on'));
$('lbimg').addEventListener('click', e => { if (e.target.id === 'lbimg') $('lb').classList.remove('on'); });
document.addEventListener('keydown', e => {
  if (!$('lb').classList.contains('on')) return;
  if (e.key === 'Escape') $('lb').classList.remove('on');
  else if (e.key === 'ArrowRight') step(1);
  else if (e.key === 'ArrowLeft') step(-1);
});

// ---- export -----------------------------------------------------------------------------------------------------
function download(name, text, type) {
  try {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], {type})); a.download = name; document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  } catch (e) { prompt('Copy:', text.slice(0, 2000)); }
}
$('exp-txt').addEventListener('click', () => {
  const seen = new Set(), lines = [];
  shown.forEach(s => s.items.forEach(it => { if (!seen.has(it.p)) { seen.add(it.p); lines.push(L.fullPath(D.root, it.p)); } }));
  download('photos.txt', lines.join('\n') + '\n', 'text/plain');
});
$('exp-csv').addEventListener('click', () => {
  const head = ['group', 'ranking', 'rank', 'path', 'overall', 'stars', ...D.models];
  const rows = [head];
  shown.forEach(s => s.items.forEach((it, i) => rows.push([s.group, s.ranker, i + 1, L.fullPath(D.root, it.p), it.o, it.r, ...D.models.map((m, k) => (it.m[k] ? it.m[k][1] : ''))])));
  download('ranking.csv', rows.map(r => r.map(L.csvCell).join(',')).join('\r\n') + '\r\n', 'text/csv');
});

// ---- wiring -----------------------------------------------------------------------------------------------------
document.querySelectorAll('#views button').forEach(b => b.addEventListener('click', () => { setView(b.dataset.v); render(); }));
$('rank').addEventListener('change', () => { S.rank = $('rank').value; render(); });
$('dir').addEventListener('change', () => { S.dir = +$('dir').value; render(); });
$('topn').addEventListener('change', () => { S.n = Math.max(1, Math.min(1000, +$('topn').value || 50)); render(); });
$('depth').addEventListener('change', () => { S.depth = +$('depth').value; render(); });
['minstars', 'person', 'tag'].forEach(id => $(id).addEventListener('change', () => { S[id] = id === 'minstars' ? +$(id).value : $(id).value; render(); }));
$('q').addEventListener('input', () => { S.q = $('q').value; render(); });
$('all').addEventListener('click', () => { S.sel = new Set(tree.paths); buildTreeUi(); render(); });
$('none').addEventListener('click', () => { S.sel = new Set(); buildTreeUi(); render(); });
$('expand').addEventListener('click', () => { S.open = new Set(tree.paths.filter(p => p && tree.children[p].length)); buildTreeUi(); });
$('collapse').addEventListener('click', () => { S.open = new Set(); buildTreeUi(); });

readHash();
buildControls();
setView(S.view);
buildTreeUi();
render();
