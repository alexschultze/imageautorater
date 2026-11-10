// Tests of the pure functions of the gallery page. Run by tests/test_cli_and_gallery.py:  node gallery_logic_test.js <logic.js>
const assert = require('assert');
const L = require(require('path').resolve(process.argv[2]));

const it = (p, o, r, scores) => ({p, f: p.includes('/') ? p.slice(0, p.lastIndexOf('/')) : '', o, r, m: scores, g: [], n: []});
// two models: model 0 higher = better, model 1 (like NIQE) lower raw score = better
const lower = [false, true];
const A = it('2023/a.jpg', 80, 4, [[4, 80, 80], [4, 5, 75]]);
const B = it('2023/b.jpg', 60, 3, [[3, 60, 60], [3, 2, 90]]);
const C = it('2023/sub/c.jpg', 40, 2, [[2, 40, 40], [2, 12, 40]]);
const D = it('2024/d.jpg', 20, 1, [[1, 20, 20], null]);
const items = [A, B, C, D];
const paths = list => list.map(i => i.p);

// folders: selecting a folder selects everything below it
const tree = L.buildTree({'2023': 2, '2023/sub': 1, '2024': 1, '': 3});
assert.deepStrictEqual(tree.paths, ['', '2023', '2023/sub', '2024']);
assert.strictEqual(tree.total['2023'], 3);
const sel = new Set();
L.setTick(tree, sel, '2023', true);
assert.deepStrictEqual([...sel].sort(), ['2023', '2023/sub']);
assert.strictEqual(L.tickState(tree, sel, '2023'), 'all');
assert.strictEqual(L.tickState(tree, sel, '2024'), 'none');
L.setTick(tree, sel, '2023/sub', false);
assert.strictEqual(L.tickState(tree, sel, '2023'), 'some');
// the URL form round-trips, including an excluded subfolder and a re-ticked folder below it
const all = new Set(tree.paths);
L.setTick(tree, all, '2023', false); L.setTick(tree, all, '2023/sub', true);
assert.deepStrictEqual([...L.decodeSel(tree, L.encodeSel(tree, all))].sort(), [...all].sort());
assert.deepStrictEqual([...L.decodeSel(tree, L.encodeSel(tree, sel))].sort(), [...sel].sort());
assert.strictEqual(L.groupKey('2023/sub/x', 1), '2023');
assert.strictEqual(L.groupKey('2023/sub/x', 2), '2023/sub');
assert.strictEqual(L.groupKey('2023', 2), '2023');
assert.strictEqual(L.groupKey('', 1), '');
assert.strictEqual(L.groupKey('2023/x', 0), '');

// rankings
assert.deepStrictEqual(paths(L.rank(items, {type: 'overall'}, lower, -1)), ['2023/a.jpg', '2023/b.jpg', '2023/sub/c.jpg', '2024/d.jpg']);
assert.deepStrictEqual(paths(L.rank(items, {type: 'overall'}, lower, 1)), ['2024/d.jpg', '2023/sub/c.jpg', '2023/b.jpg', '2023/a.jpg']);
// lower-is-better model: lowest raw score first; a photo without that model is always last
assert.deepStrictEqual(paths(L.rank(items, {type: 'model', m: 1}, lower, -1)), ['2023/b.jpg', '2023/a.jpg', '2023/sub/c.jpg', '2024/d.jpg']);
assert.deepStrictEqual(paths(L.rank(items, {type: 'model', m: 1}, lower, 1)).slice(0, 1), ['2023/sub/c.jpg']);
assert.strictEqual(paths(L.rank(items, {type: 'model', m: 1}, lower, 1)).pop(), '2024/d.jpg');
assert.deepStrictEqual(paths(L.rank(items, {type: 'name'}, lower, -1))[0], '2023/a.jpg');
// blend: only model 1 weighted = same order as model 1; unnormalised 50/50 of the 0-100 values
assert.deepStrictEqual(paths(L.rank(items, {type: 'blend', w: {1: 3}, pct: false}, lower, -1)), paths(L.rank(items, {type: 'model', m: 1}, lower, -1)));
const sc = L.makeScorer({type: 'blend', w: {0: 1, 1: 1}, pct: false}, items, lower);
assert.strictEqual(sc(A), 77.5);
assert.strictEqual(sc(D), 20);                      // a missing model does not count as zero
assert.strictEqual(L.makeScorer({type: 'blend', w: {}, pct: true}, items, lower)(A), null);
// percentile: best of three = (2 + 0.5) / 3 * 100
const pc = L.makeScorer({type: 'blend', w: {0: 1}, pct: true}, items.slice(0, 3), lower);
assert.ok(Math.abs(pc(A) - 100 * 2.5 / 3) < 1e-9 && Math.abs(pc(C) - 100 * 0.5 / 3) < 1e-9);
// ties are broken by stars, then name
const t1 = it('x/1.jpg', 50, 2, [null, null]), t2 = it('x/2.jpg', 50, 3, [null, null]);
assert.deepStrictEqual(paths(L.rank([t1, t2], {type: 'overall'}, lower, -1)), ['x/2.jpg', 'x/1.jpg']);

// comparing
const topO = L.rank(items, {type: 'overall'}, lower, -1).slice(0, 2), topN = L.rank(items, {type: 'model', m: 1}, lower, -1).slice(0, 2);
assert.strictEqual(L.sharedCount(topO, topN), 2);
assert.strictEqual(L.allShared([topO, topN]), 2);
assert.strictEqual(L.allShared([topO, items.slice(2)]), 0);
assert.strictEqual(L.membership([topO, topN]).get('2023/a.jpg'), 2);
const rows = L.delta(L.rank(items, {type: 'overall'}, lower, -1), L.rank(items, {type: 'model', m: 1}, lower, -1), 2);
assert.strictEqual(rows.length, 2);
assert.ok(rows.every(r => r.inA && r.inB));                           // the same two photos are in both tops
const wide = L.delta(L.rank(items, {type: 'overall'}, lower, -1), L.rank(items, {type: 'model', m: 1}, lower, -1), 1);
assert.deepStrictEqual(wide.map(r => [r.it.p, r.ra, r.rb, r.d, r.inA, r.inB]), [['2023/a.jpg', 1, 2, -1, true, false], ['2023/b.jpg', 2, 1, 1, false, true]]);

// export helpers
assert.strictEqual(L.csvCell('a,b'), '"a,b"');
assert.strictEqual(L.csvCell('say "x"'), '"say ""x"""');
assert.strictEqual(L.fullPath('D:\\Photos', '2023/a.jpg'), 'D:\\Photos\\2023\\a.jpg');
assert.strictEqual(L.fullPath('/mnt/p/', '2023/a.jpg'), '/mnt/p/2023/a.jpg');

// people filter
const face = (fc, n) => ({p: 'x.jpg', f: '', fc, n: n || []});
assert.ok(L.peopleOk(face(null), ''));
assert.ok(L.peopleOk(face(2), 'with') && !L.peopleOk(face(0), 'with') && !L.peopleOk(face(null), 'with'));
assert.ok(L.peopleOk(face(1, ['Anna']), 'named') && !L.peopleOk(face(1), 'named'));
assert.ok(L.peopleOk(face(0), 'without') && !L.peopleOk(face(null), 'without') && !L.peopleOk(face(1), 'without'));

// copy plan and sub folder names
assert.deepStrictEqual(L.cleanSub('_gallery\\selection/ best '), ['_gallery', 'selection', 'best']);
assert.strictEqual(L.cleanSub('../x'), null);
assert.strictEqual(L.cleanSub('  '), null);
assert.strictEqual(L.cleanSub('a:b'), null);
assert.deepStrictEqual(L.planCopy(['2023/a.jpg', '2024/a.jpg', '2024/A.JPG', '2024/noext'], 'keep').map(x => x.dst), ['2023/a.jpg', '2024/a.jpg', '2024/A.JPG', '2024/noext']);
assert.deepStrictEqual(L.planCopy(['2023/a.jpg', '2024/a.jpg', '2024/A.JPG', '2024/noext'], 'flat').map(x => x.dst), ['a.jpg', 'a_2.jpg', 'A_3.JPG', 'noext']);

// copying with the File System Access API, tested with an in-memory stand-in for the folder handles
function fakeDir(name) {
  const e = new Map();
  const notFound = () => Object.assign(new Error('not found'), {name: 'NotFoundError'});
  return {
    kind: 'directory', name, e,
    async getDirectoryHandle(n, o) { if (!e.has(n)) { if (!(o && o.create)) throw notFound(); e.set(n, fakeDir(n)); } return e.get(n); },
    async getFileHandle(n, o) {
      if (!e.has(n)) { if (!(o && o.create)) throw notFound(); e.set(n, fakeFile(n, '')); }
      return e.get(n);
    },
  };
}
function fakeFile(name, text) {
  const f = {kind: 'file', name, text, async getFile() { return new Blob([f.text]); }, async createWritable() {
    const chunks = [];
    return new WritableStream({write(c) { chunks.push(c); }, async close() { f.text = Buffer.concat(chunks).toString(); }});
  }};
  return f;
}
(async () => {
  const root = fakeDir('Photos');
  const y = await root.getDirectoryHandle('2023', {create: true});
  y.e.set('a.jpg', fakeFile('a.jpg', 'AAAA'));
  (await y.getDirectoryHandle('sub', {create: true})).e.set('b.jpg', fakeFile('b.jpg', 'BB'));
  const plan = L.planCopy(['2023/a.jpg', '2023/sub/b.jpg', '2023/missing.jpg'], 'keep');
  const prog = [];
  let res = await L.copyFiles(root, ['_gallery', 'pick'], plan, (n, t) => prog.push(n + '/' + t));
  assert.deepStrictEqual([res.copied, res.skipped, res.failed.length], [2, 0, 1]);
  assert.strictEqual(res.failed[0].rel, '2023/missing.jpg');
  assert.deepStrictEqual(prog, ['1/3', '2/3', '3/3']);
  const out = await (await (await (await root.getDirectoryHandle('_gallery')).getDirectoryHandle('pick')).getDirectoryHandle('2023')).getFileHandle('a.jpg');
  assert.strictEqual(out.text, 'AAAA');
  res = await L.copyFiles(root, ['_gallery', 'pick'], plan, null);     // repeated: same size = skipped, nothing rewritten
  assert.deepStrictEqual([res.copied, res.skipped, res.failed.length], [0, 2, 1]);
  const flat = await L.copyFiles(root, ['flat'], L.planCopy(['2023/a.jpg', '2023/sub/b.jpg'], 'flat'), null);
  assert.strictEqual(flat.copied, 2);
  assert.strictEqual((await (await root.getDirectoryHandle('flat')).getFileHandle('b.jpg')).text, 'BB');
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
