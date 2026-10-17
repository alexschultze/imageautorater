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
console.log('ok');
