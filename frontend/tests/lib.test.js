import test from "node:test";
import assert from "node:assert/strict";
import { compact, changeText, direction, periodLabel, alertTag, plural, parseISO } from "../src/lib/format.js";
import { niceTicks, valueDomain, linePath, bandPath, nearestIndex, indexForDate, xTickIndexes } from "../src/lib/chart.js";
import { sectionProgress, easeInOut, remap, clamp01, watchSection } from "../src/lib/scroll.js";
import { readPref, writePref, resolveReduced } from "../src/lib/motion.js";
import { rng, samplePoints, brightPixels, makeParticles, particleAt, particleBudget, wrapWords } from "../src/lib/particles.js";
import { makeSparkles, sparkleAt, sparkleCount } from "../src/lib/sparkles.js";

test("compact numbers", () => {
  assert.equal(compact(1234567), "1.2M");
  assert.equal(compact(15300), "15k");
  assert.equal(compact(9500), "9.5k");
  assert.equal(compact(0.191, "pct"), "19.1%");
  assert.equal(compact(0.5, "pct"), "50%");
  assert.equal(compact(-2500000), "-2.5M");
  assert.equal(compact(null), "n/a");
  assert.equal(compact(NaN), "n/a");
  assert.equal(compact(7), "7");
});

test("change text and direction", () => {
  assert.equal(changeText(0.123), "+12.3%");
  assert.equal(changeText(-0.5), "-50.0%");
  assert.equal(changeText(-0.021, true), "-2.1 pts");
  assert.equal(changeText(undefined), "n/a");
  assert.equal(direction(0.2), "up");
  assert.equal(direction(-0.2), "down");
  assert.equal(direction(0.0001), "flat");
  assert.equal(direction(null), "none");
});

test("dates and labels", () => {
  assert.equal(parseISO("2024-03-01").getUTCMonth(), 2);
  assert.equal(parseISO("bad"), null);
  assert.equal(periodLabel("2024-03-01"), "Mar 2024");
  assert.equal(periodLabel("2024-03-04", "W"), "4 Mar 2024");
  assert.equal(periodLabel("2024-04-01", "Q"), "Q2 2024");
  assert.equal(periodLabel(""), "");
});

test("alert tag", () => {
  assert.equal(alertTag({ direction: "down", pct_vs_expected: -0.26 }), "FELL 26% VS EXPECTED");
  assert.equal(alertTag({ direction: "up", pct_vs_expected: 0.1 }), "ROSE 10% VS EXPECTED");
  assert.equal(alertTag({ direction: "down", pct_vs_expected: -0.021, is_points: true }), "FELL 2.1 PTS VS EXPECTED");
  assert.equal(alertTag(null), "");
  assert.equal(plural(1, "row"), "1 row");
  assert.equal(plural(1200, "row"), "1,200 rows");
});

test("nice ticks cover the range", () => {
  const t = niceTicks(0, 87);
  assert.ok(t[0] <= 0 && t[t.length - 1] >= 87);
  assert.deepEqual(t, [0, 25, 50, 75, 100]);
  const flat = niceTicks(5, 5);
  assert.ok(flat.length >= 2 && flat[0] < 5 && flat[flat.length - 1] > 5);
  assert.deepEqual(niceTicks(null, 3), []);
  for (const [a, b] of [[0.2, 0.31], [-50, 120], [1e6, 3.4e6]]) {
    const k = niceTicks(a, b);
    assert.ok(k[0] <= a && k[k.length - 1] >= b, `${a}-${b}`);
  }
});

test("domain, paths, hover", () => {
  assert.deepEqual(valueDomain([{ value: 5, lower: 3, upper: 9 }, { value: 7 }]), [3, 9]);
  assert.equal(valueDomain([]), null);
  assert.equal(linePath([0, 10, 20], [5, 6, 7]), "M0.0,5.0L10.0,6.0L20.0,7.0");
  assert.equal(linePath([0, 10, 20], [5, null, 7]), "M0.0,5.0M20.0,7.0");
  const b = bandPath([0, 10], [1, 1], [5, 5]);
  assert.ok(b.startsWith("M0.0,5.0") && b.endsWith("Z"));
  assert.equal(bandPath([0], [1], [2]), "");
  assert.equal(nearestIndex([0, 10, 20], 14), 1);
  assert.equal(nearestIndex([], 3), -1);
});

test("dates map to points and ticks", () => {
  const pts = ["2022-01-01", "2022-02-01", "2022-03-01"].map((p) => ({ period: p }));
  assert.equal(indexForDate(pts, "2022-02-15"), 1);
  assert.equal(indexForDate(pts, "2021-01-01"), 0);
  assert.equal(indexForDate(pts, "2030-01-01"), 2);
  assert.equal(indexForDate([], "2022-01-01"), -1);
  const many = Array.from({ length: 36 }, (_, i) => ({ period: new Date(Date.UTC(2022, i, 1)).toISOString().slice(0, 10) }));
  assert.deepEqual(xTickIndexes(many), [0, 12, 24]);
  assert.ok(xTickIndexes(many.slice(2, 9)).length <= 7);
  assert.deepEqual(xTickIndexes([]), []);
});

test("scroll progress", () => {
  assert.equal(sectionProgress(0, 2000, 800), 0);
  assert.equal(sectionProgress(-600, 2000, 800), 0.5);
  assert.equal(sectionProgress(-5000, 2000, 800), 1);
  assert.equal(sectionProgress(100, 2000, 800), 0);
  assert.equal(sectionProgress(0, 500, 800), 0);
  assert.equal(easeInOut(0), 0);
  assert.equal(easeInOut(1), 1);
  assert.equal(easeInOut(0.5), 0.5);
  assert.equal(remap(0.5, 0.25, 0.75), 0.5);
  assert.equal(remap(0.1, 0.25, 0.75), 0);
  assert.equal(remap(0.9, 0.25, 0.75), 1);
  assert.equal(clamp01(-1), 0);
});

test("watchSection reports progress once per frame and can stop", () => {
  const listeners = {};
  let rafCb = null;
  const env = {
    innerHeight: 800,
    addEventListener: (n, f) => (listeners[n] = f),
    removeEventListener: (n) => delete listeners[n],
    requestAnimationFrame: (f) => (rafCb = f),
  };
  let top = 0;
  const el = { getBoundingClientRect: () => ({ top, height: 2000 }) };
  const seen = [];
  const stop = watchSection(el, (p) => seen.push(p), env);
  rafCb(); rafCb = null;
  top = -600;
  listeners.scroll(); listeners.scroll();     // two events, one frame
  rafCb();
  assert.deepEqual(seen, [0, 0.5]);
  stop();
  assert.equal(Object.keys(listeners).length, 0);
});

test("motion preference", () => {
  const mem = new Map();
  const storage = { getItem: (k) => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, v), removeItem: (k) => mem.delete(k) };
  assert.equal(readPref(storage), null);
  writePref(storage, true);
  assert.equal(readPref(storage), true);
  writePref(storage, false);
  assert.equal(readPref(storage), false);
  writePref(storage, null);
  assert.equal(readPref(storage), null);
  assert.equal(resolveReduced(null, true), true);
  assert.equal(resolveReduced(false, true), false);
  assert.equal(resolveReduced(true, false), true);
  const broken = { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); } };
  assert.equal(readPref(broken), null);
  writePref(broken, true);                      // must not throw
});

test("particles are repeatable and move from formed to scattered", () => {
  const a = rng(5), b = rng(5);
  assert.equal(a(), b());
  const flat = [];
  for (let i = 0; i < 100; i++) flat.push(i, i * 2);
  assert.equal(samplePoints(flat, 30).length, 30);
  assert.equal(samplePoints(flat, 500).length, 100);
  const data = new Uint8ClampedArray(4 * 4 * 4);
  data[(1 * 4 + 1) * 4 + 3] = 255;
  assert.deepEqual(brightPixels(data, 4, 4, 1), [1, 1]);
  const ps = makeParticles([[100, 50], [120, 50]], 800, 600);
  assert.deepEqual(makeParticles([[100, 50], [120, 50]], 800, 600), ps);
  const start = particleAt(ps[0], 0, 0);
  assert.ok(Math.abs(start.x - 100) < 3 && Math.abs(start.y - 50) < 3 && start.alpha === 1);
  const end = particleAt(ps[0], 1, 0);
  assert.ok(Math.abs(end.x - ps[0].sx) < 1e-6 && end.alpha < 0.2);
  assert.ok(particleBudget(400) < particleBudget(1400));
  assert.ok(particleBudget(1400, true) < particleBudget(1400));
});

test("wrapWords", () => {
  const measure = (s) => s.length * 10;
  assert.deepEqual(wrapWords("Understand What Changed. Discover Why.", 160, measure), ["Understand What", "Changed.", "Discover Why."]);
  assert.deepEqual(wrapWords("", 100, measure), []);
  assert.deepEqual(wrapWords("Supercalifragilistic", 50, measure), ["Supercalifragilistic"]);
});

test("sparkles", () => {
  const s = makeSparkles(10);
  assert.equal(s.length, 10);
  assert.deepEqual(makeSparkles(10), s);
  for (const t of [0, 5, 1000]) {
    const p = sparkleAt(s[0], t, 3);
    assert.ok(p.y >= 0 && p.y < 1 && p.alpha > 0 && p.alpha <= 1);
  }
  assert.ok(sparkleCount(400) < sparkleCount(1400));
});
