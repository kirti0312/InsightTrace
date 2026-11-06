// Particle headline maths. Pure functions: the canvas component only draws what these return.
import { clamp01, easeInOut } from "./scroll.js";

/** Small seeded random generator (mulberry32) so layouts are repeatable and testable. */
export function rng(seed = 1) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Pick up to `max` points from a filled-pixel list [x,y,x,y,...] (flat array), evenly thinned. */
export function samplePoints(flat, max) {
  const n = Math.floor(flat.length / 2);
  if (n <= max) return Array.from({ length: n }, (_, i) => [flat[2 * i], flat[2 * i + 1]]);
  const out = [];
  const step = n / max;
  for (let i = 0; i < max; i++) {
    const k = Math.floor(i * step);
    out.push([flat[2 * k], flat[2 * k + 1]]);
  }
  return out;
}

/** Read the bright pixels of an RGBA image into a flat [x,y,...] list (every `gap` pixels). */
export function brightPixels(data, width, height, gap = 3, threshold = 128) {
  const out = [];
  for (let y = 0; y < height; y += gap) {
    for (let x = 0; x < width; x += gap) {
      if (data[(y * width + x) * 4 + 3] > threshold) out.push(x, y);
    }
  }
  return out;
}

/** Give every target point a scatter destination and a personal delay, from a seed. */
export function makeParticles(targets, width, height, seed = 7) {
  const r = rng(seed);
  return targets.map(([tx, ty]) => {
    const angle = r() * Math.PI * 2;
    const dist = (0.25 + r() * 0.75) * Math.max(width, height) * 0.6;
    return {
      tx,
      ty,
      sx: tx + Math.cos(angle) * dist,           // where it flies to when scattered
      sy: ty + Math.sin(angle) * dist - height * 0.15,
      delay: r() * 0.35,                          // 0..0.35 of the scroll range
      size: 0.8 + r() * 1.4,
      hue: r(),                                   // 0 = violet, 1 = pink (picked by the drawing code)
      phase: r() * Math.PI * 2,
    };
  });
}

/**
 * Where is a particle at scroll progress p (0 = headline formed, 1 = fully scattered) and time t (seconds)?
 * Returns { x, y, alpha }. A tiny shimmer keeps the formed headline alive; reduced motion passes t = 0.
 */
export function particleAt(pt, p, t = 0, shimmer = 1.2) {
  const local = clamp01((p - pt.delay) / (1 - 0.35));
  const k = easeInOut(local);
  const wob = Math.sin(t * 1.3 + pt.phase) * shimmer * (1 - k);
  const wob2 = Math.cos(t * 1.1 + pt.phase) * shimmer * (1 - k);
  return {
    x: pt.tx + (pt.sx - pt.tx) * k + wob,
    y: pt.ty + (pt.sy - pt.ty) * k + wob2,
    alpha: 1 - 0.85 * k,
  };
}

/** How many particles to use: fewer on small screens. */
export function particleBudget(width, reduced = false) {
  if (reduced) return 1800;
  return width < 700 ? 2200 : width < 1100 ? 3600 : 5200;
}

/** Break a headline into lines that fit: greedy by word, using a measuring function. */
export function wrapWords(text, maxWidth, measure) {
  const words = String(text).split(/\s+/).filter(Boolean);
  const lines = [];
  let cur = "";
  for (const w of words) {
    const test = cur ? `${cur} ${w}` : w;
    if (cur && measure(test) > maxWidth) {
      lines.push(cur);
      cur = w;
    } else cur = test;
  }
  if (cur) lines.push(cur);
  return lines;
}
