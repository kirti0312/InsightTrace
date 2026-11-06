// Ambient background sparkles: positions and twinkle, pure so they can be tested.
import { rng } from "./particles.js";

export function makeSparkles(count, seed = 3) {
  const r = rng(seed);
  return Array.from({ length: count }, () => ({
    x: r(),                      // 0..1 of the width
    y: r(),                      // 0..1 of the height
    size: 0.5 + r() * 1.6,
    speed: 0.004 + r() * 0.016,  // upward drift per second (fraction of height)
    sway: 0.002 + r() * 0.006,
    twinkle: 0.5 + r() * 2.5,
    phase: r() * Math.PI * 2,
    hue: r() < 0.55 ? "violet" : r() < 0.8 ? "pink" : "blue",
  }));
}

export function sparkleAt(s, t, parallax = 0) {
  const y = (((s.y - s.speed * t) % 1) + 1) % 1;
  const x = s.x + Math.sin(t * 0.3 + s.phase) * s.sway;
  const a = 0.25 + 0.75 * (0.5 + 0.5 * Math.sin(t * s.twinkle + s.phase));
  return { x, y: (((y - parallax * s.size * 0.02) % 1) + 1) % 1, alpha: a };
}

export function sparkleCount(width, reduced = false) {
  if (reduced) return 40;
  return width < 700 ? 60 : 130;
}
