import { useEffect, useRef } from "react";
import { brightPixels, makeParticles, particleAt, particleBudget, samplePoints, wrapWords } from "../lib/particles.js";
import { easeOut } from "../lib/scroll.js";
import { usePrefs } from "../state/prefs.jsx";

const TEXT = "Understand What Changed. Discover Why.";

/**
 * The opening headline, drawn as glowing particles. `progress` (0..1, from scrolling) scatters them.
 * A real <h1> stays in the page for screen readers and search; with reduced motion the headline is plain text.
 */
export default function ParticleHeadline({ progress }) {
  const { reduced } = usePrefs();
  const wrap = useRef(null);
  const canvas = useRef(null);
  const prog = useRef(0);
  prog.current = progress;

  useEffect(() => {
    if (reduced) return undefined;
    const el = canvas.current;
    const box = wrap.current;
    const ctx = el.getContext("2d");
    let raf = 0, parts = [], w = 0, h = 0, dpr = 1, alive = true, last = -1;

    const build = () => {
      const r = box.getBoundingClientRect();
      w = Math.max(280, Math.floor(r.width));
      h = Math.max(200, Math.floor(r.height));
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      el.width = w * dpr;
      el.height = h * dpr;
      el.style.width = `${w}px`;
      el.style.height = `${h}px`;
      // 1. draw the words on a hidden canvas
      const off = document.createElement("canvas");
      off.width = w;
      off.height = h;
      const o = off.getContext("2d");
      let fs = Math.min(w / 6.2, 150);
      const font = (s) => `800 ${s}px Archivo, Arial, sans-serif`;
      let lines;
      for (let tries = 0; tries < 12; tries++) {
        o.font = font(fs);
        lines = wrapWords(TEXT.toUpperCase(), w * 0.94, (s) => o.measureText(s).width);
        if (lines.length * fs * 1.02 <= h * 0.86) break;
        fs *= 0.9;
      }
      o.font = font(fs);
      o.fillStyle = "#fff";
      o.textAlign = "center";
      o.textBaseline = "middle";
      const lh = fs * 1.02;
      const y0 = h / 2 - ((lines.length - 1) * lh) / 2;
      lines.forEach((ln, i) => o.fillText(ln, w / 2, y0 + i * lh));
      // 2. turn the lit pixels into particles
      const gap = w < 700 ? 4 : 3;
      const px = brightPixels(o.getImageData(0, 0, w, h).data, w, h, gap);
      parts = makeParticles(samplePoints(px, particleBudget(w)), w, h);
      last = -1;
    };

    const draw = (t) => {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const p = prog.current;
      const grad = ctx.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0, "#C4B5FD");
      grad.addColorStop(0.5, "#F5EEED");
      grad.addColorStop(1, "#F9A8D4");
      ctx.fillStyle = grad;
      for (const pt of parts) {
        const q = particleAt(pt, easeOut(p), t);
        ctx.globalAlpha = q.alpha * (0.55 + 0.45 * Math.sin(t * 2 + pt.phase) ** 2);
        ctx.fillRect(q.x, q.y, pt.size, pt.size);
      }
      ctx.globalAlpha = 1;
    };
    const loop = (ms) => {
      if (!alive) return;
      const p = prog.current;
      // while the page is idle and nothing moved, still shimmer gently (cheap: one pass)
      draw(ms / 1000);
      last = p;
      raf = requestAnimationFrame(loop);
    };

    const start = () => {
      if (!alive) return;
      build();
      raf = requestAnimationFrame(loop);
    };
    (document.fonts?.load ? document.fonts.load("800 80px Archivo").catch(() => {}) : Promise.resolve()).then(start);
    const onResize = () => alive && build();
    window.addEventListener("resize", onResize);
    return () => {
      alive = false;
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
    };
  }, [reduced]);

  return (
    <div className="particle-headline" ref={wrap}>
      <h1 className={reduced ? "hero-plain" : "sr-only"}>
        Understand What Changed. <span>Discover Why.</span>
      </h1>
      {!reduced && <canvas ref={canvas} aria-hidden="true" />}
    </div>
  );
}
