import { useEffect, useRef } from "react";
import { makeSparkles, sparkleAt, sparkleCount } from "../lib/sparkles.js";
import { usePrefs } from "../state/prefs.jsx";

const COLORS = { violet: "167,139,250", pink: "244,114,182", blue: "56,189,248" };

/** Slowly rising, twinkling dust behind everything. Paused when the tab is hidden; still image when motion is reduced. */
export default function Sparkles() {
  const ref = useRef(null);
  const { reduced } = usePrefs();

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas.getContext("2d");
    let w = 0, h = 0, raf = 0, dpr = 1;
    const dots = makeSparkles(sparkleCount(window.innerWidth, reduced));
    let scrollY = window.scrollY;
    const onScroll = () => (scrollY = window.scrollY);

    const size = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = window.innerWidth;
      h = window.innerHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
    };
    const draw = (t) => {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      for (const d of dots) {
        const p = sparkleAt(d, t, reduced ? 0 : scrollY / 1400);
        ctx.fillStyle = `rgba(${COLORS[d.hue]},${(p.alpha * 0.8).toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(p.x * w, p.y * h, d.size, 0, Math.PI * 2);
        ctx.fill();
      }
    };
    const loop = (ms) => {
      draw(ms / 1000);
      raf = requestAnimationFrame(loop);
    };
    size();
    if (reduced) draw(2);
    else raf = requestAnimationFrame(loop);
    const onVis = () => {
      cancelAnimationFrame(raf);
      if (!document.hidden && !reduced) raf = requestAnimationFrame(loop);
    };
    window.addEventListener("resize", size);
    window.addEventListener("scroll", onScroll, { passive: true });
    document.addEventListener("visibilitychange", onVis);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", size);
      window.removeEventListener("scroll", onScroll);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [reduced]);

  return <canvas ref={ref} className="sparkles" aria-hidden="true" />;
}
