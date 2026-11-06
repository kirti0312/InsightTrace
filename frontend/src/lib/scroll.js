// Scroll maths: turn "how far has the page scrolled" into 0..1 progress for a section.
// Natural browser scrolling is never hijacked; these functions only READ the scroll position.

export function clamp01(x) {
  return x <= 0 ? 0 : x > 1 ? 1 : x;
}

/** Progress of a tall section through the viewport: 0 when its top reaches the top of the screen,
 *  1 when its bottom reaches the bottom of the screen (use with a sticky inner stage). */
export function sectionProgress(rectTop, sectionHeight, viewportHeight) {
  const travel = sectionHeight - viewportHeight;
  if (!(travel > 0)) return 0;
  return clamp01(-rectTop / travel);
}

/** Smooth in-out curve (0..1 -> 0..1). */
export function easeInOut(t) {
  const x = clamp01(t);
  return x * x * (3 - 2 * x);
}

export function easeOut(t) {
  const x = clamp01(t);
  return 1 - (1 - x) * (1 - x) * (1 - x);
}

/** Map a global value into a window: remap(0.5, 0.25, 0.75) = 0.5 ; below the window 0 ; above 1. */
export function remap(value, from, to) {
  if (to === from) return value >= to ? 1 : 0;
  return clamp01((value - from) / (to - from));
}

/** Watch an element's section progress. Calls onProgress(0..1) at most once per animation frame while scrolling.
 *  Returns a function that stops watching. `env` lets tests pass fakes. */
export function watchSection(el, onProgress, env = globalThis) {
  let queued = false;
  let last = -1;
  const read = () => {
    queued = false;
    const r = el.getBoundingClientRect();
    const p = sectionProgress(r.top, r.height, env.innerHeight);
    if (p !== last) {
      last = p;
      onProgress(p);
    }
  };
  const schedule = () => {
    if (queued) return;
    queued = true;
    env.requestAnimationFrame(read);
  };
  env.addEventListener("scroll", schedule, { passive: true });
  env.addEventListener("resize", schedule);
  schedule();
  return () => {
    env.removeEventListener("scroll", schedule);
    env.removeEventListener("resize", schedule);
  };
}
