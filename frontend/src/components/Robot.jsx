import { useEffect, useRef } from "react";
import { usePrefs } from "../state/prefs.jsx";

/**
 * "Trace", the InsightTrace assistant. Pure SVG + CSS (no extra library):
 * floats, waves, blinks, its eyes follow the pointer, and the chest screen shows a heartbeat trace.
 * With reduced motion it stands still in a waving pose.
 */
export default function Robot({ className = "" }) {
  const ref = useRef(null);
  const { reduced } = usePrefs();

  useEffect(() => {
    if (reduced) return undefined;
    const el = ref.current;
    const on = (e) => {
      const nx = Math.max(-1, Math.min(1, (e.clientX / window.innerWidth - 0.5) * 2));
      const ny = Math.max(-1, Math.min(1, (e.clientY / window.innerHeight - 0.5) * 2));
      el.style.setProperty("--gx", (nx * 7).toFixed(2));
      el.style.setProperty("--gy", (ny * 5).toFixed(2));
    };
    window.addEventListener("pointermove", on, { passive: true });
    return () => window.removeEventListener("pointermove", on);
  }, [reduced]);

  return (
    <svg ref={ref} className={`robot ${className}`} viewBox="0 0 400 480" role="img" aria-label="Trace, the InsightTrace assistant, waving hello">
      <defs>
        <linearGradient id="rb-white" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#ffffff" />
          <stop offset="0.55" stopColor="#e6defa" />
          <stop offset="1" stopColor="#b9a8ec" />
        </linearGradient>
        <linearGradient id="rb-violet" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#a78bfa" />
          <stop offset="0.5" stopColor="#6d28d9" />
          <stop offset="1" stopColor="#3b1580" />
        </linearGradient>
        <linearGradient id="rb-visor" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#241446" />
          <stop offset="1" stopColor="#07040d" />
        </linearGradient>
        <radialGradient id="rb-eye" cx="0.4" cy="0.35" r="0.7">
          <stop offset="0" stopColor="#ffffff" />
          <stop offset="0.35" stopColor="#7dd3fc" />
          <stop offset="1" stopColor="#0ea5e9" />
        </radialGradient>
        <radialGradient id="rb-thrust" cx="0.5" cy="0.5" r="0.5">
          <stop offset="0" stopColor="#38bdf8" stopOpacity="0.9" />
          <stop offset="0.5" stopColor="#8b5cf6" stopOpacity="0.45" />
          <stop offset="1" stopColor="#f472b6" stopOpacity="0" />
        </radialGradient>
        <filter id="rb-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="rb-soft" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="9" /></filter>
      </defs>

      <ellipse className="rb-thrust" cx="200" cy="452" rx="74" ry="16" fill="url(#rb-thrust)" />

      <g className="rb-float">
        <ellipse cx="200" cy="448" rx="40" ry="9" fill="#38bdf8" opacity="0.5" filter="url(#rb-soft)" />

        {/* left arm (hanging) */}
        <g transform="translate(124,266)">
          <g className="rb-arm-idle">
            <g transform="rotate(16)">
              <rect x="-11" y="0" width="22" height="56" rx="11" fill="url(#rb-white)" />
              <circle cx="0" cy="56" r="11" fill="url(#rb-violet)" />
              <g transform="translate(0,56) rotate(10)">
                <rect x="-9" y="0" width="18" height="44" rx="9" fill="url(#rb-violet)" />
                <circle cx="0" cy="50" r="13" fill="url(#rb-white)" />
                <rect x="-11" y="56" width="6" height="20" rx="3" fill="url(#rb-white)" transform="rotate(14 -8 56)" />
                <rect x="-3" y="58" width="6" height="22" rx="3" fill="url(#rb-white)" />
                <rect x="5" y="56" width="6" height="20" rx="3" fill="url(#rb-white)" transform="rotate(-14 8 56)" />
              </g>
            </g>
          </g>
          <circle r="15" fill="url(#rb-violet)" />
        </g>

        {/* body */}
        <path d="M116 272C116 234 150 216 200 216C250 216 284 234 284 272C284 344 262 410 200 424C138 410 116 344 116 272Z" fill="url(#rb-white)" />
        <path d="M148 398C164 414 236 414 252 398C248 418 228 430 200 432C172 430 152 418 148 398Z" fill="url(#rb-violet)" />
        <ellipse cx="150" cy="262" rx="14" ry="30" fill="#fff" opacity="0.55" transform="rotate(14 150 262)" />
        <rect x="176" y="204" width="48" height="16" rx="8" fill="url(#rb-violet)" />
        <rect x="158" y="266" width="84" height="70" rx="18" fill="#0b0713" stroke="#8b5cf6" strokeWidth="3" />
        <polyline className="rb-pulse" points="166,304 183,304 192,284 203,326 213,294 219,304 234,304" fill="none" stroke="#38bdf8" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" filter="url(#rb-glow)" pathLength="100" />
        <circle cx="200" cy="396" r="4" fill="#f472b6" className="rb-bulb" />

        {/* right arm (waving) */}
        <g transform="translate(276,266)">
          <g className="rb-upper-wave">
            <g transform="rotate(-104)">
              <rect x="-11" y="0" width="22" height="54" rx="11" fill="url(#rb-white)" />
              <circle cx="0" cy="54" r="11" fill="url(#rb-violet)" />
              <g transform="translate(0,54)">
                <g className="rb-wave">
                  <g transform="rotate(-76)">
                    <rect x="-9" y="0" width="18" height="40" rx="9" fill="url(#rb-violet)" />
                    <g transform="translate(0,44)">
                      <circle r="14" fill="url(#rb-white)" />
                      <rect x="-14" y="6" width="7" height="24" rx="3.5" fill="url(#rb-white)" transform="rotate(26 -10 6)" />
                      <rect x="-3.5" y="9" width="7" height="27" rx="3.5" fill="url(#rb-white)" />
                      <rect x="7" y="6" width="7" height="24" rx="3.5" fill="url(#rb-white)" transform="rotate(-26 10 6)" />
                      <circle cx="-17" cy="30" r="3" fill="#f472b6" opacity="0.9" />
                      <circle cx="0" cy="36" r="3" fill="#f472b6" opacity="0.9" />
                      <circle cx="17" cy="30" r="3" fill="#f472b6" opacity="0.9" />
                    </g>
                  </g>
                </g>
              </g>
            </g>
          </g>
          <circle r="15" fill="url(#rb-violet)" />
        </g>

        {/* head */}
        <g className="rb-head">
          <line x1="200" y1="76" x2="200" y2="48" stroke="#a79aa6" strokeWidth="4" strokeLinecap="round" />
          <circle className="rb-bulb" cx="200" cy="40" r="8" fill="#f472b6" filter="url(#rb-glow)" />
          <rect x="90" y="112" width="26" height="58" rx="13" fill="url(#rb-violet)" />
          <rect x="284" y="112" width="26" height="58" rx="13" fill="url(#rb-violet)" />
          <rect x="102" y="74" width="196" height="136" rx="64" fill="url(#rb-white)" />
          <ellipse cx="150" cy="92" rx="34" ry="9" fill="#fff" opacity="0.7" transform="rotate(-12 150 92)" />
          <rect x="120" y="92" width="160" height="100" rx="46" fill="url(#rb-visor)" stroke="#a78bfa" strokeWidth="3.5" />
          <g className="rb-gaze">
            <g className="rb-eye">
              <circle cx="164" cy="138" r="25" fill="#0a1a33" stroke="#38bdf8" strokeWidth="4" filter="url(#rb-glow)" />
              <circle cx="164" cy="138" r="15" fill="url(#rb-eye)" />
              <circle cx="158" cy="131" r="4.5" fill="#fff" />
            </g>
            <g className="rb-eye">
              <circle cx="236" cy="138" r="25" fill="#0a1a33" stroke="#38bdf8" strokeWidth="4" filter="url(#rb-glow)" />
              <circle cx="236" cy="138" r="15" fill="url(#rb-eye)" />
              <circle cx="230" cy="131" r="4.5" fill="#fff" />
            </g>
            <path d="M184 170Q200 184 216 170" fill="none" stroke="#7dd3fc" strokeWidth="3.5" strokeLinecap="round" filter="url(#rb-glow)" />
            <path d="M144 106Q164 97 182 106M218 106Q236 97 256 106" fill="none" stroke="#7dd3fc" strokeWidth="3" strokeLinecap="round" opacity="0.75" />
          </g>
        </g>
      </g>
    </svg>
  );
}
