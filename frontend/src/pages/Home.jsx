import { useEffect, useRef, useState } from "react";
import ParticleHeadline from "../components/ParticleHeadline.jsx";
import UploadPanel from "../components/UploadPanel.jsx";
import Overview from "../components/Overview.jsx";
import Robot from "../components/Robot.jsx";
import { Link } from "../state/router.jsx";
import { useSession } from "../state/session.jsx";
import { usePrefs } from "../state/prefs.jsx";
import { easeOut, remap, watchSection } from "../lib/scroll.js";

function Reveal({ children, className = "" }) {
  const ref = useRef(null);
  const { reduced } = usePrefs();
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    if (reduced || !("IntersectionObserver" in window)) return setSeen(true);
    const io = new IntersectionObserver(([e]) => e.isIntersecting && (setSeen(true), io.disconnect()), { threshold: 0.15 });
    io.observe(ref.current);
    return () => io.disconnect();
  }, [reduced]);
  return <div ref={ref} className={`reveal ${seen ? "in" : ""} ${className}`}>{children}</div>;
}

export default function Home() {
  const { summary } = useSession();
  const { reduced } = usePrefs();
  const hero = useRef(null);
  const [p, setP] = useState(0);

  useEffect(() => {
    if (reduced || !hero.current) return undefined;
    return watchSection(hero.current, setP);
  }, [reduced]);

  const progress = reduced ? 0 : p;
  const orbIn = easeOut(remap(progress, 0.35, 0.8));

  return (
    <main className="home">
      <section ref={hero} className={`hero${reduced ? " hero-static" : ""}`} aria-label="Introduction">
        <div className="hero-stage">
          <div className="corner tl mono">AI BUSINESS INTELLIGENCE</div>
          <div className="corner tr mono">EVIDENCE, NOT GUESSES</div>
          <ParticleHeadline progress={progress} />
          <div className="orb-slot" style={{ opacity: reduced ? 1 : orbIn, transform: `scale(${0.7 + 0.3 * orbIn})` }}>
            <Robot />
          </div>
          <p className="hero-sub mono" style={{ opacity: reduced ? 1 : 1 - remap(progress, 0, 0.25) }}>
            INSIGHTTRACE FINDS WHAT CHANGED IN YOUR BUSINESS, WHO DROVE IT, AND WHETHER THAT IS A REAL CAUSE.
          </p>
          <div className="scroll-cue mono" style={{ opacity: reduced ? 0 : 1 - remap(progress, 0, 0.15) }}>SCROLL ↓</div>
          <div className="callout-tag tagA mono" style={{ opacity: reduced ? 0 : remap(progress, 0.6, 0.85) }}><i />EVERY NUMBER FROM CODE</div>
          <div className="callout-tag tagB mono" style={{ opacity: reduced ? 0 : remap(progress, 0.7, 0.95) }}><i />UNKNOWN WHEN EVIDENCE IS WEAK</div>
        </div>
      </section>

      <section className="scene" id="load" aria-labelledby="load-h">
        <Reveal>
          <p className="mono eyebrow">01 · START</p>
          <h2 id="load-h" className="display">Bring your sales data.</h2>
          <p className="lede">Upload a table, or open a demo. Nothing is invented: every figure you see is calculated from your rows.</p>
        </Reveal>
        <Reveal className="scene-body"><UploadPanel /></Reveal>
      </section>

      <section className="scene" aria-labelledby="changed-h">
        <Reveal>
          <p className="mono eyebrow">02 · WHAT CHANGED</p>
          <h2 id="changed-h" className="display">The unusual, found for you.</h2>
          <p className="lede">Each KPI is compared with what the same time of year usually brings. The shaded band is the expected range; pink dots fall outside it.</p>
        </Reveal>
        <Reveal className="scene-body">
          {summary ? <Overview compact /> : (
            <div className="glass empty-scene">
              <p className="mono">LOAD A DEMO ABOVE TO SEE THIS LIVE</p>
            </div>
          )}
        </Reveal>
      </section>

      <section className="scene finish" aria-labelledby="finish-h">
        <Reveal>
          <p className="mono eyebrow">03 · NEXT</p>
          <h2 id="finish-h" className="display">Ask why. Get evidence.</h2>
          <p className="lede">Correlation is not causation. InsightTrace checks each suspect and says <b>UNKNOWN</b> when the evidence is weak.</p>
          <Link to="/app" className="btn big mono">OPEN THE WORKSPACE →</Link>
        </Reveal>
      </section>

      <footer className="foot mono">INSIGHTTRACE · NUMBERS FROM PYTHON AND SQL · THE AI ONLY REWORDS EVIDENCE</footer>
    </main>
  );
}
