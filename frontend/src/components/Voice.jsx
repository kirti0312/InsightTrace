import { useEffect, useMemo, useRef, useState } from "react";
import AnswerCard from "./AnswerCard.jsx";
import Robot from "./Robot.jsx";
import { useSession } from "../state/session.jsx";
import { LANGUAGES, RATES, Speaker, createRecognizer, errorMessage, supportInfo } from "../lib/voice.js";

let counter = 0;

/** Ask by voice, hear the answer. Uses the browser's own speech tools; typing always works as a fallback. */
export default function Voice({ history, setHistory }) {
  const { client, summary, expired } = useSession();
  const sid = summary.session_id;
  const support = useMemo(() => supportInfo(window), []);
  const speaker = useMemo(() => (support.synthesis ? new Speaker(window.speechSynthesis, window.SpeechSynthesisUtterance) : null), [support.synthesis]);

  const [phase, setPhase] = useState("idle");              // idle | listening | thinking | speaking
  const [interim, setInterim] = useState("");
  const [notice, setNotice] = useState("");
  const [typed, setTyped] = useState("");
  const [lang, setLang] = useState(() => (/^(en-IN|en-GB|hi-IN)$/.test(navigator.language) ? navigator.language : "en-IN"));
  const [rate, setRate] = useState("1");
  const [readAloud, setReadAloud] = useState(true);
  const [autoSend, setAutoSend] = useState(true);
  const [aiOn, setAiOn] = useState(null);
  const recRef = useRef(null);
  const gotFinal = useRef(false);

  useEffect(() => {
    client.health().then((h) => setAiOn(!!h?.llm?.configured)).catch(() => setAiOn(false));
  }, [client]);
  useEffect(() => () => { speaker?.stop(); recRef.current?.stop(); }, [speaker]);      // leaving the tab silences everything

  const patch = (id, change) => setHistory((h) => h.map((x) => (x.id === id ? { ...x, ...change } : x)));

  const say = (text) => {
    if (!speaker || !readAloud) return setPhase("idle");
    const started = speaker.speak(text, { lang, rate: Number(rate), onStart: () => setPhase("speaking"), onEnd: () => setPhase("idle") });
    if (!started) setPhase("idle");
  };

  const send = async (question) => {
    const q = question.trim();
    if (!q) return;
    speaker?.stop();
    setNotice("");
    const id = ++counter;
    setHistory((h) => [...h, { id, question: q, mode: "ask", pending: true }]);
    setPhase("thinking");
    try {
      const answer = await client.ask(sid, q, aiOn === true);
      patch(id, { answer, pending: false });
      say(answer.text);
    } catch (e) {
      if (e.status === 404) return expired();
      patch(id, { error: e.message, pending: false });
      setPhase("idle");
    }
  };

  const listen = () => {
    if (phase === "listening") return recRef.current?.stop();
    speaker?.stop();
    setNotice("");
    setInterim("");
    gotFinal.current = false;
    const rec = createRecognizer(window, {
      lang,
      onInterim: setInterim,
      onFinal: (t) => {
        gotFinal.current = true;
        setInterim("");
        if (autoSend) send(t);
        else setTyped(t);
      },
      onError: (code) => setNotice(errorMessage(code)),
      onEnd: () => setPhase((p) => (p === "listening" ? "idle" : p)),
    });
    recRef.current = rec;
    setPhase("listening");
    rec.start();
  };

  const stopAll = () => {
    speaker?.stop();
    recRef.current?.stop();
    setPhase("idle");
  };

  const busy = phase === "thinking";
  const label = { idle: support.recognition ? "TAP THE MICROPHONE AND ASK" : "TYPE YOUR QUESTION BELOW", listening: "LISTENING… TAP TO STOP", thinking: "WORKING OUT THE ANSWER…", speaking: "SPEAKING… " }[phase];

  return (
    <div className="voice">
      <div className="voice-stage glass">
        <div className={`voice-robot is-${phase}`}><Robot /></div>
        <div className="voice-side">
          <p className="mono eyebrow">VOICE</p>
          {support.recognition ? (
            <button type="button" className={`mic is-${phase}`} aria-pressed={phase === "listening"} disabled={busy} onClick={listen} aria-label={phase === "listening" ? "Stop listening" : "Start listening"}>
              <span className="mic-ring r1" /><span className="mic-ring r2" />
              <svg viewBox="0 0 24 24" width="34" height="34" aria-hidden="true"><path fill="currentColor" d="M12 15a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V22h2v-3.08A7 7 0 0 0 19 12h-2Z" /></svg>
            </button>
          ) : (
            <p className="warn">Voice input is not available in this browser. Chrome or Edge on a computer or Android works best. You can type your question below, and answers can still be read aloud.</p>
          )}
          <p className="mono voice-state" role="status" aria-live="polite">{label}</p>
          <p className="voice-heard" aria-live="polite">{interim ? <><span className="dim">HEARING · </span>{interim}</> : null}</p>
          {phase === "speaking" && <button type="button" className="btn ghost mono" onClick={stopAll}>STOP SPEAKING</button>}
        </div>
      </div>

      {notice && <p className="error" role="alert">{notice}</p>}
      {aiOn === false && <p className="warn">AI is not connected on the server, so spoken questions cannot be understood yet. The Overview, Root Cause and Forecast tabs still work.</p>}

      <div className="voice-opts mono">
        <label><span className="dim">LANGUAGE</span>
          <select value={lang} onChange={(e) => setLang(e.target.value)}>{LANGUAGES.map(([c, t]) => <option key={c} value={c}>{t}</option>)}</select>
        </label>
        <label><span className="dim">SPEAKING SPEED</span>
          <select value={rate} onChange={(e) => setRate(e.target.value)}>{RATES.map(([c, t]) => <option key={c} value={c}>{t}</option>)}</select>
        </label>
        <label className="toggle"><input type="checkbox" checked={readAloud} disabled={!support.synthesis} onChange={(e) => { setReadAloud(e.target.checked); if (!e.target.checked) speaker?.stop(); }} /> READ ANSWERS ALOUD</label>
        <label className="toggle"><input type="checkbox" checked={autoSend} onChange={(e) => setAutoSend(e.target.checked)} /> SEND WHAT I SAY AUTOMATICALLY</label>
      </div>

      <form className="askbox glass" onSubmit={(e) => { e.preventDefault(); const q = typed; setTyped(""); send(q); }}>
        <label className="sr-only" htmlFor="voice-typed">Type a question instead</label>
        <input id="voice-typed" value={typed} maxLength={500} onChange={(e) => setTyped(e.target.value)} placeholder="Or type here, or check what was heard, then press Ask" />
        <button type="submit" className="btn mono" disabled={busy || !typed.trim()}>ASK</button>
      </form>

      <div className="thread">
        {history.map((h) => (
          <div key={h.id} className="voice-item">
            <AnswerCard item={h} />
            {h.answer?.text && support.synthesis && (
              <button type="button" className="link-btn mono" onClick={() => say(h.answer.text)}>▶ READ THIS ANSWER ALOUD</button>
            )}
          </div>
        ))}
      </div>

      <p className="dim voice-privacy">Privacy: speech to text is done by your browser's speech service. In Chrome, the audio is sent to Google to be turned into text. InsightTrace itself receives only the text.</p>
    </div>
  );
}
