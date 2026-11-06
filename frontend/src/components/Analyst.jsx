import { useEffect, useRef, useState } from "react";
import AnswerCard from "./AnswerCard.jsx";
import QuickTools from "./QuickTools.jsx";
import { useSession } from "../state/session.jsx";
import { suggestQuestions } from "../lib/analyst.js";

const MAX = 500;
let counter = 0;

/** Ask questions in plain English, or run a tool directly. Every answer lists the evidence it rests on. */
export default function Analyst({ history, setHistory }) {
  const { client, summary, expired } = useSession();
  const sid = summary.session_id;
  const [text, setText] = useState("");
  const [useAi, setUseAi] = useState(true);
  const [health, setHealth] = useState(null);
  const endRef = useRef(null);
  const busy = history.some((h) => h.pending);

  useEffect(() => {
    client.health().then(setHealth).catch(() => setHealth(null));
  }, [client]);
  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: "nearest", behavior: "smooth" });
  }, [history.length]);

  const aiOn = !!health?.llm?.configured;
  const patch = (id, change) => setHistory((h) => h.map((x) => (x.id === id ? { ...x, ...change } : x)));

  const start = async (question, mode, call) => {
    const id = ++counter;
    setHistory((h) => [...h, { id, question, mode, pending: true }]);
    try {
      patch(id, { answer: await call(), pending: false });
    } catch (e) {
      if (e.status === 404) return expired();
      patch(id, { error: e.message, pending: false });
    }
  };

  const ask = (q) => {
    const question = (q ?? text).trim();
    if (!question || busy) return;
    if (question.length > MAX) return;
    setText("");
    start(question, "ask", () => client.ask(sid, question, useAi && aiOn));
  };
  const runTool = (tool, args, label) => !busy && start(label, "tool", () => client.tool(sid, tool, args, useAi && aiOn));

  return (
    <div className="analyst">
      <div className="ai-status mono" role="status">
        <span className={`dot ${aiOn ? "on" : "off"}`} />
        {health === null ? "CHECKING AI…" : aiOn ? `AI CONNECTED · ${health.llm.model}` : "AI NOT CONNECTED · QUESTIONS NEED AN AI KEY ON THE SERVER, QUICK TOOLS STILL WORK"}
        {aiOn && (
          <label className="toggle"><input type="checkbox" checked={useAi} onChange={(e) => setUseAi(e.target.checked)} /> AI WORDING</label>
        )}
      </div>

      <div className="thread">
        {history.length === 0 && (
          <div className="glass intro">
            <p className="drop-title">Ask about your data.</p>
            <p className="drop-sub">Numbers always come from calculations on your file. The AI only picks which calculation to run and rewords the result, and a checker rejects any number it cannot find in the evidence.</p>
            <div className="suggest">
              {suggestQuestions(summary).map((q) => (
                <button key={q} type="button" className="chip mono suggest-chip" disabled={busy} onClick={() => ask(q)}>{q}</button>
              ))}
            </div>
          </div>
        )}
        {history.map((h) => <AnswerCard key={h.id} item={h} />)}
        <div ref={endRef} />
      </div>

      <form className="askbox glass" onSubmit={(e) => { e.preventDefault(); ask(); }}>
        <label className="sr-only" htmlFor="ask-input">Your question</label>
        <textarea
          id="ask-input"
          rows={2}
          value={text}
          maxLength={MAX + 50}
          placeholder="e.g. Why did revenue fall in autumn 2024?"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } }}
        />
        <div className="ask-row">
          <span className={`mono dim${text.length > MAX ? " over" : ""}`}>{text.length}/{MAX}</span>
          {history.length > 0 && <button type="button" className="link-btn mono" disabled={busy} onClick={() => setHistory([])}>CLEAR CHAT</button>}
          <button type="submit" className="btn mono" disabled={busy || !text.trim() || text.length > MAX}>{busy ? "THINKING…" : "ASK"}</button>
        </div>
      </form>

      <QuickTools summary={summary} busy={busy} onRun={runTool} />
    </div>
  );
}
