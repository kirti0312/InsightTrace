import { useState } from "react";
import AnswerCard from "./AnswerCard.jsx";
import { VERDICT_MEANING, bar, candidates, checkColumns, checkMark, headlineRecord } from "../lib/rootcause.js";
import { verdictInfo } from "../lib/analyst.js";

function Mark({ status }) {
  const m = checkMark(status);
  return <span className={`mk mk-${m.tone}`} title={m.label}><b aria-hidden="true">{m.mark}</b><span className="sr-only">{m.label}</span></span>;
}

/** The verdict, who contributed (bars), and the five checks for each candidate (grid), then the written evidence. */
export default function Investigation({ answer, headline }) {
  const [openRow, setOpenRow] = useState("");
  const ev = answer.evidence;
  const cands = candidates(ev);
  const cols = checkColumns(cands);
  const verdict = verdictInfo(ev?.conclusion);
  const head = headlineRecord(ev);

  return (
    <section className="invest" aria-label="Investigation result">
      <div className="verdict-panel glass">
        <div>
          <p className="mono eyebrow">VERDICT</p>
          <span className={`verdict-big badge-${verdict?.tone || "unknown"}`}>{verdict?.label || "UNKNOWN"}</span>
        </div>
        <div className="verdict-text">
          <p className="verdict-head">{head?.text || headline}</p>
          <p className="dim">{VERDICT_MEANING[ev?.conclusion] || ""}</p>
        </div>
      </div>

      {cands.length > 0 && (
        <div className="glass contrib">
          <p className="mono eyebrow">WHO CONTRIBUTED · SHARE OF THE CHANGE, AND HOW MANY OF THE CHECKS EACH PASSED</p>
          <ul className="bars">
            {cands.map((c) => {
              const v = verdictInfo(c.verdict);
              return (
                <li key={c.id} className={`bar-row v-${v?.tone || "unknown"}`}>
                  <button type="button" className="bar-btn" aria-expanded={openRow === c.id} onClick={() => setOpenRow(openRow === c.id ? "" : c.id)}>
                    <span className="bar-label">{c.title}</span>
                    <span className="bar-track" aria-hidden="true"><span className="bar-fill" style={{ width: `${bar(c.share)}%` }} /></span>
                    <span className="bar-num mono">{c.shareDisplay || "n/a"}</span>
                    <span className="bar-chk mono">{c.total ? `${c.passed}/${c.total} CHECKS` : ""}</span>
                    <span className={`badge badge-${v?.tone || "unknown"} mono`}>{v?.label || "UNKNOWN"}</span>
                  </button>
                  {openRow === c.id && (
                    <div className="bar-more">
                      <p>{c.text}</p>
                      {c.contribution && <p className="mono dim">CONTRIBUTION TO THE CHANGE · {c.contribution}</p>}
                      {c.entities.length > 0 && <p className="mono dim">MEMBERS · {c.entities.join(", ")}</p>}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
          <p className="dim grid-note">A share above 100% means the rest of the business moved the other way. A group that holds nearly all of the change is not distinctive, so it cannot be blamed on its own.</p>
        </div>
      )}

      {cols.length > 0 && (
        <div className="glass grid-wrap">
          <p className="mono eyebrow">THE FIVE CHECKS</p>
          <div className="table-wrap tall">
            <table className="check-grid">
              <thead>
                <tr><th scope="col">Candidate</th>{cols.map((n) => <th key={n} scope="col">{n}</th>)}</tr>
              </thead>
              <tbody>
                {cands.map((c) => (
                  <tr key={c.id}>
                    <th scope="row">{c.title.length > 46 ? `${c.title.slice(0, 44)}…` : c.title}</th>
                    {cols.map((n) => {
                      const k = c.checks.find((x) => x.name === n);
                      return <td key={n} title={k?.detail || ""}>{k ? <Mark status={k.status} /> : <span className="dim">–</span>}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="dim grid-note">Hover a mark for the detail. ✓ pass · ✗ fail · ? unsure. A candidate is a CAUSE only if every check passes.</p>
        </div>
      )}

      <AnswerCard item={{ id: 0, question: headline, mode: "investigate", answer }} bare />
    </section>
  );
}
