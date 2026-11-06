import { useEffect, useState } from "react";
import UploadPanel from "../components/UploadPanel.jsx";
import Overview from "../components/Overview.jsx";
import Analyst from "../components/Analyst.jsx";
import RootCause from "../components/RootCause.jsx";
import Forecast from "../components/Forecast.jsx";
import Voice from "../components/Voice.jsx";
import { useSession } from "../state/session.jsx";

const TABS = [
  ["overview", "OVERVIEW", null],
  ["analyst", "ANALYST", null],
  ["root", "ROOT CAUSE", null],
  ["forecast", "FORECAST", null],
  ["voice", "VOICE", null],
];

const BLURB = {
  analyst: "Ask questions in plain English. The AI picks one safe tool, Python or SQL computes the numbers, and every answer shows its evidence.",
  root: "Pick an alert and see who contributed, then five checks decide CAUSE, NOT A CAUSE or UNKNOWN.",
  forecast: "Next months with honest uncertainty bands, checked against the past before you rely on them.",
  voice: "Ask by voice and hear the answer read back.",
};

export default function Workspace() {
  const { summary, restoring } = useSession();
  const [tab, setTab] = useState("overview");
  const [chat, setChat] = useState([]);
  const fresh = { kpi: "", freq: "M", alerts: null, selected: "", result: null, error: "", running: false };
  const [rc, setRc] = useState(fresh);
  const freshFc = { kpi: "", freq: "month", horizon: "", result: null, error: "", running: false };
  const [fs, setFs] = useState(freshFc);
  const [vh, setVh] = useState([]);
  const [focus, setFocus] = useState(null);
  const sid = summary?.session_id;
  useEffect(() => { setChat([]); setVh([]); setRc(fresh); setFs(freshFc); setFocus(null); }, [sid]);   // eslint-disable-line react-hooks/exhaustive-deps
  const investigateFromOverview = (e) => { setFocus(e); setTab("root"); };          // a new dataset starts a new conversation

  return (
    <main className="workspace">
      <div className="ws-head">
        <p className="mono eyebrow">WORKSPACE</p>
        <h1 className="display sm">{summary ? summary.name : "Load your data"}</h1>
      </div>

      {restoring ? (
        <div className="skeleton" style={{ height: 160 }} aria-busy="true" />
      ) : !summary ? (
        <UploadPanel />
      ) : (
        <>
          <div className="tabs mono" role="tablist" aria-label="Workspace sections">
            {TABS.map(([k, t, step]) => (
              <button key={k} role="tab" type="button" aria-selected={tab === k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>
                {t}{step && <small> · {step.toUpperCase()}</small>}
              </button>
            ))}
          </div>
          <div role="tabpanel">
            {tab === "overview" ? (
              <>
                <Overview onInvestigate={investigateFromOverview} />
                <details className="data-panel">
                  <summary className="mono">DATA · CHANGE OR REMOVE</summary>
                  <UploadPanel compact />
                </details>
              </>
            ) : tab === "root" ? (
              <RootCause rc={rc} setRc={setRc} focus={focus} clearFocus={() => setFocus(null)} />
            ) : tab === "voice" ? (
              <Voice history={vh} setHistory={setVh} />
            ) : tab === "forecast" ? (
              <Forecast fs={fs} setFs={setFs} />
            ) : tab === "analyst" ? (
              <Analyst history={chat} setHistory={setChat} />
            ) : (
              <section className="glass placeholder">
                <p className="mono eyebrow">COMING IN {TABS.find((t) => t[0] === tab)[2].toUpperCase()}</p>
                <p className="lede">{BLURB[tab]}</p>
                <p className="dim">The backend for this already exists and is tested. This screen is not built yet.</p>
              </section>
            )}
          </div>
        </>
      )}
    </main>
  );
}
