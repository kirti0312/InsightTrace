// Pure helpers for the Root Cause tab: turn an evidence bundle into rows for the bars and the check grid.

export const VERDICT_MEANING = {
  CAUSE: "At least one group passed every check: it moved differently from the rest of the business, started inside the window, is not a yearly repeat, and is unusual for its own history. Still evidence, not proof.",
  NOT_A_CAUSE: "There is evidence against it: it moved with everything else, started before the window, repeats every year, or is not distinctive.",
  UNKNOWN: "The evidence is too weak to say. This part of the business is normally this noisy, or there is too little history. Nothing is claimed.",
};

/** "59%" -> 59 ; "-4.5%" -> -4.5 ; anything else -> null */
export function parseShare(display) {
  const m = /^\s*([+-]?\d+(?:\.\d+)?)\s*%\s*$/.exec(String(display ?? ""));
  return m ? Number(m[1]) : null;
}

export function checkMark(status) {
  const s = String(status || "").toLowerCase();
  if (s === "pass") return { mark: "✓", label: "PASS", tone: "pass" };
  if (s === "fail") return { mark: "✗", label: "FAIL", tone: "fail" };
  return { mark: "?", label: "UNSURE", tone: "unknown" };
}

/** Candidates (groups tested as a possible cause) in the order the engine ranked them. */
export function candidates(evidence) {
  const out = [];
  for (const r of evidence?.records || []) {
    if (r.kind !== "cause" && r.kind !== "rejected") continue;     // the "unknown" record is the verdict itself, not a candidate
    const checks = r.checks || [];
    const share = parseShare(r.facts?.share?.display);
    out.push({
      id: r.id,
      title: r.title,
      verdict: r.verdict || "",
      share,
      shareDisplay: r.facts?.share?.display ?? "",
      contribution: r.facts?.contribution?.display ?? "",
      checks,
      passed: checks.filter((c) => String(c.status).toLowerCase() === "pass").length,
      total: checks.length,
      entities: r.entities || [],
      text: r.text,
    });
  }
  return out;
}

/** Column names for the check grid: every check seen, in first-seen order. */
export function checkColumns(cands) {
  const seen = [];
  for (const c of cands) for (const k of c.checks) if (!seen.includes(k.name)) seen.push(k.name);
  return seen;
}

export function bar(share) {
  if (share == null) return 0;
  return Math.max(2, Math.min(100, Math.abs(share)));
}

/** The first headline record, for the title line. */
export function headlineRecord(evidence) {
  return (evidence?.records || []).find((r) => r.kind === "headline") || null;
}
