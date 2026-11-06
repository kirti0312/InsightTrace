"""The number validator: the gate every LLM explanation must pass before a user sees it.

The idea, in plain words
------------------------
The LLM only REWORDS evidence made by Python/SQL. It must never invent a figure. This module
checks the LLM's text against the evidence and says OK or lists exactly what is wrong.

Rules (every one is a hard rule; a text that breaks any of them is rejected)
-----------------------------------------------------------------------------
 1. Numbers must exist in the evidence. Every number in the text (after removing dates and the
    digits inside names like "Customer 242") must be a number from the evidence.
 2. Numbers must be cited. Each sentence that contains a number must end with the id(s) of the
    record(s) it comes from, e.g. [E2] or [E1, E2] (or [F] for a forecast), and its numbers must
    appear IN THOSE records, not just anywhere in the evidence.
 3. Citations must be real: [E9] is rejected when there is no record E9. [F] needs a forecast.
 4. Dates must be the evidence's dates. "1 Mar 2024", "Mar 2024" and "2024" are accepted only
    if they fall inside a window the evidence covers or are written in a record; an invented date is treated as numbers.
 5. No numbers written as words ("five customers", "half", "doubled"): write digits from the
    evidence. (Otherwise a number could be smuggled in without being checked.)
 6. No overclaiming: "proves", "caused by", "due to", "because of", "definitely"... are rejected.
    ("does not prove" and "not proven" are fine.) Correlation is not causation.
 7. Honesty about uncertainty: an UNKNOWN conclusion must say UNKNOWN; a LOW_CONFIDENCE
    forecast must say so; an explanation of a verified cause must hedge ("the evidence
    supports ...") and must cite at least one cause record.
 8. Length: short plain text (default at most 220 words).

Honest limits
-------------
  - This guarantees every number and date comes from the evidence and is attached to the right
    record. It does NOT understand meaning: a text could still pair a true number with a
    slightly misleading verb ("rose" instead of "fell"). Prompts keep the model close to the
    record sentences; the fallback (plain sentences straight from the records) is always safe.
  - Words like "most" or "a few" are not numbers and are not checked.

No LLM is used here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from engine.evidence import NUM_RE, EvidenceBundle, Record

NEG_AFTER_DIRECTION = re.compile(
    r"\b(?:fell|fall|falls|fallen|falling|dropped|drop|drops|declined|decline|declines|decreased|decrease|"
    r"decreases|shrank|rose|rise|rises|risen|rising|grew|grow|grows|increased|increase|increases|gained|lost)"
    r"\s+(?:by\s+|of\s+)?[-\u2212\u2013]\s?[$\u20b9\u20ac\u00a3]?\d", re.IGNORECASE)
CITE_RE = re.compile(r"\[\s*((?:E\d+|F)(?:\s*,\s*(?:E\d+|F))*)\s*\]")

_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
          r"Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
DMY_RE = re.compile(rf"\b(\d{{1,2}})\s+({_MONTH})\.?\s+((?:19|20)\d{{2}})\b", re.I)
MY_RE = re.compile(rf"\b({_MONTH})\.?\s+((?:19|20)\d{{2}})\b", re.I)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

NUMBER_WORDS = re.compile(
    r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|dozens|hundred|hundreds|"
    r"thousand|thousands|million|millions|billion|billions|half|double|doubled|triple|tripled|halved|"
    r"twice|thrice)\b", re.I)

# (pattern, short label). "does not prove" / "not proven" / "no proof" are allowed by the lookbehinds.
OVERCLAIMS = [
    (re.compile(r"(?<!not )(?<!n't )(?<!never )\bprove[sd]?\b", re.I), "proves"),
    (re.compile(r"(?<!not )(?<!n't )(?<!never )\bproven\b", re.I), "proven"),
    (re.compile(r"(?<!no )(?<!not )\bproof\b", re.I), "proof"),
    (re.compile(r"\b(definitely|certainly|undoubtedly|guarantee[sd]?)\b", re.I), "certainty wording"),
    (re.compile(r"\bcaused by\b", re.I), "caused by"),
    (re.compile(r"\bthe (?:root )?cause (?:is|was)\b(?!\s+(?:currently\s+|still\s+)?unknown)", re.I), "the cause is"),
    (re.compile(r"\bis responsible for\b", re.I), "is responsible for"),
    (re.compile(r"\bdue to\b", re.I), "due to"),
    (re.compile(r"\bbecause of\b", re.I), "because of"),
]
HEDGE_RE = re.compile(r"support|suggest|not prove|not proof|not proven|lead|likely|appears", re.I)
LOW_CONF_RE = re.compile(r"low[ _-]confidence|rough|unreliable|not reliable|with care|caution|uncertain", re.I)


@dataclass
class ValidationResult:
    ok: bool
    problems: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)      # numbers that had to be supported

    def message(self) -> str:
        return "; ".join(self.problems) if self.problems else "ok"


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def citations_in(text: str) -> list[str]:
    out: list[str] = []
    for m in CITE_RE.finditer(text):
        out += [c.strip() for c in m.group(1).split(",")]
    return out


def strip_cites(text: str) -> str:
    return CITE_RE.sub(" ", text)


def _norm_number(tok: str) -> str:
    return tok.replace(",", "").rstrip(".")


def _numbers(text: str) -> set[str]:
    return {_norm_number(m) for m in NUM_RE.findall(text)}


def _mon3(name: str) -> str:
    return name[:3].lower()


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if p.strip()]
    merged: list[str] = []
    for p in parts:
        if merged and CITE_RE.fullmatch(p.strip(" .")):       # a citation that landed after the full stop
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


def _remove(text: str, names, repl: str = " ") -> str:
    for n in sorted((n for n in names if n), key=len, reverse=True):
        text = text.replace(n, repl)
    return text


@dataclass
class _Dates:
    dmy: set = field(default_factory=set)       # (day, mon3, year)
    my: set = field(default_factory=set)        # (mon3, year)
    years: set = field(default_factory=set)     # "2024"

    def add_span(self, a, b):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        for ts in (a, b):
            self.dmy.add((str(ts.day), ts.strftime("%b").lower(), str(ts.year)))
        for p in pd.period_range(a, b, freq="M"):
            self.my.add((p.strftime("%b").lower(), str(p.year)))
            self.years.add(str(p.year))

    def add_month(self, ts):
        ts = pd.Timestamp(ts)
        self.my.add((ts.strftime("%b").lower(), str(ts.year)))
        self.years.add(str(ts.year))

    def strip(self, text: str, everything: bool = False) -> str:
        """Blank out date expressions that the evidence covers (all of them if `everything`)."""
        def dmy(m):
            ok = everything or (str(int(m.group(1))), _mon3(m.group(2)), m.group(3)) in self.dmy
            return " " if ok else m.group(0)

        def my(m):
            ok = everything or (_mon3(m.group(1)), m.group(2)) in self.my
            return " " if ok else m.group(0)

        def yr(m):
            return " " if (everything or m.group(0) in self.years) else m.group(0)

        text = DMY_RE.sub(dmy, text)
        text = MY_RE.sub(my, text)
        return YEAR_RE.sub(yr, text)


def _bundle_dates(bundle: EvidenceBundle | None, forecast) -> _Dates:
    d = _Dates()
    if bundle is not None:
        for w in (bundle.window, bundle.compare):
            d.add_span(pd.Timestamp(w[0]), pd.Timestamp(w[1]))
        # dates the records themselves state ("quarter starting 1 Apr 2022"): they come from the evidence
        for r in bundle.records:
            for t in [r.text] + [f.display for f in r.facts.values()] + [c["detail"] for c in r.checks]:
                for m in DMY_RE.finditer(t):
                    d.dmy.add((str(int(m.group(1))), _mon3(m.group(2)), m.group(3)))
                    d.my.add((_mon3(m.group(2)), m.group(3)))
                    d.years.add(m.group(3))
                for m in MY_RE.finditer(t):
                    d.my.add((_mon3(m.group(1)), m.group(2)))
                    d.years.add(m.group(2))
    if forecast is not None:
        for p in forecast.points["period"]:
            d.add_month(p)
        if len(forecast.history):
            d.add_month(forecast.history["period"].iloc[0])
            d.add_month(forecast.history["period"].iloc[-1])
    return d


def record_numbers(rec: Record, entities: set[str]) -> set[str]:
    """Every number a record contains (its sentence, fact displays and check sentences)."""
    texts = [rec.text] + [f.display for f in rec.facts.values()] + [c["detail"] for c in rec.checks]
    out: set[str] = set()
    for t in texts:
        t = _Dates().strip(_remove(t, entities), everything=True)
        out |= _numbers(t)
    return out


def forecast_numbers(fc) -> set[str]:
    """Every number a forecast may be described with."""
    out: set[str] = set()
    dates = _bundle_dates(None, fc)
    texts = list(fc.facts().values()) + [fc.reason] + list(fc.notes) + [str(len(fc.points))]
    if fc.validation.get("n_checks") is not None:
        texts.append(str(fc.validation["n_checks"]))
    for t in texts:
        out |= _numbers(dates.strip(t, everything=True))
    return out


def _snip(s: str, n: int = 110) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n - 3] + "..."


# ----------------------------------------------------------------------------
# The validator
# ----------------------------------------------------------------------------
def validate_explanation(text: str, bundle: EvidenceBundle | None = None, forecast=None,
                         require_citations: bool = True, max_words: int = 220) -> ValidationResult:
    """Check an explanation against an EvidenceBundle and/or a Forecast. See the module notes."""
    if bundle is None and forecast is None:
        raise ValueError("Give the validator an EvidenceBundle, a Forecast, or both.")
    problems: list[str] = []
    t = (text or "").strip()
    if not t:
        return ValidationResult(False, ["The explanation is empty."])
    n_words = len(t.split())
    if n_words > max_words:
        problems.append(f"It is too long ({n_words} words); keep it under {max_words} words.")
    if "```" in t:
        problems.append("Use plain sentences, no code blocks.")

    entities = bundle.entity_names() if bundle is not None else set()
    dates = _bundle_dates(bundle, forecast)
    valid_ids = {r.id for r in bundle.records} if bundle is not None else set()
    rec_nums = {r.id: record_numbers(r, entities) for r in bundle.records} if bundle is not None else {}
    f_nums = forecast_numbers(forecast) if forecast is not None else set()
    everything = set().union(*rec_nums.values()) | f_nums if (rec_nums or f_nums) else set()

    # Names from the user's file may contain digits AND full stops ("Smith Co. Ltd", "St. Louis 2").
    # Mask them with a plain word BEFORE splitting sentences so they cannot break the checks.
    masked = _remove(t, entities, "ENTITY")
    all_cites: list[str] = []
    needed: set[str] = set()
    for s in _sentences(masked):
        cites = citations_in(s)
        all_cites += cites
        usable: list[str] = []
        for c in cites:
            if c == "F":
                if forecast is None:
                    problems.append("It cites [F] but there is no forecast.")
                else:
                    usable.append(c)
            elif c not in valid_ids:
                problems.append(f"It cites [{c}], which is not an evidence record.")
            else:
                usable.append(c)
        body = dates.strip(strip_cites(s))
        nums = _numbers(body)
        needed |= nums
        if not nums:
            continue
        if usable:
            allowed = set().union(*[f_nums if c == "F" else rec_nums[c] for c in usable])
            bad = sorted(nums - allowed)
            if bad:
                problems.append(f"Number(s) {', '.join(bad)} are not in the record(s) cited "
                                f"({', '.join('[' + c + ']' for c in usable)}): \"{_snip(s)}\"")
        elif require_citations:
            problems.append(f"Number(s) {', '.join(sorted(nums))} appear without a citation: \"{_snip(s)}\"")
        else:
            bad = sorted(nums - everything)
            if bad:
                problems.append(f"Number(s) {', '.join(bad)} are not in the evidence: \"{_snip(s)}\"")

    if require_citations and not all_cites:
        problems.append("It has no citations; end sentences with the record ids, e.g. [E2].")

    plain = strip_cites(masked)
    stray = re.search(r"\[[^\]]{0,40}\]", plain)
    if stray:                                  # e.g. "[limits]": brackets are for evidence ids only
        problems.append(f"Square brackets are only for citations such as [E2]; remove \"{stray.group(0)}\".")
    neg = NEG_AFTER_DIRECTION.search(plain)
    if neg:                                    # "fell by -11,354": the direction word already carries the sign
        problems.append(f"Do not write a minus sign after a direction word (\"{neg.group(0).strip()}\"); "
                        f"say \"fell by 11,354\" instead.")
    for rx, label in OVERCLAIMS:
        m = rx.search(plain)
        if m:
            problems.append(f"Do not use \"{m.group(0)}\": it overstates the evidence (correlation is not causation).")
    m = NUMBER_WORDS.search(plain)
    if m:
        problems.append(f"Write numbers as digits copied from the evidence, not as the word \"{m.group(0)}\".")

    low = t.lower()
    if bundle is not None:
        if bundle.conclusion == "UNKNOWN" and "unknown" not in low:
            problems.append("The evidence conclusion is UNKNOWN; the explanation must say the cause is UNKNOWN.")
        causes = {r.id for r in bundle.records if r.kind == "cause"}
        if bundle.conclusion != "UNKNOWN" and causes:
            if not HEDGE_RE.search(t):
                problems.append("Say the evidence supports or suggests the cause; do not state it as certain.")
            if require_citations and not (causes & set(all_cites)):
                problems.append("Cite at least one cause record (" + ", ".join(sorted(causes)) + ").")
    if forecast is not None:
        if forecast.status == "LOW_CONFIDENCE" and not LOW_CONF_RE.search(t):
            problems.append("The forecast is LOW_CONFIDENCE; the explanation must say so.")
        if forecast.status == "UNKNOWN" and not re.search(r"unknown|no forecast", low):
            problems.append("The forecast is UNKNOWN; the explanation must say no forecast could be made.")
        if require_citations and forecast.status != "UNKNOWN" and "F" not in all_cites and bundle is None:
            problems.append("Cite the forecast with [F].")

    return ValidationResult(not problems, problems, all_cites, sorted(needed))
