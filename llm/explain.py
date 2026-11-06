"""Explanations: ask the LLM, check it, retry once, otherwise fall back to plain evidence sentences.

The flow (the same for evidence bundles and forecasts)
------------------------------------------------------
    1. No LLM configured?               -> use the plain summary (built only from evidence).
    2. Ask the LLM with the rules + the evidence JSON.
    3. Validate the answer (llm/validator.py).  OK?   -> return it, source = "llm".
    4. Not OK: ask once more, telling it exactly what was wrong.
    5. Still not OK, or the call failed -> the plain summary, source = "fallback",
       with `reason` and `problems` saying why. The user is never shown unchecked LLM text.

The plain summary (EvidenceBundle.summary_text(), Forecast.summary()) is written by Python from the
evidence records, so it is always safe; it is also checked by the validator in the tests.

Nothing here raises for an LLM problem (network, quota, bad key, rejected text): the app keeps working.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from engine.evidence import EvidenceBundle
from llm.client import LLMError
from llm.prompts import build_explain_prompt, build_forecast_prompt, build_retry_prompt
from llm.validator import ValidationResult, validate_explanation


@dataclass
class Explanation:
    text: str
    source: str                    # "llm" or "fallback"
    valid: bool                    # the shown text passed the validator
    attempts: int                  # LLM attempts made (0 when no LLM)
    reason: str                    # "ok", or why the fallback was used
    problems: list[str] = field(default_factory=list)   # validator problems from the last rejected attempt
    model: str | None = None

    def to_dict(self) -> dict:
        return {"text": self.text, "source": self.source, "valid": self.valid, "attempts": self.attempts,
                "reason": self.reason, "problems": list(self.problems), "model": self.model}


def _run(system: str, user: str, llm, check, fallback_text: str, fallback_check, max_tries: int) -> Explanation:
    model = getattr(llm, "model", getattr(llm, "name", None)) if llm is not None else None

    def fallback(reason: str, attempts: int, problems: list[str]) -> Explanation:
        res = fallback_check(fallback_text)
        return Explanation(fallback_text, "fallback", res.ok, attempts, reason, problems, model)

    if llm is None:
        return fallback("No LLM is configured, so the plain evidence summary is shown.", 0, [])
    problems: list[str] = []
    prev = ""
    for attempt in range(1, max(1, max_tries) + 1):
        sys_p, usr_p = (system, user) if attempt == 1 else build_retry_prompt(system, user, prev, problems)
        try:
            prev = llm.generate(sys_p, usr_p)
        except LLMError as err:
            return fallback(f"The LLM could not be used ({err}); the plain evidence summary is shown.",
                            attempt, problems)
        except Exception as err:                              # never let an LLM problem crash the app
            return fallback(f"Unexpected LLM problem ({err.__class__.__name__}); the plain evidence "
                            f"summary is shown.", attempt, problems)
        res: ValidationResult = check(prev)
        if res.ok:
            return Explanation(prev.strip(), "llm", True, attempt, "ok", [], model)
        problems = res.problems
    return fallback("The LLM's text did not pass the number/citation checks, so the plain evidence "
                    "summary is shown.", max(1, max_tries), problems)


def explain_evidence(bundle: EvidenceBundle, llm=None, max_tries: int = 2, question: str | None = None) -> Explanation:
    """Explain a verification result or an analyst answer. `llm` is a client with .generate(system, user),
    or None. `question` (optional) is the user's question; the answer is then written to address it."""
    system, user = build_explain_prompt(bundle, question)
    return _run(system, user, llm,
                lambda t: validate_explanation(t, bundle=bundle),
                bundle.summary_text(),
                lambda t: validate_explanation(t, bundle=bundle),
                max_tries)


def explain_forecast(fc, llm=None, max_tries: int = 2) -> Explanation:
    """Explain a forecast (status, months, range, how the range was checked)."""
    system, user = build_forecast_prompt(fc)
    return _run(system, user, llm,
                lambda t: validate_explanation(t, forecast=fc),
                fc.summary(),
                lambda t: validate_explanation(t, forecast=fc, require_citations=False),
                max_tries)
