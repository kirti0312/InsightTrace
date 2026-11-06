// Voice helpers: browser speech recognition (speech -> text) and speech synthesis (text -> speech).
// Everything takes the window / synth as an argument so it can be tested with fakes.

export function supportInfo(w = globalThis) {
  const Recognition = w.SpeechRecognition || w.webkitSpeechRecognition || null;
  return { recognition: !!Recognition, synthesis: !!(w.speechSynthesis && w.SpeechSynthesisUtterance), Recognition };
}

/** Text that is pleasant to hear: no [E1] / [F] citation marks, no markdown, one paragraph, optional length cap at a sentence end. */
export function speechText(text, { max = 900 } = {}) {
  let t = String(text ?? "")
    .replace(/\[\s*(?:E\d+|F)(?:\s*,\s*(?:E\d+|F))*\s*\]/g, "")
    .replace(/[*_`#>]/g, "")
    .replace(/\s*\n+\s*/g, " ")
    .replace(/\s{2,}/g, " ")
    .replace(/\s+([.,;:!?])/g, "$1")
    .trim();
  if (t.length > max) {
    const cut = t.slice(0, max);
    const end = Math.max(cut.lastIndexOf(". "), cut.lastIndexOf("; "));
    t = end > max * 0.4 ? cut.slice(0, end + 1) : cut.replace(/\s+\S*$/, "") + "…";
  }
  return t;
}

/** Split into pieces of at most maxLen characters, at sentence ends where possible (long utterances can stall in some browsers). */
export function chunkSentences(text, maxLen = 180) {
  const sentences = String(text ?? "").match(/[^.!?;]+[.!?;]*\s*/g) || [];
  const out = [];
  let cur = "";
  const push = () => { if (cur.trim()) out.push(cur.trim()); cur = ""; };
  for (const s of sentences) {
    if (s.length > maxLen) {
      push();
      let rest = s.trim();
      while (rest.length > maxLen) {
        let i = rest.lastIndexOf(" ", maxLen);
        if (i < maxLen * 0.4) i = maxLen;
        out.push(rest.slice(0, i).trim());
        rest = rest.slice(i).trim();
      }
      cur = rest + " ";
    } else if ((cur + s).length > maxLen) {
      push();
      cur = s;
    } else cur += s;
  }
  push();
  return out;
}

const ERRORS = {
  "not-allowed": "The microphone is blocked. Allow microphone access for this site in your browser, then try again.",
  "service-not-allowed": "The browser's speech service is not allowed here. You can type your question instead.",
  "no-speech": "I did not hear anything. Press the microphone and speak clearly.",
  "audio-capture": "No microphone was found. Connect one, or type your question instead.",
  network: "The browser's speech service could not be reached. Check your internet connection or type your question.",
  "language-not-supported": "This language is not supported for speech input. Choose another language.",
};
export function errorMessage(code) {
  return ERRORS[code] || "Speech input stopped unexpectedly. You can try again or type your question.";
}

/**
 * Wraps the browser's recognizer. Callbacks: onInterim(text), onFinal(text), onError(code), onEnd().
 * Returns { supported, start(), stop() }.
 */
export function createRecognizer(w, { lang = "en-US", onInterim, onFinal, onError, onEnd } = {}) {
  const { Recognition } = supportInfo(w);
  if (!Recognition) return { supported: false, start() {}, stop() {} };
  let rec = null;
  return {
    supported: true,
    start() {
      rec = new Recognition();
      rec.lang = lang;
      rec.interimResults = true;
      rec.continuous = false;
      rec.maxAlternatives = 1;
      rec.onresult = (e) => {
        let interim = "";
        let final = "";
        for (let i = e.resultIndex; i < e.results.length; i++) {
          const r = e.results[i];
          const t = r[0]?.transcript || "";
          if (r.isFinal) final += t;
          else interim += t;
        }
        if (interim) onInterim?.(interim.trim());
        if (final.trim()) onFinal?.(final.trim());
      };
      rec.onerror = (e) => onError?.(e.error || "unknown");
      rec.onend = () => onEnd?.();
      try {
        rec.start();
      } catch {
        onError?.("unknown");
        onEnd?.();
      }
    },
    stop() {
      try { rec?.stop(); } catch { /* already stopped */ }
    },
  };
}

/** Reads text aloud piece by piece; stop() cancels cleanly and ignores stale callbacks. */
export class Speaker {
  constructor(synth, Utterance) {
    this.synth = synth;
    this.Utterance = Utterance;
    this.token = 0;
    this.speaking = false;
  }

  speak(text, { lang = "en-US", rate = 1, onStart, onEnd, max = 900 } = {}) {
    this.stop();
    const chunks = chunkSentences(speechText(text, { max }));
    if (!chunks.length || !this.synth) return false;
    const my = ++this.token;
    this.speaking = true;
    onStart?.();
    const finish = () => {
      if (my !== this.token) return;
      this.speaking = false;
      onEnd?.();
    };
    const next = (i) => {
      if (my !== this.token) return;
      if (i >= chunks.length) return finish();
      const u = new this.Utterance(chunks[i]);
      u.lang = lang;
      u.rate = rate;
      u.onend = () => next(i + 1);
      u.onerror = () => finish();
      this.synth.speak(u);
    };
    next(0);
    return true;
  }

  stop() {
    this.token++;
    const was = this.speaking;
    this.speaking = false;
    try { this.synth?.cancel(); } catch { /* ignore */ }
    return was;
  }
}

export const LANGUAGES = [["en-IN", "English (India)"], ["en-US", "English (US)"], ["en-GB", "English (UK)"], ["hi-IN", "Hindi"]];
export const RATES = [["0.85", "Slower"], ["1", "Normal"], ["1.15", "Faster"]];
