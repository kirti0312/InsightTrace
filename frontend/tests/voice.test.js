import test from "node:test";
import assert from "node:assert/strict";
import { supportInfo, speechText, chunkSentences, errorMessage, createRecognizer, Speaker, LANGUAGES } from "../src/lib/voice.js";

test("support detection", () => {
  assert.deepEqual(supportInfo({}), { recognition: false, synthesis: false, Recognition: null });
  const R = function () {};
  assert.equal(supportInfo({ webkitSpeechRecognition: R }).recognition, true);
  assert.equal(supportInfo({ speechSynthesis: {}, SpeechSynthesisUtterance: function () {} }).synthesis, true);
  assert.equal(supportInfo({ speechSynthesis: {} }).synthesis, false);
});

test("speech text drops citations and markdown, and caps at a sentence end", () => {
  assert.equal(speechText("Revenue fell by 5. [E1]\nThe cause is *unknown* [E2, E3]."), "Revenue fell by 5. The cause is unknown.");
  assert.equal(speechText("Up [F] ."), "Up.");
  assert.equal(speechText(null), "");
  const long = "First sentence here. ".repeat(100);
  const t = speechText(long, { max: 100 });
  assert.ok(t.length <= 100 && t.endsWith("."));
  const noStops = speechText("word ".repeat(100), { max: 50 });
  assert.ok(noStops.length <= 51 && noStops.endsWith("…"));
});

test("chunks respect the maximum length", () => {
  const text = "One short one. Two is also short. " + "word ".repeat(80) + "end.";
  const c = chunkSentences(text, 60);
  assert.ok(c.length > 3);
  assert.ok(c.every((x) => x.length <= 61), JSON.stringify(c.map((x) => x.length)));
  assert.equal(c.join(" ").replace(/\s+/g, " ").includes("One short one."), true);
  assert.deepEqual(chunkSentences("", 50), []);
  assert.deepEqual(chunkSentences("Hi there.", 50), ["Hi there."]);
  assert.ok(chunkSentences("x".repeat(500), 100).every((x) => x.length <= 100));
});

test("error messages are friendly and always present", () => {
  assert.match(errorMessage("not-allowed"), /microphone is blocked/);
  assert.match(errorMessage("no-speech"), /did not hear/);
  assert.match(errorMessage("something-new"), /stopped unexpectedly/);
});

function fakeWindow() {
  const made = [];
  function R() { made.push(this); this.started = false; }
  R.prototype.start = function () { this.started = true; };
  R.prototype.stop = function () { this.stopped = true; };
  return { w: { SpeechRecognition: R }, made };
}

test("recognizer reports interim and final text, errors and the end", () => {
  const { w, made } = fakeWindow();
  const log = [];
  const rec = createRecognizer(w, { lang: "en-IN", onInterim: (t) => log.push(["i", t]), onFinal: (t) => log.push(["f", t]), onError: (c) => log.push(["e", c]), onEnd: () => log.push(["end"]) });
  assert.equal(rec.supported, true);
  rec.start();
  const r = made[0];
  assert.equal(r.lang, "en-IN"); assert.equal(r.interimResults, true); assert.equal(r.started, true);
  const res = (text, isFinal) => Object.assign([{ transcript: text }], { isFinal });
  r.onresult({ resultIndex: 0, results: [res("why did rev", false)] });
  r.onresult({ resultIndex: 0, results: [res("why did revenue fall", true)] });
  r.onerror({ error: "no-speech" });
  r.onend();
  assert.deepEqual(log, [["i", "why did rev"], ["f", "why did revenue fall"], ["e", "no-speech"], ["end"]]);
  rec.stop(); assert.equal(r.stopped, true);
  assert.equal(createRecognizer({}, {}).supported, false);
});

test("recognizer survives start() throwing", () => {
  function R() {}
  R.prototype.start = function () { throw new Error("already started"); };
  const log = [];
  createRecognizer({ SpeechRecognition: R }, { onError: (c) => log.push(c), onEnd: () => log.push("end") }).start();
  assert.deepEqual(log, ["unknown", "end"]);
});

function fakeSynth() {
  const spoken = [];
  const synth = { spoken, cancelled: 0, speak(u) { spoken.push(u); }, cancel() { this.cancelled++; } };
  function U(text) { this.text = text; }
  return { synth, U };
}

test("speaker reads chunk by chunk and reports the end", () => {
  const { synth, U } = fakeSynth();
  const sp = new Speaker(synth, U);
  const ev = [];
  assert.equal(sp.speak("First sentence. Second sentence. [E1]", { lang: "en-IN", rate: 1.15, onStart: () => ev.push("start"), onEnd: () => ev.push("end") }), true);
  assert.equal(synth.spoken.length, 1);
  assert.equal(synth.spoken[0].lang, "en-IN"); assert.equal(synth.spoken[0].rate, 1.15);
  assert.equal(sp.speaking, true);
  synth.spoken[0].onend();
  assert.equal(synth.spoken.length, 1);              // short sentences share one chunk
  assert.deepEqual(ev, ["start", "end"]);
  assert.equal(sp.speaking, false);
});

test("stopping ignores late callbacks from the old text", () => {
  const { synth, U } = fakeSynth();
  const sp = new Speaker(synth, U);
  const ev = [];
  sp.speak("A ".repeat(200) + "done.", { onEnd: () => ev.push("end1"), max: 2000 });
  const first = synth.spoken[0];
  assert.ok(sp.stop());
  first.onend();                                      // a late event from the cancelled speech
  first.onerror();
  assert.deepEqual(ev, []);
  assert.equal(synth.spoken.length, 1);
  assert.equal(sp.speak("", {}), false);
  assert.equal(new Speaker(null, U).speak("hello there."), false);
  assert.ok(LANGUAGES.some(([c]) => c === "en-IN"));
});
