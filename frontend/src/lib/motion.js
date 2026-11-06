// The "reduce motion" setting: the person's explicit choice wins, otherwise the system setting.
export const KEY = "insighttrace.reduceMotion";

export function readPref(storage) {
  try {
    const v = storage?.getItem(KEY);
    return v === "1" ? true : v === "0" ? false : null;
  } catch {
    return null;                       // storage can be blocked (private windows): behave as "no choice made"
  }
}

export function writePref(storage, value) {
  try {
    if (value === null) storage?.removeItem(KEY);
    else storage?.setItem(KEY, value ? "1" : "0");
  } catch {
    /* ignore: the setting then lasts only for this visit */
  }
}

export function resolveReduced(pref, systemPrefersReduced) {
  return pref === null || pref === undefined ? !!systemPrefersReduced : !!pref;
}
