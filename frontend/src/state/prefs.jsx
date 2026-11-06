import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { readPref, writePref, resolveReduced } from "../lib/motion.js";

const Ctx = createContext({ reduced: false, setReduced: () => {}, small: false });

function safeStorage() {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

export function PrefsProvider({ children }) {
  const [pref, setPref] = useState(() => readPref(safeStorage()));
  const [system, setSystem] = useState(() => !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);
  const [small, setSmall] = useState(() => window.innerWidth < 760);

  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    const on = () => setSystem(!!mq.matches);
    mq?.addEventListener?.("change", on);
    const onResize = () => setSmall(window.innerWidth < 760);
    window.addEventListener("resize", onResize);
    return () => {
      mq?.removeEventListener?.("change", on);
      window.removeEventListener("resize", onResize);
    };
  }, []);

  const reduced = resolveReduced(pref, system);
  useEffect(() => {
    document.documentElement.dataset.motion = reduced ? "reduced" : "full";
  }, [reduced]);

  const setReduced = useCallback((v) => {
    setPref(v);
    writePref(safeStorage(), v);
  }, []);

  const value = useMemo(() => ({ reduced, setReduced, small }), [reduced, setReduced, small]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const usePrefs = () => useContext(Ctx);
