import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { createClient } from "../api/client.js";

const Ctx = createContext(null);
const SID_KEY = "insighttrace.sid";

function store() {
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

export function SessionProvider({ children, client: injected }) {
  const client = useMemo(() => injected || createClient({ baseUrl: import.meta.env?.VITE_API_URL || "" }), [injected]);
  const [summary, setSummary] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [restoring, setRestoring] = useState(true);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  // Come back to the dataset of this browser tab (the server keeps it for a couple of hours).
  useEffect(() => {
    const sid = store()?.getItem(SID_KEY);
    if (!sid) {
      setRestoring(false);
      return;
    }
    client
      .summary(sid)
      .then((s) => alive.current && setSummary(s))
      .catch(() => store()?.removeItem(SID_KEY))
      .finally(() => alive.current && setRestoring(false));
  }, [client]);

  const adopt = useCallback((s) => {
    try {
      store()?.setItem(SID_KEY, s.session_id);
    } catch {
      /* ignore */
    }
    setSummary(s);
  }, []);

  const run = useCallback(
    async (fn) => {
      setBusy(true);
      setError("");
      try {
        adopt(await fn());
        return true;
      } catch (e) {
        setError(e.message || "Something went wrong.");
        return false;
      } finally {
        if (alive.current) setBusy(false);
      }
    },
    [adopt],
  );

  const loadDemo = useCallback((name = "synthetic") => run(() => client.demo(name)), [run, client]);
  const upload = useCallback((file) => run(() => client.upload(file)), [run, client]);
  const clear = useCallback(() => {
    try {
      store()?.removeItem(SID_KEY);
    } catch {
      /* ignore */
    }
    setSummary(null);
    setError("");
  }, []);
  const expired = useCallback(() => {
    clear();
    setError("Your session expired. Please load the data again.");
  }, [clear]);

  const value = useMemo(
    () => ({ client, summary, busy, error, restoring, loadDemo, upload, clear, expired, setError }),
    [client, summary, busy, error, restoring, loadDemo, upload, clear, expired],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useSession = () => useContext(Ctx);
