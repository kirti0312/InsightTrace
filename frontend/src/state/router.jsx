import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

const Ctx = createContext({ path: "/", go: () => {} });

export function normalise(p) {
  const s = String(p || "/").split("?")[0].split("#")[0].replace(/\/+$/, "");
  return s === "" ? "/" : s;
}

export function RouterProvider({ children }) {
  const [path, setPath] = useState(() => normalise(window.location.pathname));
  useEffect(() => {
    const on = () => setPath(normalise(window.location.pathname));
    window.addEventListener("popstate", on);
    return () => window.removeEventListener("popstate", on);
  }, []);
  const go = useCallback((to) => {
    const p = normalise(to);
    if (p === normalise(window.location.pathname)) return;
    window.history.pushState({}, "", p);
    setPath(p);
    window.scrollTo(0, 0);
  }, []);
  const value = useMemo(() => ({ path, go }), [path, go]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useRouter = () => useContext(Ctx);

export function Link({ to, children, className, ...rest }) {
  const { go, path } = useRouter();
  return (
    <a
      href={to}
      className={`${className || ""}${normalise(to) === path ? " is-active" : ""}`}
      onClick={(e) => {
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
        e.preventDefault();
        go(to);
      }}
      {...rest}
    >
      {children}
    </a>
  );
}
