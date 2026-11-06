import Logo from "./Logo.jsx";
import { Link } from "../state/router.jsx";
import { usePrefs } from "../state/prefs.jsx";

export default function TopBar() {
  const { reduced, setReduced } = usePrefs();
  return (
    <header className="topbar">
      <Link to="/" className="brand" aria-label="InsightTrace home"><Logo /></Link>
      <nav className="nav mono" aria-label="Main">
        <Link to="/">HOME</Link>
        <Link to="/app">WORKSPACE</Link>
      </nav>
      <button
        type="button"
        className="motion-switch mono"
        role="switch"
        aria-checked={reduced}
        onClick={() => setReduced(!reduced)}
        title="Turn animations down for comfort or speed"
      >
        <span className="switch-dot" aria-hidden="true" />
        {reduced ? "MOTION: CALM" : "MOTION: FULL"}
      </button>
    </header>
  );
}
