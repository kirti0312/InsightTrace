// InsightTrace logo (concept A): a lens, a trace line running through it, and a glowing node at the cause.
export function LogoMark({ size = 32 }) {
  return (
    <svg className="logo-mark" width={size} height={size} viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <defs>
        <linearGradient id="lg1" x1="6" y1="6" x2="42" y2="42">
          <stop offset="0" stopColor="#38BDF8" />
          <stop offset="0.55" stopColor="#8B5CF6" />
          <stop offset="1" stopColor="#F472B6" />
        </linearGradient>
      </defs>
      <circle cx="21" cy="21" r="14" stroke="url(#lg1)" strokeWidth="3" />
      <path d="M31.5 31.5 42 42" stroke="url(#lg1)" strokeWidth="3.5" strokeLinecap="round" />
      <path d="M11 24 16 24 19 15 24 29 27 21 31 21" stroke="#F5EEED" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <circle className="logo-node" cx="24" cy="29" r="2.6" fill="#F472B6" />
    </svg>
  );
}

export default function Logo({ size = 30 }) {
  return (
    <span className="logo" aria-label="InsightTrace">
      <LogoMark size={size} />
      <span className="logo-word">
        INSIGHT<b>TRACE</b>
      </span>
    </span>
  );
}
