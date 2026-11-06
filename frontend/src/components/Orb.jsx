/** Stand-in for the 3D robot "Trace" (the real Three.js robot arrives in Step 13b). Pure CSS glow. */
export default function Orb({ style }) {
  return (
    <div className="orb" style={style} aria-hidden="true">
      <span className="orb-core" />
      <span className="orb-ring r1" />
      <span className="orb-ring r2" />
    </div>
  );
}
