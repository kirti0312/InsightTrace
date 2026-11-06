# InsightTrace UI design PROPOSAL (for the user's approval; nothing is built yet)

Based on the user's brief (cinematic dark 3D site, particle headline, recurring 3D robot, scroll-driven scenes,
glass analytics panels) and the style notes in docs/DESIGN_REFERENCE.md (NOPAL screenshots, used as inspiration only).
Everything below is TEXT so it can be approved or changed before any React code is written.

---------------------------------------------------------------------------------------------------
## 0. Two things to decide first (my recommendations)

1. ONE SITE, TWO MODES.
   - "/" = the cinematic story page (scroll scenes 1 to 9 below). Its panels show REAL answers from the demo dataset
     (loaded in the background through the API), never made-up numbers.
   - "/app" = the working screen for the user's own file: same colours, glass panels and a small robot, but a plain,
     fast layout (tabs: Overview, Analyst, Root Cause, Forecast, Voice) and only a light particle background.
   Reason: heavy scroll-jacking 3D is great for a first impression and bad for daily work, and the dashboard must stay
   functional on weak laptops and phones.
2. THE ROBOT IS BUILT IN CODE (no downloaded model): rounded head, glowing visor, floating body and hands,
   antenna ring. No licence, no download size, easy to pose; it can be replaced by a .glb model later.

---------------------------------------------------------------------------------------------------
## 1. Look and feel

Palette (from the reference, plus the electric blue you asked for)
| Role | Value |
|---|---|
| Background | #09060B to #150D17 (near-black plum), subtle vignette |
| Violet glow | #903E6F -> #6D28D9 (violet), bright core #C8578B |
| Pink accent | #F472B6 / #BF8AAA |
| Electric blue accent | #38BDF8 / #3B82F6 (lines, data highlights, robot eyes) |
| Text | #F5EEED (never pure white); muted text #B9AEB5 |
| Verdict colours | CAUSE = electric blue #38BDF8, NOT_A_CAUSE = muted grey-pink #8B7F88, UNKNOWN = amber #FBBF24 |
| Anomaly | hot pink-red #FB4F7A pulse |
Glass panel: translucent dark (rgba 20,12,28,0.55), 1 px cream-violet border at 18 percent, blur 16 px, soft inner glow.
Type: display = wide geometric grotesque, uppercase, regular weight (free option: "Space Grotesk" or "Unbounded" wide
cut via Google Fonts); small text and numbers = uppercase-style monospace ("JetBrains Mono" / "IBM Plex Mono").
Callout tags (from the reference): cream box, dark mono text, tiny corner brackets, thin leader line to a dot;
they "draw in" when a chart point or claim settles.

Logo (appears top-left on every screen, and as the favicon)
- Mark: a glowing lens ring; through it runs a thin "trace" line that rises, dips and ends in a pulsing node (the anomaly).
  Violet -> pink -> blue gradient stroke on dark. The node pulses slowly.
- Wordmark: INSIGHTTRACE, wide uppercase sans, letter-spaced; "INSIGHT" regular cream, "TRACE" with the gradient.
- Delivered as inline SVG (crisp, tiny, themable). Two variants will be drawn at build time for you to pick: A = lens + trace,
  B = lens + monogram "IT".

Always-on background effects (cheap, continuous)
- Drifting sparkle particles in 3 depth layers (slow parallax, twinkle), a few "shooting data streaks" every few seconds,
  a slowly breathing violet/pink glow behind the robot, faint grid-lines that shimmer toward the horizon, soft light-sweep
  on glass panels when they appear. All of it is switched off or frozen under reduced motion.

---------------------------------------------------------------------------------------------------
## 2. Technology evaluation and choice

| Tool | Use it for | Verdict |
|---|---|---|
| Three.js with React Three Fiber (+ drei, postprocessing) | particle headline, robot, glow, camera | YES: real-time 3D inside React |
| GSAP ScrollTrigger | one scroll "director": turns scroll position into 0..1 progress per scene, pins the 3D stage, scrubs camera and uniforms | YES: precise and light. (GSAP is free to use under its standard licence; I will re-check its terms at deployment) |
| Framer Motion | UI motion in the DOM: cards, lists, drawers, tab changes, hover and layout animation | YES, but ONLY for panels, never for 3D |
| drei ScrollControls | alternative scroll for 3D | NO: ties scrolling to the canvas and hijacks it |
| Lenis smooth-scroll | extra smoothing | NO for now: native scrolling keeps it natural and accessible |

How they fit together
- ONE fixed full-screen <Canvas> behind the page (the 3D stage). The page's text and panels are normal HTML above it
  and scroll natively (no scroll-jacking).
- ScrollTrigger writes scene progress into a small shared store (not React state); the 3D code reads it every frame.
  So scrolling never re-renders React, which keeps it smooth.
- Headline particles: the sentence is drawn once to a hidden 2D canvas, its pixels become ~6,000 target points; a shader
  moves each particle between "text" and "scattered" using the scroll progress. (Same idea for the logo reveal.)
- Performance plan: 3D code is lazy-loaded; a CSS gradient + static headline shows instantly while it loads; pixel
  ratio capped (1.5 desktop, 1 mobile); particles 6,000 desktop / 2,500 mobile; glow (bloom) only on capable GPUs;
  adaptive quality (drops quality automatically if frames get slow); rendering pauses when the tab is hidden or
  the stage is off screen; no WebGL -> the same page with the static fallback.
- Accessibility: prefers-reduced-motion = static headline text, robot in one still pose, no parallax or camera
  moves, panels simply fade in; a visible "Reduce motion" switch for everyone. Keyboard focus rings and text
  alternatives for every chart (a data table toggle). Colour is never the only signal (verdicts also have words and icons).
- Mobile: portrait layout, headline stacked, robot smaller and lower, panels become full-width stacked cards,
  evidence opens as a bottom sheet, tap instead of hover, simplified camera moves, fewer particles.

---------------------------------------------------------------------------------------------------
## 3. The story page "/" : scroll scenes (text layouts)

Top bar on every scene:  [logo mark] INSIGHTTRACE      PRODUCT   HOW IT WORKS   TRY THE DEMO      [UPLOAD FILE]  [reduce motion]

### Scene 1 - Hero (0 percent scroll)
```
 [logo] INSIGHTTRACE            HOW IT WORKS   TRY THE DEMO        [UPLOAD FILE]
 ----------------------------------------------------------------------------------
   UNDERSTAND WHAT CHANGED.                        . * .   sparkles drift
        DISCOVER WHY.                 (headline is formed from glowing particles,
                                       gently shimmering)
        small mono line:  "Finds what moved in your business numbers, then tests
                           the cause with evidence. Says UNKNOWN when it cannot tell."
              [ TRY THE DEMO ]   [ UPLOAD YOUR FILE ]
                                                          violet-pink glow low on the right
 scroll hint: thin line + "SCROLL"
```
### Scene 2 - Particles scatter, the robot appears (10 to 25 percent)
```
   the headline's particles loosen and swirl outward in a spiral; the camera pushes in;
   from the centre of the swirl a small glass-and-chrome robot condenses and floats up
                      .  *        _____
                  *      .       /_ o o_\      eyes = electric blue, visor glows
                        .       |  ___  |      slow float + slow turn
                                 \_/ \_/
   caption (mono, bottom-left):  "MEET TRACE. IT READS YOUR NUMBERS AND SHOWS ITS WORK."
```
### Scene 3 - Load data (25 to 38 percent)  [real: POST /api/demo, /api/upload, GET summary]
```
   robot drifts left, glass panel rises on the right:
   +------------------------------------------+
   | 1  LOAD YOUR DATA                        |
   |   [ drop a CSV / Excel file here ]       |
   |   or  [ use the demo dataset ]           |
   |   after loading:  58,000 ROWS | JAN 2022 - DEC 2024 | 7 KPIs | 11 COLUMNS
   |   warnings (if any) in amber             |
   +------------------------------------------+
```
### Scene 4 - What changed (38 to 52 percent)  [real: headlines, alerts]
```
   KPI cards orbit in around the robot, then settle in a row; a line chart draws itself;
   an anomaly point pulses pink-red; a tag draws in: "REVENUE FELL 26% VS EXPECTED, AUG-NOV 2024"
   +--------+ +--------+ +--------+ +--------+        alerts list (severity dots)
   |REVENUE | |PROFIT  | |ORDERS  | |MARGIN  |        - Revenue fell ...   [ WHY? ]
   | 7.17M  | | ...    | | ...    | | 19.1%  |        - Profit margin fell ... [ WHY? ]
   +--------+ +--------+ +--------+ +--------+
```
### Scene 5 - Find out why (52 to 66 percent)  [real: POST alerts/why]
```
   the camera dives into the pink-red anomaly; the chart band unfolds into an INVESTIGATION BOARD:
   contributor bars (who moved) -> candidate nodes -> beams run from each to the robot;
   five check lights tick one by one (size, differs from rest, started in window, not a repeat, unusual for history)
   result badges land:   [ CAUSE ] Customer 242, 286, 312, 317, 330     [ NOT A CAUSE ] Technology (fell with everyone)
                         [ UNKNOWN ] ...  
```
### Scene 6 - See the proof (66 to 78 percent)  [real: evidence records + SQL + rows]
```
   a claim card is selected; leader lines connect  CLAIM -> QUERY -> ROWS
   +-------------------------+      +------------------------------------------+
   | E2  THE EVIDENCE SUPPORTS|-----| SELECT ... FROM sales WHERE ...         |
   |     these 5 customers... |      | rows: customer | before | now | change |
   +-------------------------+      +------------------------------------------+
   citations like [E2] in the explanation highlight their record on hover
```
### Scene 7 - Ask anything (78 to 88 percent)  [real: POST ask, optional voice]
```
   robot faces the viewer; chat panel; the visor shows a sound wave while it speaks
   > "Which region has the lowest profit margin?"      [mic]  [speaker on/off]
   answer text with [E#] citations, evidence rows beneath, a note when the plain wording was used
```
### Scene 8 - Look ahead (88 to 96 percent)  [real: forecast tool]
```
   history line continues into a glowing forecast line with a translucent 80 percent band;
   status badge OK / LOW CONFIDENCE / UNKNOWN; mono line: "THE BAND HELD 71% OF 129 PAST CHECKS"
   reminder: "A FORECAST EXTENDS THE PAST. IT DOES NOT KNOW ABOUT PROMOTIONS OR SHOCKS."
```
### Scene 9 - Finish (96 to 100 percent)
```
   robot waves once and settles; big line "TRY IT ON YOUR OWN NUMBERS"  [ UPLOAD YOUR FILE ]  [ OPEN THE APP ]
   footer: logo, "EVIDENCE SUPPORTS; IT DOES NOT PROVE", links
```

---------------------------------------------------------------------------------------------------
## 4. The working app "/app" (functional first)

```
 [logo] INSIGHTTRACE   OVERVIEW | ANALYST | ROOT CAUSE | FORECAST | VOICE        file: demo.csv  [reduce motion]
 ------------------------------------------------------------------------------------------------------------
 left rail (collapsible): data summary, KPI picker, date range
 main: glass panels (the same components as the story page, full size, real data, keyboard accessible)
 corner: small robot (idle float; "thinking" pulse while the API works; speaks in Voice)
 background: sparkles only (light)
```
- OVERVIEW: KPI cards + trend chart (monthly/quarterly/weekly) + alerts list with WHY? buttons.
- ANALYST: question box, suggested questions, answer + citations + evidence/SQL drawer; use-AI-wording toggle.
- ROOT CAUSE: breakdown/contributors charts, verdict cards, five checks, evidence drawer (query and rows).
- FORECAST: forecast chart with band, status, back-test line, horizon picker.
- VOICE: big mic button, transcript, the robot answers aloud (browser speech; needs Chrome/Edge for listening).

Mobile: tabs become a bottom bar; panels stack; evidence = bottom sheet; robot docked small at the top right.

---------------------------------------------------------------------------------------------------
## 5. Component list (charts and widgets)
KPI card (value, change vs last month and year) | trend line with alert markers | bar chart (breakdown) |
contributor bars (who moved) | investigation board (nodes and beams) | verdict badge | five-check strip |
evidence drawer (SQL + rows) | forecast chart with band | alert list | question box + answer with citations |
mic and speaker controls | upload dropzone + data summary | robot (states: idle, thinking, speaking, pointing, shrug for UNKNOWN) |
particle headline | logo | callout tag | reduce-motion switch.
Charts are drawn with a light library (Recharts or visx style, SVG) so they stay crisp, accessible and quick; 3D is used for
atmosphere and storytelling, not to draw the data.

---------------------------------------------------------------------------------------------------
## 6. Build plan (matches the roadmap)
- Step 13: project setup (Vite + React, JavaScript), design tokens, logo, API client, the 3D stage (particles, robot, glow),
  scroll director, Home story scenes 1 to 4 + Upload/Demo + Overview.
- Step 14: Analyst page and scenes 7.   - Step 15: Root Cause (scenes 5 and 6).   - Step 16: Forecast (scene 8).
- Step 17: Voice and the speaking robot; scene 9; polish; mobile and reduced-motion pass.
- Optional extra after 17: the talking presenter (present the findings, or explain an uploaded PPT).

## 7. Honest limits and risks
- I could NOT install the web packages (React, Three.js, GSAP) in my build sandbox (no npm access), so I cannot run or look at the site
  myself. I will write the code in small, separate pieces, test the plain logic (scroll maths, particle sampling, API client)
  with Node's built-in tests, and you run `npm install` and `npm run dev` and tell me what you see (screenshots or error text).
  Visual polish and frame rate must be judged on your machines.
- Real-time 3D costs battery and GPU: older phones will get the lighter mode automatically; very old devices get the static page.
- Speech listening works in Chrome/Edge, not in Firefox; speaking works almost everywhere.
- The story page shows real demo answers, so its first load calls the API (a free host may take a few seconds to wake up);
  the scenes show a skeleton shimmer until the data arrives.
- Free hosting limits: the 3D bundle adds download weight (about 0.5 to 1 MB compressed). It is lazy-loaded.
