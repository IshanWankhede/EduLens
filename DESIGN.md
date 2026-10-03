# EduLens: Design System

## 1. Philosophy

Academic + technology. Calm, premium, minimal. The interface should make statistics feel approachable without hiding rigor: strong hierarchy, generous spacing, and every chart or result paired with a plain-English reading. Decoration never competes with data.

Principles: **clarity over flash · consistency · honest visuals (no misleading axes) · motion only to orient.**

## 2. Color Tokens (CSS variables on `:root`)

Values are proposed design choices. Contrast ratios must be checked with a tool (e.g., browser dev tools / WebAIM) during Phase 9 and adjusted; they are not asserted here.

```css
:root {
  /* Background & surfaces */
  --bg:            #0B1020;   /* midnight navy */
  --bg-gradient:   radial-gradient(1200px 600px at 10% -10%, #1B2250 0%, transparent 60%),
                   radial-gradient(900px 500px at 100% 0%, #2A1B5C 0%, transparent 55%),
                   var(--bg);
  --surface:       #11172E;
  --card:          rgba(255,255,255,0.05);   /* glass */
  --card-solid:    #151C38;
  --border:        rgba(255,255,255,0.10);

  /* Brand */
  --primary:       #6366F1;   /* indigo */
  --primary-soft:  #818CF8;   /* for text/links on dark */
  --secondary:     #8B5CF6;   /* violet */
  --accent-blue:   #3B82F6;   /* electric blue */
  --accent-cyan:   #22D3EE;   /* cyan accent, used sparingly */
  --gradient:      linear-gradient(135deg, #6366F1 0%, #3B82F6 55%, #22D3EE 100%);

  /* Text */
  --text:          #E8ECF8;
  --text-muted:    #A3ADC8;

  /* Semantic (always paired with icon + text) */
  --info:          #60A5FA;
  --notice:        #FBBF24;
  --evidence:      #34D399;   /* "evidence against H0": not "pass" */
  --no-evidence:   #94A3B8;   /* "insufficient evidence": not "fail" */

  /* Effects */
  --shadow:        0 10px 30px rgba(3, 6, 20, 0.45);
  --shadow-hover:  0 16px 40px rgba(3, 6, 20, 0.6);
  --radius:        16px;
  --radius-sm:     10px;
  --blur:          12px;
}
```

**Light mode:** out of scope for v1; tokens are structured so a second set can be added later.

## 3. Typography

- Headings: **Sora** or **Plus Jakarta Sans**; body/UI: **Inter**; numerals/code: **JetBrains Mono** (tabular numbers for tables and metric cards).
- Loaded from Google Fonts with system fallbacks (`system-ui, -apple-system, Segoe UI, sans-serif`). If offline use matters, vendor the fonts into `assets/`.
- Scale (rem): 0.875 (caption) · 1 (body) · 1.125 · 1.5 (H3) · 2 (H2) · 3 (hero H1, clamp for mobile).
- Body line-height 1.6; minimum body size 16 px; max text width ~70ch.

## 4. Spacing, Radius, Shadow

- 4-px base scale: 4, 8, 12, 16, 24, 32, 48, 64.
- Cards: `--radius` (16 px); inputs/buttons/chips: `--radius-sm` (10 px).
- Elevation: base card shadow `--shadow`; hover `--shadow-hover` with 2–4 px upward translate.

## 5. Components

| Component | Spec |
|---|---|
| **Card** | Glass: `background: var(--card); border: 1px solid var(--border); backdrop-filter: blur(var(--blur));` fallback to `--card-solid` where `backdrop-filter` is unsupported |
| **Metric card** | Label (muted, caption), large number (mono, tabular), optional note; animated count-up (CSS counter or lightweight JS-free approach); shows exact value as text |
| **Probability card** | Large percentage + fraction (e.g., `n(A∩B) / n(B)`); label states the formula, e.g., "P(High \| Study time ≥ 3)" |
| **Status chip** | Icon + text ("Evidence against H₀" / "Insufficient evidence"); never color only |
| **Assumptions panel** | Collapsible; list of assumptions with check result (✔ / ⚠ with text) and suggested alternative |
| **Pipeline strip** | DATA → STATISTICS → PROBABILITY → MODELLING → INSIGHTS, chevron-linked cards (stacks vertically on narrow screens) |
| **Test result flow** | TEST → STATISTIC → P-VALUE → DECISION → INTERPRETATION as a five-step row/stack |
| **Disclaimer banner** | Always visible on Prediction page; verbatim text from PRD |
| **Buttons** | Primary: gradient background, 44 px min height; secondary: outline; visible focus ring (2 px `--accent-cyan`, 2 px offset) |
| **Tables** | Zebra rows at low opacity, sticky header, right-aligned numerics (mono) |

## 6. Navigation

Custom sidebar: logo/wordmark, dataset selector, CSV upload, nav list (Overview → About), settings expander (confidence level, α, thresholds, seed), footer with dataset citation. Active item has a left accent bar **and** bold text (not color alone). Navigation labels use icon + text.

## 7. Charts

- Plotly for interactive charts (scatter, box, violin, heatmap, probability bars); Matplotlib/Seaborn where static output is clearer (Q–Q, residual diagnostics).
- Shared template: transparent background, `--text` axis labels, subtle grid, font Inter.
- Palette: colorblind-safe categorical sequence (e.g., indigo, cyan, amber, magenta) plus markers/dash/pattern differences so color is never the only cue; sequential/diverging maps for heatmaps with a labeled colorbar.
- Every chart has a title, axis labels with units, and a one-line caption or "How to read this" expander.
- Truthful scales: bar charts start at zero; probabilities use a 0–100% axis.

## 8. Animations

| Use | Implementation |
|---|---|
| Section fade-in | CSS `@keyframes fadeUp` (opacity + 8 px translate, ≤ 400 ms) |
| Card hover | `transform` + shadow transition (≤ 200 ms) |
| Metric count-up | CSS `@property`-based counter where supported; static fallback |
| Progress / probability bars | CSS width transition (≤ 600 ms) |
| Background | Very slow gradient shift (≥ 20 s loop) on hero only |
| Loading | Streamlit spinner with themed text |

Rules: animate `transform`/`opacity` only; no autoplay loops elsewhere; honor `@media (prefers-reduced-motion: reduce)` by disabling all of the above.

## 9. Responsive Design

- Breakpoints: ≥ 1200 (desktop), 768–1199 (tablet), < 768 (mobile).
- Use `st.columns` with CSS to stack columns below 768 px; fluid type via `clamp()`.
- Wide tables/charts scroll horizontally inside their container; no page-level horizontal scroll.
- Mobile limits of Streamlit (e.g., sidebar collapse behavior, plot interactions) are accepted and documented.

## 10. Accessibility

- Contrast: aim for WCAG AA (4.5:1 body text, 3:1 large text/UI); verify each token pair.
- No info by color alone (icons, labels, patterns).
- Every input has a visible label and help text; `aria-label` on custom HTML controls.
- Chart titles, alt-style captions, and a data-table expander for each key chart.
- Keyboard focus visible; logical tab order; min touch target 44 px.
- Plain-language interpretation next to every statistical output.
- Reduced-motion support (above).

## 11. Streamlit CSS Strategy

1. Single stylesheet `assets/styles/edulens.css`, loaded in `ui/theme.py` and injected once with `st.markdown("<style>…</style>", unsafe_allow_html=True)`.
2. Configure base theme in `.streamlit/config.toml` (`base="dark"`, `primaryColor`, `backgroundColor`, `secondaryBackgroundColor`, `textColor`, font) so native widgets match without heavy overrides.
3. Prefer **our own classes** (`.el-card`, `.el-metric`, …) rendered through helper functions. Override Streamlit internals (`[data-testid=...]`) only when necessary and centralize them in one section of the CSS, because they may change across Streamlit versions.
4. Hide only non-essential chrome (e.g., footer); keep the header/menu accessible unless there is a reason.
5. Escape all user-derived strings before inserting them into HTML.
6. Test with the pinned Streamlit version after each upgrade.
