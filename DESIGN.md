# EduLens: Design System

## 1. Philosophy

EduLens uses a quiet, premium charcoal-and-plum visual language inspired by glass dashboard
interfaces. The chrome should frame—not decorate or distort—the analysis. Every chart and result
retains its title, units, caption, and methodological context.

**Clarity over flash · one continuous workspace · honest data display · motion only to orient.**

The UI-1 redesign replaces the former navy/indigo palette with a near-black charcoal canvas, soft
mauve corner glows, translucent plum-tinted cards, and a restrained plum gradient for active and
primary states. This visual change does not alter statistical calculations or data.

## 2. Color Tokens

The foreground/background ratios below use the WCAG 2.2 relative-luminance contrast formula,
computed for each foreground token against the opaque background and surface/card fallback. Ratios
are measured values, not estimates; translucent glass layers are assessed against the opaque
`--card-solid` fallback because the glow and underlying content affect their final composite.

| Foreground token | On `--bg` | On `--surface` | On `--card-solid` |
|---|---:|---:|---:|
| `--text` | 16.95:1 | 15.90:1 | 14.71:1 |
| `--muted` | 8.95:1 | 8.39:1 | 7.76:1 |
| `--primary` | 10.61:1 | 9.95:1 | 9.21:1 |
| `--secondary` | 9.49:1 | 8.90:1 | 8.23:1 |
| `--focus` | 12.76:1 | 11.96:1 | 11.07:1 |
| `--positive` | 10.86:1 | 10.19:1 | 9.43:1 |
| `--warning` | 12.02:1 | 11.27:1 | 10.42:1 |

Each listed pair exceeds WCAG AA's 4.5:1 normal-text target. Decorative borders and glow colors
are not text foregrounds and are not included as text contrast pairs. White text on the active-pill
gradient also exceeds 4.5:1 at the three measured stops (`#6F5278`: 6.66:1, `#855F7F`: 5.33:1,
`#604762`: 8.16:1).

```css
:root {
  --bg: #111014;
  --bg-gradient: radial-gradient(ellipse at 0% 0%, rgba(112, 70, 112, 0.26), transparent 38%),
                 radial-gradient(ellipse at 100% 100%, rgba(92, 58, 91, 0.22), transparent 42%),
                 #111014;
  --surface: #19171D;
  --card: rgba(33, 30, 37, 0.82);
  --card-solid: #211E25;
  --border: rgba(245, 230, 248, 0.12);
  --primary: #D9B4F2;
  --secondary: #E2A6C7;
  --text: #F5F1F5;
  --muted: #B9AFBB;
  --gradient: linear-gradient(135deg, #6F5278 0%, #855F7F 54%, #604762 100%);
  --shadow: 0 18px 50px rgba(0, 0, 0, 0.36);
  --radius: 22px;
}
```

Semantic accents pair color with text and an icon: positive `#77D6B1`, warning `#F2C879`, and
information `#B7A0D1`. Color is never the only signal. Light mode remains out of scope.

## 3. Typography

- Interface/headings: Inter, then Plus Jakarta Sans, then system sans-serif fallbacks.
- Numbers: tabular numerals with a system monospace fallback for metric values.
- Hierarchy: large page titles, bold computed metrics, quiet uppercase/compact labels, and muted
  explanatory captions.
- Body line-height 1.6, minimum 16 px, and a maximum reading width around 70ch.
- No remote font is required; offline operation uses local/system fallbacks.

## 4. Surfaces, Spacing, Radius, Shadow

- App shell: a broad dark-glass panel with a subtle border, large radius, and inner highlight.
- Sidebar: a separate charcoal panel with rounded corners and a soft inset glow.
- Cards: translucent dark surface, 1 px low-opacity border, faint inset highlight, and soft shadow.
- Primary cards use 20–24 px radii; controls and chips are pill-shaped.
- Spacing follows a 4 px base scale: 4, 8, 12, 16, 24, 32, 48, 64.
- Hover lift is a small upward transform; transitions animate transform/opacity only.

## 5. Components

| Component | Spec |
|---|---|
| **Glass card** | Dark translucent surface, subtle border, inner glow, blur where available, opaque fallback |
| **Metric card** | Small muted label, large bold computed number, optional arrow + text delta chip, optional explanatory note |
| **Probability card** | Prominent percentage plus its empirical formula/count fraction |
| **Status chip** | Icon + explicit status text; never color only |
| **Pill tabs** | Rounded segmented tab buttons with a plum active state and keyboard focus ring |
| **Range selector** | Labelled pill options; options are supplied from real context, never invented result values |
| **Assumptions panel** | Collapsible list with check status, explanatory text, and alternatives |
| **Navigation** | Brand and dataset controls first; icon-and-text page links; active link is a plum gradient pill; About and Settings stay at the sidebar bottom |
| **Pipeline strip** | DATA → STATISTICS → PROBABILITY → MODELLING → INSIGHTS; stacks on narrow screens |
| **Test result flow** | TEST → STATISTIC → P-VALUE → DECISION → INTERPRETATION |
| **Tables** | Wide tables scroll inside their own container; numeric values remain tabular |
| **Inputs** | Visible labels, rounded charcoal surfaces, descriptive help, visible keyboard focus |

## 6. Navigation and App Shell

The app remains an 11-page Streamlit multipage application. The shell groups the existing page
links into Start here, Explore, and Models. About and Settings are placed after those links at the
bottom of the sidebar. Active page links use the plum gradient and retain icon and text labels.
The main workspace is framed as one rounded glass panel.

## 7. Charts

- Plotly remains the interactive chart library; Matplotlib/Seaborn remain for static diagnostics.
- Transparent plotting surfaces, thin plum-white lines, restrained plum area fills, muted minimal
  grids, and a dark tooltip card define the chart treatment.
- A colorblind-safe categorical sequence is retained, with marker/pattern/label distinctions.
- Every figure keeps its title, axis labels, and explanatory caption.
- Honest scales remain unchanged: bars start at zero and probabilities use a 0–100% axis.
- The highlighted point and hover details are data-driven from the plotted observations; the theme
  does not add synthetic observations or example values.

## 8. Motion

- Section entrance: brief fade-up, using opacity and transform.
- Cards: hover lift using transform; opacity transitions support subtle state changes.
- Metrics: brief value fade; the exact computed value is always present as text.
- Background: slow drift of the plum glow using opacity and transform only.
- No JavaScript or autoplay chart animation is added.
- `prefers-reduced-motion: reduce` disables animation and transitions throughout the app.

## 9. Responsive Design

- Desktop uses the existing Streamlit columns and broad centered app shell.
- At narrow widths, columns collapse to one column, cards and controls expand to available width,
  and the pipeline stacks.
- Dataframes and wide tables scroll horizontally within their Streamlit frame rather than widening
  the page.
- Streamlit's mobile sidebar collapse behavior is retained.

## 10. Accessibility

- Text contrast token pairs are listed and measured in §2.
- Keyboard focus is visibly indicated with a high-contrast plum outline.
- Status and delta chips include explicit text and icons in addition to color.
- Controls remain labelled by Streamlit labels; custom HTML receives escaped user-provided text.
- Chart titles, axis labels, captions, and truthful scales are preserved.
- Interactive targets have a 44 px minimum height where applicable.
- Reduced-motion settings are honored.

## 11. Streamlit CSS Strategy

1. `assets/styles/edulens.css` is the canonical stylesheet, injected by `src/ui/theme.py`.
2. `.streamlit/config.toml` mirrors the dark plum palette for native Streamlit widgets.
3. Shared UI primitives live in `src/ui/components.py`; page scripts keep the existing routes.
4. Streamlit internal selectors remain centralized in the stylesheet because their DOM may change.
5. All interpolated user text in custom HTML is escaped.
6. The redesign changes presentation only. Statistical functions and data outputs are untouched.
