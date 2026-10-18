---
name: diagnose
metadata:
  version: 0af88ad (2026-10-18)
description: >
  Diagnose espresso extraction issues by correlating machine telemetry with taste feedback.
  Use when user says: "what went wrong", "analyze that shot", "why did it taste [sour/bitter/flat]",
  "my shots are inconsistent", or asks about pressure spikes, flow issues, or extraction problems.
  Fetches shot data via analyze_shot MCP tool, interprets patterns, and provides actionable recommendations.
---

# Espresso Diagnostic Skill

Diagnose espresso extraction issues by correlating Gaggimate telemetry data with taste feedback.

*Adapted from [gaggimate-barista](https://github.com/charleshall888/gaggimate-barista) by Charlie Hall.*

## Diagnostic Workflow

### 1. GATHER Information

**Required inputs:**
- Shot ID (from `list_recent_shots` if not provided)
- Taste feedback: sour, bitter, flat, astringent, balanced, or specific descriptors
- Any visual observations (channeling, spurting, blonde early, etc.)

**Fetch telemetry:**
```
Use: analyze_shot(shot_id)
```
This returns **summary-level** diagnostics by default (resistance, channeling risk,
temperature stability, profile compliance). If you need per-phase breakdowns or
curve shape data, escalate progressively:
```
Use: analyze_shot(shot_id, detail="per_phase")            # per-phase diagnostics (no samples)
Use: analyze_shot(shot_id, detail="per_phase_detailed")   # per-phase diagnostics + ~5 averaged samples per phase
Use: analyze_shot(shot_id, detail="time_series", resolution_hz=2)  # whole-shot curve in one pass instead of phase-by-phase
```
See the complete metric reference, annotation bands, and interpretation cheat sheet:
```
Use: read_knowledge(action="read", filename="diagnostics/SHOT_DIAGNOSTICS_REFERENCE")
```

**Fetch shot notes if available:**
```
Use: manage_shot_notes(shot_id, action="get")
```

**Always ensure output weight is logged.** If the shot notes or telemetry don't include dose out, ask the user: "What was the output weight in the cup?" The brew ratio is essential for diagnosis. Log it via `manage_shot_notes(shot_id, action="update", dose_out=X)` once obtained.

### 1b. CHECK Coffee History

If the coffee being diagnosed is known, load its tracking file to check for patterns across recent shots:

1. List available coffees: `manage_coffee(action="list")`
2. Load the relevant file: `manage_coffee(action="read", coffee_name="<name>")`
3. Compare the current shot against previous entries in the shot log table

**Why this matters:** Trend analysis helps distinguish between one-off puck prep issues and systematic dialing problems. For example, if the last 3 shots were all sour, that's a persistent under-extraction pattern requiring a larger grind adjustment — not a one-off fluke.

### 1c. IDENTIFY Shot Style

Before analyzing, identify the shot style so you compare against the right expectations. Use three tiers of detection (try in order, use first that succeeds):

**Tier 1 — Fetch profile definition (preferred):**

If `analyze_shot` returns a `profile_id`, fetch the full profile:
```
Use: manage_profile(action="get", profile_id=<profile_id>)
```

Classify by phase structure:
- Has a phase with pump off (`"target": "power"` and `"pressure": 0`) → **Bloom**
- Flow-targeted extraction with flow >= 3.5 ml/s → **Turbo**
- Brew phase with pressure declining >= 4 bar over >= 15s → **Lever Decline**
- Extraction pressure <= 7 bar + high volumetric target (>= dose × 3.5) → **Allongé**
- Extraction pressure < 9 bar, no bloom, no decline → **Dark/Gentle**
- Default → **Classic 9-Bar**

**Tier 2 — Profile name keywords (fallback):**

Match `profile_name` from `analyze_shot`:
- Contains "turbo" → **Turbo**
- Contains "bloom", "natural bloom" → **Bloom**
- Contains "allongé", "allonge", "lungo" → **Allongé**
- Contains "lever", "decline" → **Lever Decline**
- Contains "gentle", "dark", "milk" → **Dark/Gentle**
- Otherwise → **Classic 9-Bar**

**Tier 3 — Telemetry fingerprint (last resort):**

Load the telemetry patterns file and see the Style Detection Fingerprints section:
```
Use: read_knowledge(action="read", filename="diagnostics/TELEMETRY_PATTERNS")
```

### 2. ANALYZE Telemetry

**Determine the final weight.** Prefer telemetry data — load `read_knowledge(action="read", filename="diagnostics/TELEMETRY_PATTERNS")` for scale artifact detection and estimation methods. If telemetry is unavailable or looks unreliable (spikes, drops to 0g, nulls), ask the user for the actual weight.
State the dose out you're using and how you derived it, then move on.

**Load style-specific expectations** from `read_knowledge(action="read", filename="PRESSURE_GUIDE")` (Pressure by Shot Style) and `read_knowledge(action="read", filename="PROFILE_LIBRARY")` (Quick Reference table). Compare telemetry against style-specific ranges — not generic 9-bar ranges.

**Universal thresholds** (style-independent):

| Metric | Normal | Anomaly |
|--------|--------|---------|
| Temperature variance | ±1°C from target | >2°C = equipment instability |
| Pressure spike above profile target | — | >1.0 bar above = almost certainly too fine / channeling (>0.5 unusual) |

Flag an anomaly only when a metric falls **outside the identified style's expected range**.

### 2b. COMPARE Intended vs Actual (when profile definition available)

**Profile compliance metrics** are included in the diagnostics automatically
when target pressure/flow data is in the shot. Key fields:
- `pressure_rmse_bar` — how closely the machine followed target pressure (lower = better)
- `max_pressure_overshoot_bar` — largest single overshoot above target
- `flow_rmse_ml_s` — flow adherence (when target flow data available)
- `max_flow_overshoot_ml_s` — largest flow above target (when target flow data available)
- `max_flow_undershoot_ml_s` — largest flow below target (when target flow data available)

> **Grind diagnosis priority:** Flow deviation is a more reliable grind indicator
> than pressure overshoot. The Gaggimate/gaggiuino PID actively controls pump power
> to maintain target pressure, so pressure overshoot is artificially limited by the
> controller. Flow rate is a *consequence* of grind + dose + puck prep and cannot be
> masked by the pump — making `max_flow_overshoot_ml_s` / `max_flow_undershoot_ml_s`
> the stronger signal for grind mismatch. Always check flow deviation first when
> diagnosing grind issues.

| Comparison | Interpretation |
|------------|----------------|
| `max_flow_overshoot_ml_s` > 0.7 | Grind too coarse — flow significantly above target (strongest grind signal) |
| `max_flow_undershoot_ml_s` > 0.7 | Grind too fine — flow significantly below target (strongest grind signal) |
| `max_pressure_overshoot_bar` > 1.0 | Grind too fine — machine can't push water through puck (>0.5 already unusual) |
| `max_pressure_undershoot_bar` > 1.0 (non-bloom context) | Grind too coarse |
| `pressure_rmse_bar` annotation POOR | Profile not being followed — check grind, dose, or profile params |
| Pressure exceeded target by >1.0 bar | Grind too fine or dose too high (highly unusual overshoot) |
| Pressure never reached target (>1.0 bar below) | Context-dependent: normal for post-bloom ramps, too coarse for non-bloom |
| Volumetric target reached much earlier than phase duration | Grind too coarse (flow too fast) |
| Volumetric target not reached within phase duration | Grind too fine (flow too slow) |
| Decline phase pressure stayed >1 bar above target floor | Grind too fine — high puck resistance |

### 3. CORRELATE Taste with Telemetry

Cross-reference the user's taste feedback with telemetry patterns.

**See:** `read_knowledge(action="read", filename="diagnostics/TELEMETRY_PATTERNS")` for detailed correlation matrix and `read_knowledge(action="read", filename="diagnostics/DIAGNOSTIC_TREES")` for taste-based decision trees.

### 4. RECOMMEND Actions

Provide specific, prioritized recommendations:
1. **Primary adjustment** (most likely to fix the issue)
2. **Secondary adjustment** (if primary doesn't work)
3. **Profile consideration** (if extraction mechanics need changing)

Always explain WHY each recommendation addresses the diagnosed issue.

---

## Response Format

```
## Shot Analysis: [Shot ID]

### Identified Style: [Style Name]
(detected via [Tier 1: profile definition / Tier 2: profile name / Tier 3: telemetry fingerprint])

### Telemetry Summary
- **Pressure:** [peak] bar (style expects: X-Y bar)
- **Flow:** [avg] ml/s, first drip at [X]s (style expects: A-B ml/s)
- **Temperature:** [avg]°C (target: X°C, variance: ±X°C)
- **Timing:** [total]s total (style expects: E-Fs)

### Phase Comparison (if profile definition available)
[Intended vs actual for each phase — pressure, flow, timing]

### Diagnosis
[Correlation between telemetry and reported taste, interpreted through style-specific expectations]

### Recommendations
1. **[Primary fix]** — [specific action with reasoning]
2. **[Secondary fix]** — [backup if primary doesn't work]
3. **[Profile consideration]** — [if applicable]

### What to Watch For
[What user should observe on next shot to confirm diagnosis]
```
