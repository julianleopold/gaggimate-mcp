"""Transform shot data to AI-friendly format.

This module converts raw binary shot data into a structured format
optimized for AI analysis and natural language processing.
"""

import math
from typing import NotRequired, Optional, TypedDict

from gaggimate_mcp.parsers.shot import ShotData


class TemperatureSummary(TypedDict):
    """Temperature summary statistics."""
    min_c: float
    max_c: float
    avg_c: float
    target_avg_c: float


class PressureSummary(TypedDict):
    """Pressure summary statistics."""
    min_bar: float
    max_bar: float
    avg_bar: float
    peak_time_s: float


class FlowSummary(TypedDict):
    """Flow summary statistics."""
    total_volume_ml: float
    avg_flow_ml_s: float
    peak_flow_ml_s: float
    time_to_first_drip_s: Optional[float]


class ExtractionSummary(TypedDict):
    """Extraction timing summary."""
    preinfusion_time_s: float
    main_extraction_time_s: float
    total_time_s: float


class ShotSummary(TypedDict):
    """Complete shot summary statistics."""
    temperature: TemperatureSummary
    pressure: PressureSummary
    flow: FlowSummary
    extraction: ExtractionSummary


class TransformedSample(TypedDict):
    """Transformed sample data point."""
    time_seconds: float
    temperature_c: float
    pressure_bar: float
    flow_ml_s: float
    weight_g: float


class PhaseDiagnostics(TypedDict, total=False):
    """Per-phase diagnostic metrics. Fields vary by phase_type.

    Common fields: phase_type, avg_pressure_bar, avg_flow_ml_s,
    pressure_rmse_bar, flow_rmse_ml_s, annotations.

    Preinfusion: ramp_rate_bar_s, saturation_time_s
    Brew: resistance_avg, resistance_slope, channeling_risk,
          pressure_stability_bar, flow_stability_ml_s
    Decline: taper_rate_bar_s, taper_smoothness
    """
    phase_type: str
    avg_pressure_bar: float
    avg_flow_ml_s: float
    pressure_rmse_bar: float
    flow_rmse_ml_s: float
    # Preinfusion
    ramp_rate_bar_s: float
    saturation_time_s: float
    # Brew
    resistance_avg: float
    resistance_slope: float
    channeling_risk: str
    pressure_stability_bar: float
    flow_stability_ml_s: float
    # Decline
    taper_rate_bar_s: float
    taper_smoothness: float
    annotations: dict[str, str]


class PhaseData(TypedDict):
    """Phase data for AI analysis."""
    name: str
    phase_number: int
    start_time_seconds: float
    duration_seconds: float
    sample_count: int
    avg_temperature_c: float
    avg_pressure_bar: float
    total_flow_ml: float
    samples: NotRequired[list[TransformedSample]]
    diagnostics: NotRequired[PhaseDiagnostics]


class ResistanceDiagnostics(TypedDict):
    """Puck resistance diagnostics — the master diagnostic metric.

    Computed as pressure / flow² (quadratic Darcy model).
    Captures grind fineness, puck prep quality, channeling, and erosion.
    """
    avg: float
    std: float
    slope: float
    peak: float
    peak_timing_pct: float
    annotations: dict[str, str]


class ChannelingIndicators(TypedDict):
    """Channeling detection from pressure/flow volatility.

    High volatility in the free variable (flow in pressure mode,
    pressure in flow mode) indicates channeling.
    """
    pressure_volatility_bar: float
    flow_volatility_ml_s: float
    pressure_max_drop_rate_bar_s: float
    flow_acceleration_late_ml_s2: float
    overall_risk: str
    annotations: dict[str, str]


class TemperatureDiagnostics(TypedDict):
    """Temperature stability vs target during brew phase."""
    overshoot_c: float
    undershoot_c: float
    stability_std_c: float
    annotations: dict[str, str]


class ExtractionMetrics(TypedDict):
    """Extraction quality metrics for brew phase."""
    pressure_auc_bar_s: float
    pressure_slope_brew_bar_s: float
    flow_slope_brew_ml_s2: float
    flow_avg_brew_ml_s: float
    annotations: dict[str, str]


class WeightDiagnostics(TypedDict):
    """Weight/yield curve diagnostics from scale data."""
    rate_avg_g_s: Optional[float]
    rate_std_g_s: Optional[float]
    scale_connected: bool
    annotations: dict[str, str]


class ProfileComplianceMetrics(TypedDict):
    """Profile adherence metrics — RMSE between target and actual.

    Measures how well the machine followed the programmed profile.
    pressure_overshoot > 1.0 bar is highly unusual and almost certainly
    indicates grind too fine, excessive dose, or a puck preparation issue.

    **Important:** Flow deviation is a more reliable grind indicator than
    pressure overshoot because the Gaggimate/gaggiuino PID actively
    controls pump power to maintain target pressure.  Pressure overshoot
    is therefore artificially limited by the controller.  Flow rate,
    however, is a *consequence* of grind+dose+puck-prep and cannot be
    masked by the pump — making flow overshoot/undershoot the stronger
    signal for diagnosing grind mismatch.
    """
    pressure_rmse_bar: float
    flow_rmse_ml_s: Optional[float]
    max_pressure_overshoot_bar: float
    max_pressure_undershoot_bar: float
    max_flow_overshoot_ml_s: Optional[float]
    max_flow_undershoot_ml_s: Optional[float]
    annotations: dict[str, str]


class ShotDiagnostics(TypedDict):
    """Complete shot diagnostics for AI analysis.

    Pre-computed features with interpretive annotations for optimal
    LLM reasoning. Features are grouped by diagnostic purpose.
    """
    resistance: ResistanceDiagnostics
    channeling: ChannelingIndicators
    temperature: TemperatureDiagnostics
    extraction: ExtractionMetrics
    weight: WeightDiagnostics
    profile_compliance: Optional[ProfileComplianceMetrics]


class SummaryDiagnostics(TypedDict):
    """Lightweight diagnostics for summary detail level.

    Contains only the key indicators an LLM needs for a quick
    assessment, minimising token usage while preserving actionability.
    """
    resistance_avg: float
    resistance_slope: float
    channeling_risk: str
    temperature_stability_c: float
    pressure_rmse_bar: float
    max_overshoot_bar: float
    flow_rmse_ml_s: Optional[float]
    max_flow_overshoot_ml_s: Optional[float]
    scale_connected: bool
    annotations: dict[str, str]


class TransformedShot(TypedDict):
    """Transformed shot data for AI analysis."""
    shot_id: str
    profile_name: str
    profile_id: str
    timestamp: int
    duration_seconds: float
    final_weight_g: Optional[float]
    summary: ShotSummary
    phases: list[PhaseData]
    diagnostics: Optional[ShotDiagnostics | SummaryDiagnostics]
    detail_level: NotRequired[str]


def calculate_total_volume(samples: list[dict], interval_ms: int) -> float:
    """Calculate total volume from flow samples.

    Args:
        samples: List of sample dictionaries with 'pf' (puck flow) field
        interval_ms: Sample interval in milliseconds

    Returns:
        Total volume in ml, rounded to 1 decimal place
    """
    total_volume = 0.0
    interval_seconds = interval_ms / 1000.0

    for sample in samples:
        flow = sample.get('pf', 0.0)  # ml/s
        total_volume += flow * interval_seconds  # ml

    return round(total_volume * 10) / 10


def calculate_summary(shot: ShotData) -> ShotSummary:
    """Calculate summary statistics for shot.

    Args:
        shot: Parsed shot data

    Returns:
        Summary statistics
    """
    samples = shot.samples

    # Extract values - only include samples that have the measurement
    temperatures = [s['ct'] for s in samples if 'ct' in s]
    target_temps = [s['tt'] for s in samples if 'tt' in s]
    pressures = [s['cp'] for s in samples if 'cp' in s]
    flows = [s['pf'] for s in samples if 'pf' in s]
    times = [s.get('t', 0.0) / 1000.0 for s in samples]  # Convert to seconds

    # Temperature summary
    temp_summary = TemperatureSummary(
        min_c=round(min(temperatures) * 10) / 10 if temperatures else 0.0,
        max_c=round(max(temperatures) * 10) / 10 if temperatures else 0.0,
        avg_c=round(sum(temperatures) / len(temperatures) * 10) / 10 if temperatures else 0.0,
        target_avg_c=round(sum(target_temps) / len(target_temps) * 10) / 10 if target_temps else 0.0,
    )

    # Pressure summary
    peak_pressure = max(pressures) if pressures else 0.0
    peak_pressure_index = pressures.index(peak_pressure) if pressures and peak_pressure > 0 else 0
    peak_time = times[peak_pressure_index] if peak_pressure_index < len(times) else 0.0

    pressure_summary = PressureSummary(
        min_bar=round(min(pressures) * 10) / 10 if pressures else 0.0,
        max_bar=round(max(pressures) * 10) / 10 if pressures else 0.0,
        avg_bar=round(sum(pressures) / len(pressures) * 10) / 10 if pressures else 0.0,
        peak_time_s=round(peak_time * 10) / 10,
    )

    # Flow summary
    total_volume = calculate_total_volume(samples, shot.sample_interval)
    avg_flow = round(sum(flows) / len(flows) * 10) / 10 if flows else 0.0
    peak_flow = round(max(flows) * 10) / 10 if flows else 0.0

    # Find time to first drip (first non-zero flow)
    time_to_first_drip = None
    for i, flow in enumerate(flows):
        if flow > 0.0:
            time_to_first_drip = round(times[i] * 10) / 10 if i < len(times) else None
            break

    flow_summary = FlowSummary(
        total_volume_ml=total_volume,
        avg_flow_ml_s=avg_flow,
        peak_flow_ml_s=peak_flow,
        time_to_first_drip_s=time_to_first_drip,
    )

    # Extraction timing
    # Preinfusion: time from start until pressure reaches 50% of max
    preinfusion_time = 0.0
    if peak_pressure > 0:
        threshold = peak_pressure * 0.5
        for i, pressure in enumerate(pressures):
            if pressure >= threshold:
                preinfusion_time = times[i] if i < len(times) else 0.0
                break

    total_time = shot.duration / 1000.0  # Convert to seconds
    main_extraction_time = max(0.0, total_time - preinfusion_time)

    extraction_summary = ExtractionSummary(
        preinfusion_time_s=round(preinfusion_time * 10) / 10,
        main_extraction_time_s=round(main_extraction_time * 10) / 10,
        total_time_s=round(total_time * 10) / 10,
    )

    return ShotSummary(
        temperature=temp_summary,
        pressure=pressure_summary,
        flow=flow_summary,
        extraction=extraction_summary,
    )


def process_phases(shot: ShotData) -> list[PhaseData]:
    """Process shot phases for AI analysis.

    Backward-compatible wrapper around ``_build_phases``.
    Returns phases with representative samples but no diagnostics.

    Args:
        shot: Parsed shot data

    Returns:
        List of phase data with statistics and representative samples
    """
    return _build_phases(shot, include_samples=True, include_diagnostics=False)


# ═══════════════════════════════════════════════════════════════════
# DIAGNOSTIC FEATURE COMPUTATION
# Physics-informed features with interpretive annotations for LLM
# reasoning. See docs/research/ESPRESSO_SHOT_FEATURE_RESEARCH.md.
# ═══════════════════════════════════════════════════════════════════

# --- Annotation threshold bands ---
# Ascending: value < upper_bound → label
# Pressure volatility uses Coefficient of Variation (CV = std/mean)
# when mean pressure >= _CV_MIN_PRESSURE_BAR, otherwise falls back
# to absolute bands.  CV normalises by operating pressure so that a
# 0.3 bar swing at 2 bar is treated the same as a 1.35 bar swing at
# 9 bar (~15% relative instability in both cases).
_CV_MIN_PRESSURE_BAR: float = 1.0

_PRESSURE_CV_BANDS: list[tuple[float, str]] = [
    (0.02, "VERY_STABLE"),
    (0.05, "STABLE"),
    (0.10, "MODERATE_JITTER"),
    (0.18, "JITTERY"),
    (float('inf'), "VOLATILE"),
]

# Absolute fallback bands — used only when mean pressure < _CV_MIN_PRESSURE_BAR
_PRESSURE_VOLATILITY_BANDS: list[tuple[float, str]] = [
    (0.15, "VERY_STABLE"),
    (0.35, "STABLE"),
    (0.6, "MODERATE_JITTER"),
    (1.0, "JITTERY"),
    (float('inf'), "VOLATILE"),
]

_FLOW_VOLATILITY_BANDS: list[tuple[float, str]] = [
    (0.10, "VERY_STABLE"),
    (0.25, "STABLE"),
    (0.50, "MODERATE_JITTER"),
    (0.80, "JITTERY"),
    (float('inf'), "VOLATILE"),
]

_RESISTANCE_LEVEL_BANDS: list[tuple[float, str]] = [
    (0.5, "VERY_LOW"),
    (1.5, "LOW"),
    (3.0, "MODERATE"),
    (5.0, "HIGH"),
    (float('inf'), "VERY_HIGH"),
]

_RESISTANCE_STABILITY_BANDS: list[tuple[float, str]] = [
    (0.2, "VERY_STABLE"),
    (0.5, "STABLE"),
    (1.0, "MODERATE"),
    (float('inf'), "VOLATILE"),
]

_RESISTANCE_PEAK_TIMING_BANDS: list[tuple[float, str]] = [
    (0.15, "EARLY"),
    (0.35, "GOOD_TIMING"),
    (0.60, "MID_SHOT"),
    (float('inf'), "LATE"),
]

_FLOW_ACCELERATION_BANDS: list[tuple[float, str]] = [
    (0.02, "STABLE"),
    (0.05, "SLIGHT_ACCELERATION"),
    (0.10, "MODERATE_ACCELERATION"),
    (float('inf'), "RAPID_ACCELERATION"),
]

_TEMP_OVERSHOOT_BANDS: list[tuple[float, str]] = [
    (0.5, "MINIMAL"),
    (1.0, "SLIGHT"),
    (2.0, "MODERATE"),
    (float('inf'), "SIGNIFICANT"),
]

_TEMP_STABILITY_BANDS: list[tuple[float, str]] = [
    (0.3, "VERY_STABLE"),
    (0.8, "STABLE"),
    (1.5, "MODERATE"),
    (float('inf'), "UNSTABLE"),
]

# Descending: value >= lower_bound → label
_RESISTANCE_SLOPE_BANDS: list[tuple[float, str]] = [
    (0.05, "INCREASING"),
    (-0.02, "FLAT"),
    (-0.08, "GRADUAL_DECLINE"),
    (-0.15, "MODERATE_DECLINE"),
    (float('-inf'), "STEEP_DECLINE"),
]

_PRESSURE_DROP_RATE_BANDS: list[tuple[float, str]] = [
    (-1.0, "NORMAL"),
    (-2.5, "MODERATE_DROP"),
    (-5.0, "STEEP_DROP"),
    (float('-inf'), "CLIFF"),
]

_PROFILE_ADHERENCE_BANDS: list[tuple[float, str]] = [
    (0.3, "EXCELLENT"),
    (0.8, "GOOD"),
    (1.5, "FAIR"),
    (float('inf'), "POOR"),
]

_PRESSURE_OVERSHOOT_BANDS: list[tuple[float, str]] = [
    (0.25, "WITHIN_TOLERANCE"),
    (0.5, "MINOR_OVERSHOOT"),
    (1.0, "NOTABLE_OVERSHOOT"),
    (float('inf'), "SEVERE_OVERSHOOT"),
]

_FLOW_DEVIATION_BANDS: list[tuple[float, str]] = [
    (0.3, "WITHIN_TOLERANCE"),
    (0.7, "MINOR_DEVIATION"),
    (1.5, "NOTABLE_DEVIATION"),
    (float('inf'), "SEVERE_DEVIATION"),
]

_TAPER_SMOOTHNESS_BANDS: list[tuple[float, str]] = [
    (0.2, "VERY_SMOOTH"),
    (0.5, "SMOOTH"),
    (1.0, "MODERATE"),
    (float('inf'), "ROUGH"),
]

_RAMP_RATE_BANDS: list[tuple[float, str]] = [
    (0.5, "GENTLE"),
    (1.5, "MODERATE"),
    (3.0, "BRISK"),
    (5.0, "AGGRESSIVE"),
    (float('inf'), "VERY_AGGRESSIVE"),
]


# --- Helper functions ---

def _annotate_ascending(value: float, bands: list[tuple[float, str]]) -> str:
    """Classify value using ascending threshold bands (upper_bound, label)."""
    for upper, label in bands:
        if value < upper:
            return label
    return bands[-1][1]


def _annotate_descending(value: float, bands: list[tuple[float, str]]) -> str:
    """Classify value using descending threshold bands (lower_bound, label)."""
    for lower, label in bands:
        if value >= lower:
            return label
    return bands[-1][1]


def _safe_mean(values: list[float]) -> float:
    """Calculate mean, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_std(values: list[float]) -> float:
    """Calculate population standard deviation, returning 0.0 for < 2 values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _pressure_volatility_label(std: float, mean: float) -> str:
    """Classify pressure volatility using CV when possible.

    Uses coefficient of variation (std/mean) when mean pressure is
    above ``_CV_MIN_PRESSURE_BAR``.  Falls back to absolute std
    bands for very-low-pressure profiles where CV would be noisy.
    """
    if mean >= _CV_MIN_PRESSURE_BAR and mean > 0:
        cv = std / mean
        return _annotate_ascending(cv, _PRESSURE_CV_BANDS)
    return _annotate_ascending(std, _PRESSURE_VOLATILITY_BANDS)


# Minimum number of steady-state samples required for a reliable
# channeling assessment.  Fewer than this → INSUFFICIENT_DATA.
_MIN_STEADY_STATE_SAMPLES: int = 5


def _trim_ramp_up(
    pressures: list[float],
    flows: list[float],
    samples: list[dict],
    threshold_pct: float = 0.90,
) -> tuple[list[float], list[float], list[dict]]:
    """Exclude the ramp-up portion from a brew/hold phase.

    Returns the suffix of *pressures*, *flows*, and *samples* starting
    from the first index where pressure reaches ``threshold_pct`` of the
    phase-maximum pressure.  If pressure never reaches the threshold the
    original lists are returned unchanged.
    """
    if not pressures:
        return pressures, flows, samples

    peak = max(pressures)
    if peak <= 0:
        return pressures, flows, samples

    target = peak * threshold_pct
    for i, p in enumerate(pressures):
        if p >= target:
            return pressures[i:], flows[i:], samples[i:]

    return pressures, flows, samples


def _linear_slope(values: list[float], dt: float) -> float:
    """Calculate linear regression slope (units per second).

    Args:
        values: Ordered measurements
        dt: Time interval between consecutive samples in seconds

    Returns:
        Slope in units/second, or 0.0 if insufficient data
    """
    n = len(values)
    if n < 2 or dt <= 0:
        return 0.0
    x = [i * dt for i in range(n)]
    x_mean = sum(x) / n
    y_mean = sum(values) / n
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, values))
    denominator = sum((xi - x_mean) ** 2 for xi in x)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _get_brew_phase_samples(shot: ShotData) -> list[dict]:
    """Extract samples from the brew/extraction phase (excluding pre-infusion).

    Uses phase definitions if available, otherwise falls back to
    detecting when pressure reaches 50% of peak.
    """
    samples = shot.samples
    if not samples:
        return []

    if shot.phases:
        brew_samples: list[dict] = []
        for i, phase in enumerate(shot.phases):
            if _classify_phase_by_name(phase.phase_name) == "preinfusion":
                continue
            start_idx = phase.sample_index
            end_idx = (
                shot.phases[i + 1].sample_index
                if i + 1 < len(shot.phases)
                else len(samples)
            )
            brew_samples.extend(samples[start_idx:end_idx])
        return brew_samples if brew_samples else samples

    # Fallback: skip samples before pressure reaches 50% of peak
    pressures = [s.get('cp', 0.0) for s in samples]
    peak = max(pressures) if pressures else 0.0
    if peak > 0:
        threshold = peak * 0.5
        for i, p in enumerate(pressures):
            if p >= threshold:
                return samples[i:]
    return samples


def _assess_channeling_risk(
    pressure_vol: float,
    flow_vol: float,
    max_drop_rate: float,
    flow_accel: float,
) -> str:
    """Assess overall channeling risk from individual indicators."""
    score = 0
    if pressure_vol >= 0.35:
        score += 1
    if pressure_vol >= 0.6:
        score += 1
    if flow_vol >= 0.25:
        score += 1
    if flow_vol >= 0.5:
        score += 1
    if max_drop_rate <= -1.5:
        score += 1
    if max_drop_rate <= -3.0:
        score += 1
    if flow_accel >= 0.05:
        score += 1
    if flow_accel >= 0.10:
        score += 1
    if score <= 1:
        return "LOW"
    if score <= 3:
        return "MODERATE"
    if score <= 5:
        return "HIGH"
    return "VERY_HIGH"


def _round2(value: float) -> float:
    """Round to 2 decimal places."""
    return round(value * 100) / 100


def _compute_rmse(actual: list[float], target: list[float]) -> float:
    """Compute RMSE between actual and target value sequences."""
    if not actual or len(actual) != len(target):
        return 0.0
    sse = sum((a - t) ** 2 for a, t in zip(actual, target))
    return math.sqrt(sse / len(actual))


# --- Phase classification ---

_PREINFUSION_KEYWORDS = (
    'preinfusion', 'pre-infusion', 'pi', 'soak',
    'bloom', 'fill', 'preinfuse',
)

_DECLINE_KEYWORDS = (
    'decline', 'taper', 'ramp-down', 'ramp down',
    'cool down', 'cooldown',
)


def _classify_phase_by_name(name: str) -> Optional[str]:
    """Try to classify a phase name via keyword matching.

    Returns 'preinfusion', 'brew', 'decline', or None if unrecognised.
    Uses substring matching so creative names like 'Gentle Pre-infusion'
    or 'Blooming Phase' still match.
    """
    normalized = name.lower().strip()
    for kw in _PREINFUSION_KEYWORDS:
        if kw in normalized:
            return "preinfusion"
    for kw in _DECLINE_KEYWORDS:
        if kw in normalized:
            return "decline"
    return None


def _classify_phase_by_telemetry(
    phase_samples: list[dict], phase_index: int, total_phases: int,
) -> str:
    """Fallback: classify a phase from its telemetry shape.

    Heuristics:
    - First phase with low avg pressure and rising trend → preinfusion
    - Last phase with declining pressure trend → decline
    - Everything else → brew
    """
    pressures = [s.get('cp', 0.0) for s in phase_samples]
    if len(pressures) < 2:
        return "brew"

    avg_p = sum(pressures) / len(pressures)
    slope = (pressures[-1] - pressures[0]) / max(len(pressures) - 1, 1)

    # First phase, low pressure, rising → preinfusion
    if phase_index == 0 and avg_p < 5.0 and slope >= 0:
        return "preinfusion"

    # Last phase (not first), declining pressure → decline
    if phase_index == total_phases - 1 and phase_index > 0 and slope < -0.3:
        return "decline"

    return "brew"


def _classify_phase(
    name: str,
    phase_samples: Optional[list[dict]] = None,
    phase_index: int = 0,
    total_phases: int = 1,
) -> str:
    """Classify a phase into preinfusion / brew / decline.

    Tries keyword matching on the phase name first.  Falls back to
    telemetry-based heuristics when the name is not recognised.
    """
    result = _classify_phase_by_name(name)
    if result is not None:
        return result
    if phase_samples is not None:
        return _classify_phase_by_telemetry(
            phase_samples, phase_index, total_phases,
        )
    return "brew"


def compute_shot_diagnostics(shot: ShotData) -> Optional[ShotDiagnostics]:
    """Compute diagnostic features from shot telemetry data.

    Pre-computes physics-informed features with interpretive annotations
    for optimal LLM reasoning. Features are grouped by diagnostic purpose:
    puck resistance, channeling indicators, temperature stability,
    extraction quality, and weight/yield analysis.

    Returns None if insufficient data for meaningful analysis (< 5 total
    samples or < 3 brew-phase samples).
    """
    samples = shot.samples
    if not samples or len(samples) < 5:
        return None

    dt = shot.sample_interval / 1000.0  # seconds per sample

    brew_samples = _get_brew_phase_samples(shot)
    if len(brew_samples) < 3:
        return None

    # Extract brew-phase arrays
    brew_pressures = [s.get('cp', 0.0) for s in brew_samples]
    brew_flows = [s.get('pf', 0.0) for s in brew_samples]
    brew_temps = [s.get('ct', 0.0) for s in brew_samples]
    brew_target_temps = [s.get('tt', 0.0) for s in brew_samples]

    # Full-shot pressure for AUC
    all_pressures = [s.get('cp', 0.0) for s in samples]

    # Weight data (may not be available if no Bluetooth scale)
    brew_weights = [s.get('v', 0.0) for s in brew_samples]
    has_scale = any(w > 0 for w in brew_weights)

    # ── PUCK RESISTANCE (the master diagnostic) ──────────────
    # R = P / F² (quadratic Darcy model)
    resistance_values: list[float] = []
    for p, f in zip(brew_pressures, brew_flows):
        if f > 0.1:  # Skip near-zero flow to avoid division artifacts
            resistance_values.append(p / (f * f))

    # Stalled shot detection: when all flows are ≤ 0.1 ml/s, the puck is
    # choked and resistance is effectively infinite — not zero.
    stalled_shot = len(resistance_values) == 0 and len(brew_flows) > 0

    r_avg = _round2(_safe_mean(resistance_values))
    r_std = _round2(_safe_std(resistance_values))
    r_slope = _round2(_linear_slope(resistance_values, dt))
    if resistance_values:
        r_peak_val = max(resistance_values)
        r_peak = _round2(r_peak_val)
        r_peak_idx = resistance_values.index(r_peak_val)
    else:
        r_peak = 0.0
        r_peak_idx = 0
    r_peak_timing = (
        _round2(r_peak_idx / len(resistance_values))
        if resistance_values else 0.0
    )

    if stalled_shot:
        resistance_annotations: dict[str, str] = {
            "level": "STALLED",
            "stability": "NO_DATA",
            "erosion": "NO_DATA",
            "saturation": "NO_DATA",
            "stalled_shot": "true",
            "note": "All flow readings ≤ 0.1 ml/s — puck is choked. "
                    "Resistance is effectively infinite, not zero. "
                    "Likely causes: grind far too fine, excessive dose, "
                    "or severe puck compression.",
        }
    else:
        resistance_annotations = {
            "level": _annotate_ascending(r_avg, _RESISTANCE_LEVEL_BANDS),
            "stability": _annotate_ascending(r_std, _RESISTANCE_STABILITY_BANDS),
            "erosion": _annotate_descending(r_slope, _RESISTANCE_SLOPE_BANDS),
            "saturation": _annotate_ascending(
                r_peak_timing, _RESISTANCE_PEAK_TIMING_BANDS
            ),
        }

    resistance = ResistanceDiagnostics(
        avg=r_avg,
        std=r_std,
        slope=r_slope,
        peak=r_peak,
        peak_timing_pct=r_peak_timing,
        annotations=resistance_annotations,
    )

    # ── CHANNELING INDICATORS ────────────────────────────────
    # Trim ramp-up to compute volatility only on steady-state data
    ss_pressures, ss_flows, _ = _trim_ramp_up(
        brew_pressures, brew_flows, brew_samples,
    )

    if len(ss_pressures) >= _MIN_STEADY_STATE_SAMPLES:
        p_volatility = _round2(_safe_std(ss_pressures))
        f_volatility = _round2(_safe_std(ss_flows))
        p_mean = _safe_mean(ss_pressures)

        # Max pressure drop rate (most negative dP/dt)
        p_derivatives: list[float] = []
        for i in range(1, len(ss_pressures)):
            dp_dt = (ss_pressures[i] - ss_pressures[i - 1]) / dt
            p_derivatives.append(dp_dt)
        p_max_drop = _round2(min(p_derivatives)) if p_derivatives else 0.0

        # Flow acceleration in late shot (last 40% of steady-state)
        late_start = int(len(ss_flows) * 0.6)
        late_flows = ss_flows[late_start:]
        f_accel_late = _round2(_linear_slope(late_flows, dt))

        overall_risk = _assess_channeling_risk(
            p_volatility, f_volatility, p_max_drop, f_accel_late
        )

        ramp_excluded = len(brew_pressures) - len(ss_pressures)
        channeling_annotations: dict[str, str] = {
            "pressure_stability": _pressure_volatility_label(
                p_volatility, p_mean,
            ),
            "flow_stability": _annotate_ascending(
                f_volatility, _FLOW_VOLATILITY_BANDS
            ),
            "pressure_drop": _annotate_descending(
                p_max_drop, _PRESSURE_DROP_RATE_BANDS
            ),
            "late_flow_trend": _annotate_ascending(
                f_accel_late, _FLOW_ACCELERATION_BANDS
            ),
        }
        if ramp_excluded > 0:
            channeling_annotations["note"] = (
                f"{ramp_excluded} ramp-up samples excluded from stability calculation"
            )
    else:
        # Insufficient steady-state data — compute from raw brew data
        p_volatility = _round2(_safe_std(brew_pressures))
        f_volatility = _round2(_safe_std(brew_flows))
        p_mean = _safe_mean(brew_pressures)
        p_max_drop = 0.0
        f_accel_late = 0.0
        overall_risk = "INSUFFICIENT_DATA"
        channeling_annotations = {
            "pressure_stability": _pressure_volatility_label(
                p_volatility, p_mean,
            ),
            "flow_stability": _annotate_ascending(
                f_volatility, _FLOW_VOLATILITY_BANDS
            ),
            "pressure_drop": _annotate_descending(
                p_max_drop, _PRESSURE_DROP_RATE_BANDS
            ),
            "late_flow_trend": _annotate_ascending(
                f_accel_late, _FLOW_ACCELERATION_BANDS
            ),
            "note": (
                f"Only {len(ss_pressures)} samples at ≥90% of peak pressure "
                f"(need {_MIN_STEADY_STATE_SAMPLES}). "
                "Channeling assessment unreliable for this phase length."
            ),
        }

    channeling = ChannelingIndicators(
        pressure_volatility_bar=p_volatility,
        flow_volatility_ml_s=f_volatility,
        pressure_max_drop_rate_bar_s=p_max_drop,
        flow_acceleration_late_ml_s2=f_accel_late,
        overall_risk=overall_risk,
        annotations=channeling_annotations,
    )

    # ── TEMPERATURE DIAGNOSTICS ──────────────────────────────
    # Filter to samples where target temp is recorded (tt > 0) so that
    # stability, overshoot and undershoot are computed on the same
    # population — avoids incoherent comparisons for LLM consumers.
    brew_temps_with_target = [
        ct for ct, tt in zip(brew_temps, brew_target_temps) if tt > 0
    ]
    temp_deviations = [
        ct - tt
        for ct, tt in zip(brew_temps, brew_target_temps)
        if tt > 0  # Skip if no target temp recorded
    ]
    t_overshoot = max(temp_deviations) if temp_deviations else 0.0
    t_undershoot = abs(min(temp_deviations)) if temp_deviations else 0.0
    t_std = _round2(_safe_std(brew_temps_with_target))

    temperature = TemperatureDiagnostics(
        overshoot_c=_round2(max(0.0, t_overshoot)),
        undershoot_c=_round2(max(0.0, t_undershoot)),
        stability_std_c=t_std,
        annotations={
            "overshoot": _annotate_ascending(
                max(0.0, t_overshoot), _TEMP_OVERSHOOT_BANDS
            ),
            "undershoot": _annotate_ascending(
                max(0.0, t_undershoot), _TEMP_OVERSHOOT_BANDS
            ),
            "stability": _annotate_ascending(t_std, _TEMP_STABILITY_BANDS),
        },
    )

    # ── EXTRACTION METRICS ───────────────────────────────────
    # Pressure AUC (trapezoidal approximation over full shot)
    p_auc = _round2(sum(p * dt for p in all_pressures))
    p_slope = _round2(_linear_slope(brew_pressures, dt))
    f_slope = _round2(_linear_slope(brew_flows, dt))
    f_avg = _round2(_safe_mean(brew_flows))

    extraction = ExtractionMetrics(
        pressure_auc_bar_s=p_auc,
        pressure_slope_brew_bar_s=p_slope,
        flow_slope_brew_ml_s2=f_slope,
        flow_avg_brew_ml_s=f_avg,
        annotations={
            "pressure_trend": _annotate_descending(
                p_slope, _RESISTANCE_SLOPE_BANDS
            ),
            "flow_trend": (
                "DECLINING" if f_slope < -0.02
                else "STABLE" if f_slope < 0.02
                else "INCREASING"
            ),
        },
    )

    # ── WEIGHT / YIELD DIAGNOSTICS ───────────────────────────
    w_rate_avg: Optional[float] = None
    w_rate_std: Optional[float] = None
    weight_annotations: dict[str, str] = {}

    if has_scale:
        weight_rates: list[float] = []
        for i in range(1, len(brew_weights)):
            rate = (brew_weights[i] - brew_weights[i - 1]) / dt
            if rate >= 0:  # Only accumulating weight
                weight_rates.append(rate)
        if weight_rates:
            w_rate_avg = _round2(_safe_mean(weight_rates))
            w_rate_std = _round2(_safe_std(weight_rates))
            weight_annotations["rate_stability"] = _annotate_ascending(
                w_rate_std, _FLOW_VOLATILITY_BANDS
            )
    else:
        weight_annotations["note"] = "No scale data available"

    weight = WeightDiagnostics(
        rate_avg_g_s=w_rate_avg,
        rate_std_g_s=w_rate_std,
        scale_connected=has_scale,
        annotations=weight_annotations,
    )

    # ── PROFILE COMPLIANCE ────────────────────────────────────
    profile_compliance = _compute_profile_compliance(samples, dt)

    return ShotDiagnostics(
        resistance=resistance,
        channeling=channeling,
        temperature=temperature,
        extraction=extraction,
        weight=weight,
        profile_compliance=profile_compliance,
    )


def _compute_profile_compliance(
    samples: list[dict], dt: float,
) -> Optional[ProfileComplianceMetrics]:
    """Compute how well the shot follows the target profile.

    Compares actual pressure/flow against target pressure/flow
    (tp/tf fields) using RMSE and max overshoot/undershoot.
    pressure_overshoot > 1.0 bar is highly unusual and almost certainly
    indicates grind too fine, excessive dose, or a puck preparation issue.
    """
    # Only include samples where target pressure was recorded
    p_pairs = [(s.get('cp', 0.0), s['tp']) for s in samples if 'tp' in s]
    if len(p_pairs) < 3:
        return None

    p_actual = [a for a, _ in p_pairs]
    p_target = [t for _, t in p_pairs]
    p_rmse = _round2(_compute_rmse(p_actual, p_target))

    deviations = [a - t for a, t in p_pairs]
    max_overshoot = _round2(max(0.0, max(deviations)))
    max_undershoot = _round2(max(0.0, abs(min(deviations))))

    # Flow compliance (optional — tf may not always be set)
    f_pairs = [(s.get('pf', 0.0), s['tf']) for s in samples if 'tf' in s]
    f_rmse: Optional[float] = None
    max_flow_overshoot: Optional[float] = None
    max_flow_undershoot: Optional[float] = None
    if len(f_pairs) >= 3:
        f_rmse = _round2(_compute_rmse(
            [a for a, _ in f_pairs], [t for _, t in f_pairs],
        ))
        f_deviations = [a - t for a, t in f_pairs]
        max_flow_overshoot = _round2(max(0.0, max(f_deviations)))
        max_flow_undershoot = _round2(max(0.0, abs(min(f_deviations))))

    annotations: dict[str, str] = {
        "pressure_adherence": _annotate_ascending(
            p_rmse, _PROFILE_ADHERENCE_BANDS
        ),
        "pressure_overshoot": _annotate_ascending(
            max_overshoot, _PRESSURE_OVERSHOOT_BANDS
        ),
    }
    if f_rmse is not None:
        annotations["flow_adherence"] = _annotate_ascending(
            f_rmse, _PROFILE_ADHERENCE_BANDS
        )
    if max_flow_overshoot is not None:
        annotations["flow_overshoot"] = _annotate_ascending(
            max_flow_overshoot, _FLOW_DEVIATION_BANDS
        )
    if max_flow_undershoot is not None:
        annotations["flow_undershoot"] = _annotate_ascending(
            max_flow_undershoot, _FLOW_DEVIATION_BANDS
        )

    return ProfileComplianceMetrics(
        pressure_rmse_bar=p_rmse,
        flow_rmse_ml_s=f_rmse,
        max_pressure_overshoot_bar=max_overshoot,
        max_pressure_undershoot_bar=max_undershoot,
        max_flow_overshoot_ml_s=max_flow_overshoot,
        max_flow_undershoot_ml_s=max_flow_undershoot,
        annotations=annotations,
    )


def _compute_phase_diagnostics(
    phase_samples: list[dict], phase_type: str, dt: float,
) -> PhaseDiagnostics:
    """Compute diagnostics specific to a phase type.

    Metrics computed depend on phase_type:
    - preinfusion: ramp rate, saturation time
    - brew: resistance, channeling, stability
    - decline: taper rate, taper smoothness
    All phases get RMSE vs target and basic stats.
    """
    pressures = [s.get('cp', 0.0) for s in phase_samples]
    flows = [s.get('pf', 0.0) for s in phase_samples]

    avg_p = _round2(_safe_mean(pressures))
    avg_f = _round2(_safe_mean(flows))

    # Per-phase RMSE vs target
    p_pairs = [(s.get('cp', 0.0), s['tp']) for s in phase_samples if 'tp' in s]
    p_rmse = _round2(_compute_rmse(
        [a for a, _ in p_pairs], [t for _, t in p_pairs],
    )) if p_pairs else 0.0

    f_pairs = [(s.get('pf', 0.0), s['tf']) for s in phase_samples if 'tf' in s]
    f_rmse = _round2(_compute_rmse(
        [a for a, _ in f_pairs], [t for _, t in f_pairs],
    )) if f_pairs else 0.0

    annotations: dict[str, str] = {
        "pressure_adherence": _annotate_ascending(
            p_rmse, _PROFILE_ADHERENCE_BANDS
        ),
    }

    result: PhaseDiagnostics = {
        "phase_type": phase_type,
        "avg_pressure_bar": avg_p,
        "avg_flow_ml_s": avg_f,
        "pressure_rmse_bar": p_rmse,
        "flow_rmse_ml_s": f_rmse,
        "annotations": annotations,
    }

    if phase_type == "preinfusion":
        ramp_rate = _round2(_linear_slope(pressures, dt))
        # Saturation: time for flow to stabilise
        sat_time = _round2(len(phase_samples) * dt)
        for i in range(2, len(flows)):
            window = flows[max(0, i - 2):i + 1]
            if len(window) >= 2 and _safe_std(window) < 0.15 and _safe_mean(window) > 0.1:
                sat_time = _round2(i * dt)
                break
        result["ramp_rate_bar_s"] = ramp_rate
        result["saturation_time_s"] = sat_time
        annotations["ramp_rate"] = _annotate_ascending(
            abs(ramp_rate), _RAMP_RATE_BANDS
        )

    elif phase_type == "brew":
        r_values = [
            p / (f * f) for p, f in zip(pressures, flows) if f > 0.1
        ]
        r_avg = _round2(_safe_mean(r_values))
        r_slope = _round2(_linear_slope(r_values, dt))

        # Trim ramp-up for stability metrics
        ss_p, ss_f, _ = _trim_ramp_up(pressures, flows, phase_samples)
        ss_count = len(ss_p)

        if ss_count >= _MIN_STEADY_STATE_SAMPLES:
            p_vol = _round2(_safe_std(ss_p))
            f_vol = _round2(_safe_std(ss_f))
            p_mean = _safe_mean(ss_p)
            p_derivs = [
                (ss_p[i] - ss_p[i - 1]) / dt
                for i in range(1, len(ss_p))
            ]
            p_max_drop = min(p_derivs) if p_derivs else 0.0
            late_start = int(len(ss_f) * 0.6)
            f_accel = _linear_slope(ss_f[late_start:], dt)
            risk = _assess_channeling_risk(p_vol, f_vol, p_max_drop, f_accel)
        else:
            p_vol = _round2(_safe_std(pressures))
            f_vol = _round2(_safe_std(flows))
            p_mean = _safe_mean(pressures)
            risk = "INSUFFICIENT_DATA"

        result["resistance_avg"] = r_avg
        result["resistance_slope"] = r_slope
        result["channeling_risk"] = risk
        result["pressure_stability_bar"] = p_vol
        result["flow_stability_ml_s"] = f_vol
        annotations["resistance_level"] = _annotate_ascending(
            r_avg, _RESISTANCE_LEVEL_BANDS
        )
        annotations["resistance_erosion"] = _annotate_descending(
            r_slope, _RESISTANCE_SLOPE_BANDS
        )
        annotations["channeling"] = risk
        annotations["pressure_stability"] = _pressure_volatility_label(
            p_vol, p_mean,
        )
        annotations["flow_stability"] = _annotate_ascending(
            f_vol, _FLOW_VOLATILITY_BANDS
        )
        if ss_count < _MIN_STEADY_STATE_SAMPLES:
            annotations["channeling_note"] = (
                f"Only {ss_count} samples at >=90% of peak pressure "
                f"(need {_MIN_STEADY_STATE_SAMPLES}). "
                "Assessment unreliable for this phase length."
            )

    elif phase_type == "decline":
        taper_rate = _round2(_linear_slope(pressures, dt))
        p_derivs = [
            (pressures[i] - pressures[i - 1]) / dt
            for i in range(1, len(pressures))
        ]
        taper_smooth = _round2(_safe_std(p_derivs)) if p_derivs else 0.0
        result["taper_rate_bar_s"] = taper_rate
        result["taper_smoothness"] = taper_smooth
        annotations["taper_smoothness"] = _annotate_ascending(
            taper_smooth, _TAPER_SMOOTHNESS_BANDS
        )

    return result


def compute_summary_diagnostics(shot: ShotData) -> Optional[SummaryDiagnostics]:
    """Compute lightweight diagnostics for the summary detail level.

    Returns only key indicators an LLM needs for a quick assessment.
    """
    samples = shot.samples
    if not samples or len(samples) < 5:
        return None

    dt = shot.sample_interval / 1000.0
    brew_samples = _get_brew_phase_samples(shot)
    if len(brew_samples) < 3:
        return None

    brew_pressures = [s.get('cp', 0.0) for s in brew_samples]
    brew_flows = [s.get('pf', 0.0) for s in brew_samples]
    brew_temps = [s.get('ct', 0.0) for s in brew_samples]
    brew_weights = [s.get('v', 0.0) for s in brew_samples]

    # Resistance
    r_values = [p / (f * f) for p, f in zip(brew_pressures, brew_flows) if f > 0.1]
    stalled_shot = len(r_values) == 0 and len(brew_flows) > 0
    r_avg = _round2(_safe_mean(r_values))
    r_slope = _round2(_linear_slope(r_values, dt))

    # Channeling — trim ramp-up to assess steady-state only
    ss_pressures, ss_flows, _ = _trim_ramp_up(
        brew_pressures, brew_flows, brew_samples,
    )
    if len(ss_pressures) >= _MIN_STEADY_STATE_SAMPLES:
        p_vol = _safe_std(ss_pressures)
        f_vol = _safe_std(ss_flows)
        p_derivs = [
            (ss_pressures[i] - ss_pressures[i - 1]) / dt
            for i in range(1, len(ss_pressures))
        ]
        p_max_drop = min(p_derivs) if p_derivs else 0.0
        late_start = int(len(ss_flows) * 0.6)
        f_accel = _linear_slope(ss_flows[late_start:], dt)
        risk = _assess_channeling_risk(p_vol, f_vol, p_max_drop, f_accel)
    else:
        p_vol = _safe_std(brew_pressures)
        f_vol = _safe_std(brew_flows)
        risk = "INSUFFICIENT_DATA"

    # Temperature — filter to samples with target temp (tt > 0) for
    # consistency with overshoot/undershoot in detailed diagnostics.
    brew_target_temps = [s.get('tt', 0.0) for s in brew_samples]
    brew_temps_with_target = [
        ct for ct, tt in zip(brew_temps, brew_target_temps) if tt > 0
    ]
    t_std = _round2(_safe_std(brew_temps_with_target if brew_temps_with_target else brew_temps))

    # Profile compliance
    p_rmse = 0.0
    max_overshoot = 0.0
    p_targets = [(s.get('cp', 0.0), s['tp']) for s in brew_samples if 'tp' in s]
    if p_targets:
        p_rmse = _round2(_compute_rmse(
            [a for a, _ in p_targets], [t for _, t in p_targets],
        ))
        devs = [a - t for a, t in p_targets]
        max_overshoot = _round2(max(0.0, max(devs)))

    # Flow compliance (optional — tf may not always be set)
    f_rmse: Optional[float] = None
    max_flow_overshoot: Optional[float] = None
    f_targets = [(s.get('pf', 0.0), s['tf']) for s in brew_samples if 'tf' in s]
    if len(f_targets) >= 3:
        f_rmse = _round2(_compute_rmse(
            [a for a, _ in f_targets], [t for _, t in f_targets],
        ))
        f_devs = [a - t for a, t in f_targets]
        max_flow_overshoot = _round2(max(0.0, max(f_devs)))

    # Scale
    has_scale = any(w > 0 for w in brew_weights)

    annotations: dict[str, str] = {
        "resistance_level": "STALLED" if stalled_shot else _annotate_ascending(r_avg, _RESISTANCE_LEVEL_BANDS),
        "resistance_erosion": "NO_DATA" if stalled_shot else _annotate_descending(r_slope, _RESISTANCE_SLOPE_BANDS),
        "channeling_risk": risk,
        "temperature_stability": _annotate_ascending(t_std, _TEMP_STABILITY_BANDS),
        "pressure_adherence": _annotate_ascending(p_rmse, _PROFILE_ADHERENCE_BANDS),
        "pressure_overshoot": _annotate_ascending(
            max_overshoot, _PRESSURE_OVERSHOOT_BANDS
        ),
    }
    if f_rmse is not None:
        annotations["flow_adherence"] = _annotate_ascending(
            f_rmse, _PROFILE_ADHERENCE_BANDS
        )
    if max_flow_overshoot is not None:
        annotations["flow_overshoot"] = _annotate_ascending(
            max_flow_overshoot, _FLOW_DEVIATION_BANDS
        )
    if stalled_shot:
        annotations["stalled_shot"] = "true"

    return SummaryDiagnostics(
        resistance_avg=r_avg,
        resistance_slope=r_slope,
        channeling_risk=risk,
        temperature_stability_c=t_std,
        pressure_rmse_bar=p_rmse,
        max_overshoot_bar=max_overshoot,
        flow_rmse_ml_s=f_rmse,
        max_flow_overshoot_ml_s=max_flow_overshoot,
        scale_connected=has_scale,
        annotations=annotations,
    )


# ═══════════════════════════════════════════════════════════════════
# DETAIL LEVELS
# ═══════════════════════════════════════════════════════════════════

VALID_DETAIL_LEVELS = ("summary", "per_phase", "per_phase_detailed")


def transform_shot_for_ai(
    shot: ShotData, detail: str = "summary",
) -> TransformedShot:
    """Transform shot data for AI analysis.

    Converts raw binary shot data into a structured format optimised
    for AI analysis. The *detail* parameter controls output granularity:

    - **summary** (default): Key indicators only — resistance avg/slope,
      channeling risk, temperature stability, profile compliance RMSE
      and max overshoot. Phases listed with stats but no samples.
      Minimal token usage.
    - **per_phase**: Full shot diagnostics *plus* per-phase diagnostics
      (preinfusion/brew/decline specific metrics). No samples yet —
      use this to identify *which* phase has an issue.
    - **per_phase_detailed**: Everything in per_phase *plus* ~5
      evenly-spaced averaged samples per phase to reveal curve shape.
      Use after per_phase to see what the pressure/flow curve looks like.

    Args:
        shot: Parsed shot data
        detail: Granularity level ("summary", "per_phase", "per_phase_detailed")

    Returns:
        Transformed shot data with diagnostics appropriate to detail level
    """
    if detail not in VALID_DETAIL_LEVELS:
        detail = "summary"

    summary_stats = calculate_summary(shot)
    dt = shot.sample_interval / 1000.0

    if detail == "summary":
        phases = _build_phases(shot, include_samples=False, include_diagnostics=False, dt=dt)
        diagnostics: Optional[ShotDiagnostics | SummaryDiagnostics] = compute_summary_diagnostics(shot)
    elif detail == "per_phase":
        phases = _build_phases(shot, include_samples=False, include_diagnostics=True, dt=dt)
        diagnostics = compute_shot_diagnostics(shot)
    else:  # per_phase_detailed
        phases = _build_phases(shot, include_samples=True, include_diagnostics=True, dt=dt)
        diagnostics = compute_shot_diagnostics(shot)

    return TransformedShot(
        shot_id=shot.id,
        profile_name=shot.profile_name,
        profile_id=shot.profile_id,
        timestamp=shot.timestamp,
        duration_seconds=round(shot.duration / 1000.0 * 10) / 10,
        final_weight_g=shot.weight,
        summary=summary_stats,
        phases=phases,
        diagnostics=diagnostics,
        detail_level=detail,
    )


def _build_phases(
    shot: ShotData,
    *,
    include_samples: bool = True,
    include_diagnostics: bool = False,
    dt: float = 0.0,
) -> list[PhaseData]:
    """Build phase list with configurable content.

    This is the internal implementation behind ``process_phases`` and
    the detail-level logic in ``transform_shot_for_ai``.
    """
    phases: list[PhaseData] = []
    samples = shot.samples
    if not samples:
        return phases

    if shot.phases:
        for i, phase in enumerate(shot.phases):
            start_index = phase.sample_index
            end_index = (
                shot.phases[i + 1].sample_index
                if i + 1 < len(shot.phases)
                else len(samples)
            )
            phase_samples = samples[start_index:end_index]
            if not phase_samples:
                continue

            temperatures = [s['ct'] for s in phase_samples if 'ct' in s]
            pressures = [s['cp'] for s in phase_samples if 'cp' in s]
            avg_temp = round(sum(temperatures) / len(temperatures) * 10) / 10 if temperatures else 0.0
            avg_pressure = round(sum(pressures) / len(pressures) * 10) / 10 if pressures else 0.0
            total_flow = calculate_total_volume(phase_samples, shot.sample_interval)

            start_time = phase_samples[0].get('t', 0.0) / 1000.0
            end_time = phase_samples[-1].get('t', 0.0) / 1000.0
            duration = max(0.0, end_time - start_time + shot.sample_interval / 1000.0)

            pd: PhaseData = {
                "name": phase.phase_name,
                "phase_number": phase.phase_number,
                "start_time_seconds": round(start_time * 10) / 10,
                "duration_seconds": round(duration * 10) / 10,
                "sample_count": len(phase_samples),
                "avg_temperature_c": avg_temp,
                "avg_pressure_bar": avg_pressure,
                "total_flow_ml": total_flow,
            }

            if include_samples:
                pd["samples"] = _select_samples(phase_samples)

            if include_diagnostics and dt > 0 and len(phase_samples) >= 3:
                phase_type = _classify_phase(
                    phase.phase_name,
                    phase_samples=phase_samples,
                    phase_index=i,
                    total_phases=len(shot.phases),
                )
                pd["diagnostics"] = _compute_phase_diagnostics(
                    phase_samples, phase_type, dt,
                )

            phases.append(pd)
    else:
        # No phases defined — single 'extraction' phase
        temperatures = [s.get('ct', 0.0) for s in samples]
        pressures = [s.get('cp', 0.0) for s in samples]
        avg_temp = round(sum(temperatures) / len(temperatures) * 10) / 10 if temperatures else 0.0
        avg_pressure = round(sum(pressures) / len(pressures) * 10) / 10 if pressures else 0.0
        total_flow = calculate_total_volume(samples, shot.sample_interval)

        pd = {
            "name": "extraction",
            "phase_number": 0,
            "start_time_seconds": 0.0,
            "duration_seconds": round(shot.duration / 1000.0 * 10) / 10,
            "sample_count": len(samples),
            "avg_temperature_c": avg_temp,
            "avg_pressure_bar": avg_pressure,
            "total_flow_ml": total_flow,
        }

        if include_samples:
            pd["samples"] = _select_samples(samples)

        if include_diagnostics and dt > 0 and len(samples) >= 3:
            pd["diagnostics"] = _compute_phase_diagnostics(samples, "brew", dt)

        phases.append(pd)  # type: ignore[arg-type]

    return phases


def _select_samples(samples: list[dict]) -> list[TransformedSample]:
    """Select ~5 representative samples from a phase.

    Returns ~5 evenly-spaced data points with local averaging (±1
    neighbour) to smooth single-sample noise while preserving the
    curve shape. This gives the consuming agent enough resolution to
    see trends without full time-series token cost.
    """
    n = len(samples)
    if n <= 5:
        indices = range(n)
    else:
        # 5 evenly-spaced anchor indices
        indices = sorted(set(
            round(i * (n - 1) / 4) for i in range(5)
        ))

    result: list[TransformedSample] = []
    for idx in indices:
        if idx >= len(samples):
            continue

        # Average over a ±1 window around the anchor
        lo = max(0, idx - 1)
        hi = min(len(samples), idx + 2)  # exclusive
        window = samples[lo:hi]
        wn = len(window)
        result.append(TransformedSample(
            time_seconds=round((samples[idx].get('t', 0.0) / 1000.0) * 10) / 10,
            temperature_c=round(sum(s.get('ct', 0.0) for s in window) / wn * 10) / 10,
            pressure_bar=round(sum(s.get('cp', 0.0) for s in window) / wn * 10) / 10,
            flow_ml_s=round(sum(s.get('pf', 0.0) for s in window) / wn * 10) / 10,
            weight_g=round(sum(s.get('v', 0.0) for s in window) / wn * 10) / 10,
        ))
    return result
