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
          flow_jitter_ml_s, pressure_jitter_bar
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
    # Brew — channeling fields match top-level ChannelingIndicators naming
    resistance_avg: float
    resistance_slope: float
    channeling_risk: str
    flow_jitter_ml_s: float
    pressure_jitter_bar: float
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
    """Puck-stability signals from brew-phase telemetry.

    Four independent indicators, each catching a different channeling
    signature.  The agent should reason about them together, not in
    isolation — a single flag is usually noise; two or more aligned
    flags are a real signal.  Descriptor fields (``*_spread``, shape
    annotations) provide context for interpreting the indicators but
    do not contribute to the risk score.
    """
    # --- Indicators (contribute to channeling_risk) ---
    flow_jitter_ml_s: float
    """Sample-to-sample flow instability after removing intended trajectory.

    Std of first-differences of puck flow.  Insensitive to profile-designed
    ramps — only measures deviation from whatever trend the profile
    commanded.  Catches micro-spikes, oscillation, puck collapsing/
    re-forming.  Misses slow drift that stays within profile shape.
    Unreliable when the steady-state window has fewer than ~8 samples.
    """

    flow_vs_target_residual_ml_s: Optional[float]
    """How far actual flow strayed from the commanded target_flow.

    Std of (actual - target).  ``None`` when no target flow is commanded
    (pure pressure-led profiles).  Catches systematic inability to hold
    the commanded flow curve — often the clearest channeling fingerprint
    on flow-led profiles.
    """

    pressure_max_drop_rate_bar_s: float
    """Steepest instantaneous pressure collapse (bar/second).

    Most-negative single-sample dP/dt.  NOT average; worst-single-sample.
    Catches abrupt channel opening (pressure cliff).  Complementary to
    jitter — a single cliff can register as low jitter but high max_drop.
    Unreliable when the window includes the volumetric-cutoff tail;
    check ``annotations.note`` for mention of trailing zero-flow trimming
    in that case.
    """

    flow_acceleration_late_ml_s2: float
    """Detrended late-window flow slope over the last 40% of steady state.

    Computed as ``late_slope - overall_slope`` rather than the raw late
    slope alone.  Positive = the late window is accelerating relative to
    the overall steady-state trend; negative = it is tapering relative to
    that trend.  Catches late runaway channeling, but can miss channeling
    that develops early and then stabilises.
    """

    # --- Descriptors (context, not scored) ---
    flow_spread_ml_s: float
    """Population std of raw flow values.  NOT a stability metric — includes
    intended profile ramps.  Pair with ``annotations.flow_shape`` to
    distinguish intentional ramps from unintended spread.
    """

    pressure_jitter_bar: float
    """Sample-to-sample pressure instability.  Rarely the primary signal
    but a sanity check — genuine channeling often shows on both variables.
    Used as the secondary-indicator fallback when no target_flow is set.
    """

    # --- Overall assessment ---
    channeling_risk: str
    """LOW | MODERATE | HIGH | VERY_HIGH | INSUFFICIENT_DATA.

    See ``annotations.primary_signal`` to know which indicator(s) drove
    the rating.
    """

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

# Jitter bands — applied to first-difference std (V5).  Tighter numbers than
# raw std because jitter isolates sample-to-sample noise and discards the
# designed-profile trend.  Calibrated against 26 real Amigo Alturas shots
# that measured 0.013-0.020 ml/s jitter across the board (rock-solid by
# design).  JITTERY is where genuine channeling should register.
_FLOW_JITTER_BANDS: list[tuple[float, str]] = [
    (0.025, "VERY_STABLE"),
    (0.050, "STABLE"),
    (0.100, "MODERATE_JITTER"),
    (0.200, "JITTERY"),
    (float('inf'), "VOLATILE"),
]

_PRESSURE_JITTER_BANDS: list[tuple[float, str]] = [
    (0.05, "VERY_STABLE"),
    (0.10, "STABLE"),
    (0.20, "MODERATE_JITTER"),
    (0.40, "JITTERY"),
    (float('inf'), "VOLATILE"),
]

# How far actual flow strayed from target flow (V6).  Reused magnitude
# bands from FLOW_DEVIATION but renamed because the contextual meaning
# differs — this is tracking error over the steady-state window, not
# per-phase RMSE.
_FLOW_VS_TARGET_BANDS: list[tuple[float, str]] = [
    (0.15, "WITHIN_TOLERANCE"),
    (0.35, "MINOR_DEVIATION"),
    (0.70, "NOTABLE_DEVIATION"),
    (float('inf'), "SEVERE_DEVIATION"),
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


def _jitter_std(values: list[float]) -> float:
    """Population std of first-differences — noise around whatever trend exists.

    A flat signal has zero jitter; a smooth linear ramp has near-zero jitter
    (all differences equal); an oscillating or spiky signal has high jitter.
    Insensitive to intentional profile trajectories.
    """
    if len(values) < 3:
        return 0.0
    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
    return _safe_std(diffs)


def _late_flow_runaway(flows: list[float], dt: float) -> float:
    """Excess flow acceleration in the last 40% of the window (ml/s²).

    Computed as ``late_slope - overall_slope``.  Returns 0 when the late
    window matches the overall trend — so a linear flow ramp (profile by
    design) scores 0, while a flat-then-kicks-up trace scores positive.

    This is the channeling-specific reading of flow acceleration: only
    *excess* acceleration beyond the designed trajectory is a runaway
    signal.  Absolute magnitude alone would false-positive on every
    flow-ramp profile.
    """
    n = len(flows)
    if n < 6:
        return 0.0
    late_start = int(n * 0.6)
    late_slope = _linear_slope(flows[late_start:], dt)
    overall_slope = _linear_slope(flows, dt)
    return late_slope - overall_slope


def _flow_shape_label(flows: list[float], dt: float) -> str:
    """Classify the trajectory of a flow window as FLAT, RAMPING_UP, or RAMPING_DOWN.

    Uses linear-regression slope in ml/s per second.  Bands chosen so that
    a designed 0.1 ml/s² ramp registers as RAMPING_UP / _DOWN while normal
    noise in a flat extraction stays FLAT.  Provided as context so the agent
    can interpret ``flow_spread_ml_s`` correctly — high spread on a ramping
    profile is intentional, not channeling.
    """
    if len(flows) < 2:
        return "FLAT"
    slope = _linear_slope(flows, dt)
    if slope > 0.03:
        return "RAMPING_UP"
    if slope < -0.03:
        return "RAMPING_DOWN"
    return "FLAT"


def _residual_std_vs_target(samples: list[dict]) -> Optional[float]:
    """Population std of (actual_flow - target_flow) where target is commanded.

    Measures how far actual puck flow strayed from the profile's commanded
    trajectory.  Returns ``None`` when target_flow is not commanded for
    enough samples (pressure-led profiles), so the agent can distinguish
    "flow tracked target poorly" from "no target was set".

    Samples with ``tf <= 0`` are excluded as they are not part of the
    commanded trajectory.
    """
    pairs = [
        (s.get('pf', 0.0), s['tf'])
        for s in samples
        if s.get('tf', 0.0) > 0
    ]
    if len(pairs) < 3:
        return None
    residuals = [actual - target for actual, target in pairs]
    return _safe_std(residuals)


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


def _strip_flow_edges(
    pressures: list[float],
    flows: list[float],
    samples: list[dict],
    thr: float = 0.1,
) -> tuple[list[float], list[float], list[dict], tuple[int, int]]:
    """Remove leading and trailing samples where flow is below ``thr`` ml/s.

    Complements ``_trim_ramp_up`` by also handling:
    - Leading zero-flow samples (pressure ramped, valve not yet open)
    - Trailing zero-flow samples (volumetric cutoff hit, pressure trapped)

    Returns trimmed pressures, flows, samples, and a ``(lead, tail)`` tuple
    counting samples removed from each end.  When every sample is below
    threshold the returned lists are empty and ``lead == len(flows)``.
    """
    n = len(flows)
    i = 0
    while i < n and flows[i] < thr:
        i += 1
    if i == n:
        return [], [], [], (n, 0)
    j = n - 1
    while j > i and flows[j] < thr:
        j -= 1
    return (
        pressures[i:j + 1],
        flows[i:j + 1],
        samples[i:j + 1],
        (i, n - 1 - j),
    )


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
    flow_jitter: float,
    flow_vs_tgt: Optional[float],
    pressure_max_drop_rate: float,
    flow_acceleration_late: float,
    pressure_jitter: float,
) -> str:
    """Assess channeling risk from four independent indicators.

    The 4 indicators each contribute 0-2 points (8 total):

    1. **flow_jitter** — sample-to-sample flow instability (V5). Primary
       channeling fingerprint; insensitive to designed flow ramps.
    2. **flow_vs_tgt / pressure_jitter** — how well the shot tracked the
       commanded variable.  ``flow_vs_tgt`` is used when the profile
       commands flow; ``pressure_jitter`` is the fallback for pressure-led
       profiles.  Either indicator occupies the same scoring slot so the
       total score range (0-8) is consistent across profile types.
    3. **pressure_max_drop_rate** — steepest single-sample pressure
       collapse.  Catches cliff-type events (channel opening) that jitter
       may miss when isolated to a single sample.
    4. **flow_acceleration_late** — late-shot flow runaway trend.  Catches
       sustained channeling that develops at the end of extraction.

    Mapping: 0-1 → LOW, 2-3 → MODERATE, 4-5 → HIGH, 6-8 → VERY_HIGH.
    """
    score = 0

    # Indicator 1: flow jitter
    if flow_jitter >= 0.05:
        score += 1
    if flow_jitter >= 0.10:
        score += 1

    # Indicator 2: flow-vs-target primary, pressure-jitter fallback
    if flow_vs_tgt is not None:
        if flow_vs_tgt >= 0.35:
            score += 1
        if flow_vs_tgt >= 0.70:
            score += 1
    else:
        if pressure_jitter >= 0.10:
            score += 1
        if pressure_jitter >= 0.20:
            score += 1

    # Indicator 3: pressure cliff
    if pressure_max_drop_rate <= -1.5:
        score += 1
    if pressure_max_drop_rate <= -3.0:
        score += 1

    # Indicator 4: late-shot flow runaway
    if flow_acceleration_late >= 0.05:
        score += 1
    if flow_acceleration_late >= 0.10:
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


def _window_confidence(n: int) -> str:
    """Map steady-state sample count to a confidence label for channeling assessment.

    Drives ``annotations.window_confidence`` so the agent can discount
    HIGH/VERY_HIGH ratings when they come from a tiny window.
    """
    if n < _MIN_STEADY_STATE_SAMPLES:
        return "INSUFFICIENT"
    if n < 8:
        return "LOW"
    if n < 15:
        return "MEDIUM"
    return "HIGH"


def _channeling_primary_signal(
    flow_jitter: float,
    flow_vs_tgt: Optional[float],
    pressure_max_drop_rate: float,
    flow_acceleration_late: float,
    pressure_jitter: float,
) -> str:
    """Name the indicators contributing to a non-zero risk score.

    Returns a comma-separated list (``"flow_jitter,pressure_cliff"``) or
    ``"none"``.  Tells the agent *why* the rating fired without requiring
    it to re-derive the scoring from raw numbers.
    """
    signals: list[str] = []
    if flow_jitter >= 0.05:
        signals.append("flow_jitter")
    if flow_vs_tgt is not None and flow_vs_tgt >= 0.35:
        signals.append("flow_vs_target")
    elif flow_vs_tgt is None and pressure_jitter >= 0.10:
        signals.append("pressure_jitter_fallback")
    if pressure_max_drop_rate <= -1.5:
        signals.append("pressure_cliff")
    if flow_acceleration_late >= 0.05:
        signals.append("late_flow_runaway")
    return ",".join(signals) if signals else "none"


def _channeling_guidance(
    risk: str,
    primary: str,
    confidence: str,
    flow_shape: str,
) -> str:
    """One-sentence interpretation to orient the agent before it reads numbers.

    Not a replacement for the indicator values — a framing layer so the
    agent can cross-check its own reading of the data against an expected
    interpretation.
    """
    if risk == "INSUFFICIENT_DATA":
        return (
            "Steady-state window too short for a reliable channeling "
            "assessment — treat other diagnostics as the primary signal."
        )
    if risk == "LOW":
        if primary != "none":
            return (
                f"LOW overall; sub-threshold signals noted ({primary}) "
                "but did not aggregate into concern."
            )
        if flow_shape == "FLAT":
            return "Flat flow held steadily — no channeling signature."
        return (
            f"Flow traces a {flow_shape.lower().replace('_', ' ')} "
            "trajectory cleanly — no channeling signature."
        )
    if confidence in ("LOW", "MEDIUM") and risk in ("HIGH", "VERY_HIGH"):
        return (
            f"{risk} rating from a small steady-state window "
            f"(confidence={confidence}); verify against flow_shape and primary_signal."
        )
    signal_count = 0 if primary == "none" else primary.count(",") + 1
    if signal_count >= 2:
        return (
            f"{signal_count} independent indicators align ({primary}) — "
            "channeling likely real."
        )
    return f"Single-indicator flag ({primary}); verify against other diagnostics."


def _build_channeling(
    brew_pressures: list[float],
    brew_flows: list[float],
    brew_samples: list[dict],
    dt: float,
) -> ChannelingIndicators:
    """Compute the full ChannelingIndicators block from brew-phase data.

    Applies the two-step steady-state trim (``_trim_ramp_up`` then
    ``_strip_flow_edges``) and derives all indicators, descriptors, and
    annotations.  Shared between full-shot and per-phase code paths so
    both produce the same schema.
    """
    # Step 1: pressure-based ramp-up trim
    ss_pressures, ss_flows, ss_samples = _trim_ramp_up(
        brew_pressures, brew_flows, brew_samples,
    )
    ramp_excluded = len(brew_pressures) - len(ss_pressures)

    # Step 2: strip leading/trailing zero-flow samples (valve closed / cutoff)
    ss_pressures, ss_flows, ss_samples, (zf_lead, zf_tail) = _strip_flow_edges(
        ss_pressures, ss_flows, ss_samples,
    )

    n = len(ss_pressures)
    confidence = _window_confidence(n)

    if n < _MIN_STEADY_STATE_SAMPLES:
        return ChannelingIndicators(
            flow_jitter_ml_s=0.0,
            flow_vs_target_residual_ml_s=None,
            pressure_max_drop_rate_bar_s=0.0,
            flow_acceleration_late_ml_s2=0.0,
            flow_spread_ml_s=_round2(_safe_std(ss_flows)) if ss_flows else 0.0,
            pressure_jitter_bar=0.0,
            channeling_risk="INSUFFICIENT_DATA",
            annotations={
                "flow_jitter": "N/A",
                "flow_vs_target": "N/A",
                "pressure_drop": "N/A",
                "late_flow_trend": "N/A",
                "pressure_jitter": "N/A",
                "flow_shape": _flow_shape_label(ss_flows, dt),
                "window_confidence": confidence,
                "primary_signal": "none",
                "guidance": _channeling_guidance(
                    "INSUFFICIENT_DATA", "none", confidence,
                    _flow_shape_label(ss_flows, dt),
                ),
                "note": (
                    f"Only {n} steady-state samples after trim "
                    f"(ramp_excluded={ramp_excluded}, "
                    f"zero_flow_lead={zf_lead}, zero_flow_tail={zf_tail}); "
                    f"need {_MIN_STEADY_STATE_SAMPLES} for assessment."
                ),
            },
        )

    # Indicators — keep raw values for scoring, round only for output
    flow_jitter_raw = _jitter_std(ss_flows)
    pressure_jitter_raw = _jitter_std(ss_pressures)
    flow_vs_tgt_raw = _residual_std_vs_target(ss_samples)
    p_derivatives = [
        (ss_pressures[i] - ss_pressures[i - 1]) / dt
        for i in range(1, len(ss_pressures))
    ]
    p_max_drop_raw = min(p_derivatives) if p_derivatives else 0.0
    f_accel_late_raw = _late_flow_runaway(ss_flows, dt)

    flow_jitter = _round2(flow_jitter_raw)
    pressure_jitter = _round2(pressure_jitter_raw)
    flow_vs_tgt = (
        _round2(flow_vs_tgt_raw) if flow_vs_tgt_raw is not None else None
    )
    p_max_drop = _round2(p_max_drop_raw)
    f_accel_late = _round2(f_accel_late_raw)

    # Descriptors
    flow_spread = _round2(_safe_std(ss_flows))
    flow_shape = _flow_shape_label(ss_flows, dt)

    risk = _assess_channeling_risk(
        flow_jitter=flow_jitter_raw,
        flow_vs_tgt=flow_vs_tgt_raw,
        pressure_max_drop_rate=p_max_drop_raw,
        flow_acceleration_late=f_accel_late_raw,
        pressure_jitter=pressure_jitter_raw,
    )
    primary = _channeling_primary_signal(
        flow_jitter_raw, flow_vs_tgt_raw, p_max_drop_raw, f_accel_late_raw, pressure_jitter_raw,
    )

    annotations: dict[str, str] = {
        "flow_jitter": _annotate_ascending(flow_jitter, _FLOW_JITTER_BANDS),
        "flow_vs_target": (
            _annotate_ascending(flow_vs_tgt, _FLOW_VS_TARGET_BANDS)
            if flow_vs_tgt is not None else "N/A"
        ),
        "pressure_drop": _annotate_descending(
            p_max_drop, _PRESSURE_DROP_RATE_BANDS
        ),
        "late_flow_trend": _annotate_ascending(
            f_accel_late, _FLOW_ACCELERATION_BANDS
        ),
        "pressure_jitter": _annotate_ascending(
            pressure_jitter, _PRESSURE_JITTER_BANDS
        ),
        "flow_shape": flow_shape,
        "window_confidence": confidence,
        "primary_signal": primary,
        "guidance": _channeling_guidance(
            risk, primary, confidence, flow_shape,
        ),
    }
    if ramp_excluded or zf_lead or zf_tail:
        parts = []
        if ramp_excluded:
            parts.append(f"{ramp_excluded} ramp-up")
        if zf_lead:
            parts.append(f"{zf_lead} leading zero-flow")
        if zf_tail:
            parts.append(f"{zf_tail} trailing zero-flow")
        annotations["note"] = (
            f"Trimmed {', '.join(parts)} samples before assessment."
        )

    return ChannelingIndicators(
        flow_jitter_ml_s=flow_jitter,
        flow_vs_target_residual_ml_s=flow_vs_tgt,
        pressure_max_drop_rate_bar_s=p_max_drop,
        flow_acceleration_late_ml_s2=f_accel_late,
        flow_spread_ml_s=flow_spread,
        pressure_jitter_bar=pressure_jitter,
        channeling_risk=risk,
        annotations=annotations,
    )


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
    'saturate', 'saturation',  # Stock GaggiMate phase name — see issue #18
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

    resistance = ResistanceDiagnostics(
        avg=r_avg,
        std=r_std,
        slope=r_slope,
        peak=r_peak,
        peak_timing_pct=r_peak_timing,
        annotations={
            "level": _annotate_ascending(r_avg, _RESISTANCE_LEVEL_BANDS),
            "stability": _annotate_ascending(r_std, _RESISTANCE_STABILITY_BANDS),
            "erosion": _annotate_descending(r_slope, _RESISTANCE_SLOPE_BANDS),
            "saturation": _annotate_ascending(
                r_peak_timing, _RESISTANCE_PEAK_TIMING_BANDS
            ),
        },
    )

    # ── CHANNELING INDICATORS ────────────────────────────────
    channeling = _build_channeling(
        brew_pressures, brew_flows, brew_samples, dt,
    )

    # ── TEMPERATURE DIAGNOSTICS ──────────────────────────────
    temp_deviations = [
        ct - tt
        for ct, tt in zip(brew_temps, brew_target_temps)
        if tt > 0  # Skip if no target temp recorded
    ]
    t_overshoot = max(temp_deviations) if temp_deviations else 0.0
    t_undershoot = abs(min(temp_deviations)) if temp_deviations else 0.0
    t_std = _round2(_safe_std(brew_temps))

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

        # Delegate channeling assessment to the shared builder so the
        # per-phase and full-shot outputs share schema + behavior.
        ch = _build_channeling(pressures, flows, phase_samples, dt)

        result["resistance_avg"] = r_avg
        result["resistance_slope"] = r_slope
        result["channeling_risk"] = ch["channeling_risk"]
        result["flow_jitter_ml_s"] = ch["flow_jitter_ml_s"]
        result["pressure_jitter_bar"] = ch["pressure_jitter_bar"]

        annotations["resistance_level"] = _annotate_ascending(
            r_avg, _RESISTANCE_LEVEL_BANDS
        )
        annotations["resistance_erosion"] = _annotate_descending(
            r_slope, _RESISTANCE_SLOPE_BANDS
        )
        annotations["channeling"] = ch["channeling_risk"]
        # Fold the rich channeling annotations under a namespace so
        # the agent can read them without key collisions.
        for key in (
            "flow_jitter", "flow_vs_target", "pressure_drop",
            "late_flow_trend", "pressure_jitter", "flow_shape",
            "window_confidence", "primary_signal", "guidance",
        ):
            annotations[f"channeling_{key}"] = ch["annotations"][key]
        if "note" in ch["annotations"]:
            annotations["channeling_note"] = ch["annotations"]["note"]

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

    # Resistance — over the steady-state extraction window (same trim the
    # channeling builder uses) so a low-flow ramp/bloom doesn't inflate the
    # summary number; fall back to the whole brew window when the trimmed
    # window is too short. Deliberate divergence — see issue #17.
    ss_pressures, ss_flows, ss_samples = _trim_ramp_up(
        brew_pressures, brew_flows, brew_samples,
    )
    ss_pressures, ss_flows, ss_samples, _ = _strip_flow_edges(
        ss_pressures, ss_flows, ss_samples,
    )
    if len(ss_flows) < _MIN_STEADY_STATE_SAMPLES:
        ss_pressures, ss_flows, ss_samples = brew_pressures, brew_flows, brew_samples
    r_values = [p / (f * f) for p, f in zip(ss_pressures, ss_flows) if f > 0.1]
    r_avg = _round2(_safe_mean(r_values))
    r_slope = _round2(_linear_slope(r_values, dt))

    # Channeling — delegate to shared builder so the summary shares the
    # same trim + scoring logic as the full and per-phase outputs.
    ch = _build_channeling(brew_pressures, brew_flows, brew_samples, dt)
    risk = ch["channeling_risk"]

    # Temperature
    t_std = _round2(_safe_std(brew_temps))

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
        "resistance_level": _annotate_ascending(r_avg, _RESISTANCE_LEVEL_BANDS),
        "resistance_erosion": _annotate_descending(r_slope, _RESISTANCE_SLOPE_BANDS),
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
