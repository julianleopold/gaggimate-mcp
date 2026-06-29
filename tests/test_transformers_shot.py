"""Tests for shot transformer."""

from gaggimate_mcp.parsers.shot import ShotData, PhaseTransition
from gaggimate_mcp.transformers.shot import (
    transform_shot_for_ai,
    calculate_summary,
    process_phases,
    calculate_total_volume,
    compute_shot_diagnostics,
    compute_summary_diagnostics,
    _safe_mean,
    _safe_std,
    _linear_slope,
    _compute_rmse,
    _classify_phase,
    _classify_phase_by_name,
    _classify_phase_by_telemetry,
    _compute_phase_diagnostics,
    _compute_profile_compliance,
    _get_brew_phase_samples,
    _annotate_ascending,
    _annotate_descending,
    _assess_channeling_risk,
    _build_phases,
    _pressure_volatility_label,
    _trim_ramp_up,
    _MIN_STEADY_STATE_SAMPLES,
    VALID_DETAIL_LEVELS,
    _PRESSURE_VOLATILITY_BANDS,
    _RESISTANCE_SLOPE_BANDS,
    _PRESSURE_DROP_RATE_BANDS,
    _PROFILE_ADHERENCE_BANDS,
    _PRESSURE_OVERSHOOT_BANDS,
    _FLOW_DEVIATION_BANDS,
    _RAMP_RATE_BANDS,
    _TAPER_SMOOTHNESS_BANDS,
    _PREINFUSION_KEYWORDS,
    _DECLINE_KEYWORDS,
)


class TestShotTransformer:
    """Test shot transformation for AI analysis."""

    def test_calculate_total_volume(self):
        """Test volume calculation from flow samples."""
        samples = [
            {'pf': 2.0},  # 2 ml/s
            {'pf': 3.0},  # 3 ml/s
            {'pf': 2.5},  # 2.5 ml/s
        ]
        interval_ms = 100  # 0.1 seconds

        volume = calculate_total_volume(samples, interval_ms)

        # (2.0 + 3.0 + 2.5) * 0.1 = 0.75 ml
        assert volume == 0.8  # Rounded to 1 decimal

    def test_calculate_summary_basic(self):
        """Test summary statistics calculation."""
        shot = ShotData(
            id='1',
            version=4,
            fields_mask=0xFF,
            sample_count=5,
            sample_interval=100,
            profile_id='test',
            profile_name='Test Profile',
            timestamp=1640000000,
            rating=4,
            duration=25000,
            weight=36.0,
            samples=[
                {'t': 0, 'ct': 90.0, 'tt': 93.0, 'cp': 0.0, 'tp': 9.0, 'pf': 0.0},
                {'t': 100, 'ct': 91.0, 'tt': 93.0, 'cp': 2.0, 'tp': 9.0, 'pf': 1.0},
                {'t': 200, 'ct': 92.0, 'tt': 93.0, 'cp': 9.0, 'tp': 9.0, 'pf': 2.5},
                {'t': 300, 'ct': 93.0, 'tt': 93.0, 'cp': 8.5, 'tp': 9.0, 'pf': 2.0},
                {'t': 400, 'ct': 93.0, 'tt': 93.0, 'cp': 8.0, 'tp': 9.0, 'pf': 1.5},
            ],
            phases=[],
        )

        summary = calculate_summary(shot)

        # Temperature
        assert summary['temperature']['min_c'] == 90.0
        assert summary['temperature']['max_c'] == 93.0
        assert summary['temperature']['avg_c'] == 91.8
        assert summary['temperature']['target_avg_c'] == 93.0

        # Pressure
        assert summary['pressure']['min_bar'] == 0.0
        assert summary['pressure']['max_bar'] == 9.0
        assert summary['pressure']['avg_bar'] == 5.5
        assert summary['pressure']['peak_time_s'] == 0.2  # At sample 2

        # Flow
        assert summary['flow']['total_volume_ml'] == 0.7
        assert summary['flow']['avg_flow_ml_s'] == 1.4
        assert summary['flow']['peak_flow_ml_s'] == 2.5
        assert summary['flow']['time_to_first_drip_s'] == 0.1  # At sample 1

        # Extraction timing
        # Preinfusion is 0.2s (time to reach 50% of peak pressure)
        # Total time is 25.0s (from shot.duration)
        # Main extraction is total - preinfusion
        assert summary['extraction']['preinfusion_time_s'] == 0.2
        assert summary['extraction']['main_extraction_time_s'] == 24.8
        assert summary['extraction']['total_time_s'] == 25.0

    def test_process_phases_with_transitions(self):
        """Test phase processing with defined transitions."""
        shot = ShotData(
            id='1',
            version=5,
            fields_mask=0xFF,
            sample_count=6,
            sample_interval=100,
            profile_id='test',
            profile_name='Test Profile',
            timestamp=1640000000,
            rating=4,
            duration=30000,
            weight=40.0,
            samples=[
                {'t': 0, 'ct': 90.0, 'cp': 2.0, 'pf': 0.5, 'phase': 0},
                {'t': 100, 'ct': 91.0, 'cp': 3.0, 'pf': 0.8, 'phase': 0},
                {'t': 200, 'ct': 92.0, 'cp': 4.0, 'pf': 1.0, 'phase': 0},
                {'t': 300, 'ct': 93.0, 'cp': 9.0, 'pf': 2.5, 'phase': 1},
                {'t': 400, 'ct': 93.0, 'cp': 8.5, 'pf': 2.0, 'phase': 1},
                {'t': 500, 'ct': 93.0, 'cp': 8.0, 'pf': 1.5, 'phase': 1},
            ],
            phases=[
                PhaseTransition(sample_index=0, phase_number=0, phase_name='Preinfusion'),
                PhaseTransition(sample_index=3, phase_number=1, phase_name='Extraction'),
            ],
        )

        phases = process_phases(shot)

        assert len(phases) == 2

        # Preinfusion phase
        assert phases[0]['name'] == 'Preinfusion'
        assert phases[0]['phase_number'] == 0
        assert phases[0]['start_time_seconds'] == 0.0
        assert phases[0]['duration_seconds'] == 0.3
        assert phases[0]['sample_count'] == 3
        assert phases[0]['avg_temperature_c'] == 91.0
        assert phases[0]['avg_pressure_bar'] == 3.0
        assert len(phases[0]['samples']) == 3  # Beginning, middle, end

        # Extraction phase
        assert phases[1]['name'] == 'Extraction'
        assert phases[1]['phase_number'] == 1
        assert phases[1]['start_time_seconds'] == 0.3
        assert phases[1]['duration_seconds'] == 0.3
        assert phases[1]['sample_count'] == 3

    def test_process_phases_without_transitions(self):
        """Test phase processing when no transitions defined."""
        shot = ShotData(
            id='1',
            version=4,
            fields_mask=0xFF,
            sample_count=3,
            sample_interval=100,
            profile_id='test',
            profile_name='Test Profile',
            timestamp=1640000000,
            rating=4,
            duration=30000,
            weight=40.0,
            samples=[
                {'t': 0, 'ct': 90.0, 'cp': 2.0, 'pf': 0.5},
                {'t': 100, 'ct': 92.0, 'cp': 9.0, 'pf': 2.5},
                {'t': 200, 'ct': 93.0, 'cp': 8.0, 'pf': 2.0},
            ],
            phases=[],
        )

        phases = process_phases(shot)

        # Should create single 'extraction' phase
        assert len(phases) == 1
        assert phases[0]['name'] == 'extraction'
        assert phases[0]['phase_number'] == 0
        assert phases[0]['start_time_seconds'] == 0.0
        assert phases[0]['duration_seconds'] == 30.0
        assert phases[0]['sample_count'] == 3

    def test_transform_shot_for_ai(self):
        """Test complete shot transformation."""
        shot = ShotData(
            id='000123',
            version=5,
            fields_mask=0xFF,
            sample_count=4,
            sample_interval=100,
            profile_id='medium_roast',
            profile_name='Medium Roast',
            timestamp=1640000000,
            rating=5,
            duration=28000,
            weight=38.5,
            samples=[
                {'t': 0, 'ct': 90.0, 'tt': 93.0, 'cp': 2.0, 'tp': 9.0, 'pf': 0.5, 'phase': 0},
                {'t': 100, 'ct': 91.0, 'tt': 93.0, 'cp': 4.0, 'tp': 9.0, 'pf': 1.0, 'phase': 0},
                {'t': 200, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'tp': 9.0, 'pf': 2.5, 'phase': 1},
                {'t': 300, 'ct': 93.0, 'tt': 93.0, 'cp': 8.5, 'tp': 9.0, 'pf': 2.0, 'phase': 1},
            ],
            phases=[
                PhaseTransition(sample_index=0, phase_number=0, phase_name='Preinfusion'),
                PhaseTransition(sample_index=2, phase_number=1, phase_name='Extraction'),
            ],
        )

        transformed = transform_shot_for_ai(shot)

        # Metadata
        assert transformed['shot_id'] == '000123'
        assert transformed['profile_name'] == 'Medium Roast'
        assert transformed['profile_id'] == 'medium_roast'
        assert transformed['timestamp'] == 1640000000
        assert transformed['duration_seconds'] == 28.0
        assert transformed['final_weight_g'] == 38.5

        # Summary
        assert 'summary' in transformed
        assert 'temperature' in transformed['summary']
        assert 'pressure' in transformed['summary']
        assert 'flow' in transformed['summary']
        assert 'extraction' in transformed['summary']

        # Phases
        assert len(transformed['phases']) == 2
        assert transformed['phases'][0]['name'] == 'Preinfusion'
        assert transformed['phases'][1]['name'] == 'Extraction'

    def test_transform_shot_no_weight(self):
        """Test transformation when weight is not available."""
        shot = ShotData(
            id='1',
            version=4,
            fields_mask=0xFF,
            sample_count=2,
            sample_interval=100,
            profile_id='test',
            profile_name='Test',
            timestamp=1640000000,
            rating=0,
            duration=25000,
            weight=None,
            samples=[
                {'t': 0, 'ct': 90.0, 'cp': 0.0, 'pf': 0.0},
                {'t': 100, 'ct': 93.0, 'cp': 9.0, 'pf': 2.0},
            ],
            phases=[],
        )

        transformed = transform_shot_for_ai(shot)

        assert transformed['final_weight_g'] is None

    def test_transform_shot_incomplete(self):
        """Test transformation with incomplete shot data."""
        shot = ShotData(
            id='1',
            version=4,
            fields_mask=0xFF,
            sample_count=2,
            sample_interval=100,
            profile_id='test',
            profile_name='Test',
            timestamp=1640000000,
            rating=0,
            duration=25000,
            weight=None,
            samples=[
                {'t': 0, 'ct': 90.0, 'cp': 0.0, 'pf': 0.0},
                {'t': 100, 'ct': 93.0, 'cp': 9.0, 'pf': 2.0},
            ],
            phases=[],
            incomplete=True,
        )

        transformed = transform_shot_for_ai(shot)

        # Should still transform successfully
        assert transformed['shot_id'] == '1'
        assert len(transformed['phases']) == 1


class TestHelperFunctions:
    """Tests for diagnostic helper functions."""

    def test_safe_mean_empty(self):
        assert _safe_mean([]) == 0.0

    def test_safe_mean_values(self):
        assert _safe_mean([2.0, 4.0, 6.0]) == 4.0

    def test_safe_std_empty(self):
        assert _safe_std([]) == 0.0

    def test_safe_std_single(self):
        assert _safe_std([5.0]) == 0.0

    def test_safe_std_values(self):
        # std of [2, 4, 6] = sqrt(((−2)^2 + 0^2 + 2^2) / 3) = sqrt(8/3) ≈ 1.633
        result = _safe_std([2.0, 4.0, 6.0])
        assert abs(result - 1.633) < 0.01

    def test_linear_slope_flat(self):
        assert _linear_slope([5.0, 5.0, 5.0, 5.0], 1.0) == 0.0

    def test_linear_slope_increasing(self):
        # [0, 1, 2, 3] with dt=1 → slope = 1.0
        result = _linear_slope([0.0, 1.0, 2.0, 3.0], 1.0)
        assert abs(result - 1.0) < 0.01

    def test_linear_slope_decreasing(self):
        # [3, 2, 1, 0] with dt=0.5 → slope = -2.0 (per second)
        result = _linear_slope([3.0, 2.0, 1.0, 0.0], 0.5)
        assert abs(result - (-2.0)) < 0.01

    def test_linear_slope_insufficient(self):
        assert _linear_slope([], 1.0) == 0.0
        assert _linear_slope([5.0], 1.0) == 0.0

    def test_annotate_ascending(self):
        assert _annotate_ascending(0.1, _PRESSURE_VOLATILITY_BANDS) == "VERY_STABLE"
        assert _annotate_ascending(0.2, _PRESSURE_VOLATILITY_BANDS) == "STABLE"
        assert _annotate_ascending(0.5, _PRESSURE_VOLATILITY_BANDS) == "MODERATE_JITTER"
        assert _annotate_ascending(0.8, _PRESSURE_VOLATILITY_BANDS) == "JITTERY"
        assert _annotate_ascending(1.5, _PRESSURE_VOLATILITY_BANDS) == "VOLATILE"

    def test_annotate_descending(self):
        assert _annotate_descending(0.1, _RESISTANCE_SLOPE_BANDS) == "INCREASING"
        assert _annotate_descending(0.0, _RESISTANCE_SLOPE_BANDS) == "FLAT"
        assert _annotate_descending(-0.05, _RESISTANCE_SLOPE_BANDS) == "GRADUAL_DECLINE"
        assert _annotate_descending(-0.1, _RESISTANCE_SLOPE_BANDS) == "MODERATE_DECLINE"
        assert _annotate_descending(-0.2, _RESISTANCE_SLOPE_BANDS) == "STEEP_DECLINE"

    def test_annotate_descending_pressure_drop(self):
        assert _annotate_descending(-0.3, _PRESSURE_DROP_RATE_BANDS) == "NORMAL"
        assert _annotate_descending(-1.5, _PRESSURE_DROP_RATE_BANDS) == "MODERATE_DROP"
        assert _annotate_descending(-3.0, _PRESSURE_DROP_RATE_BANDS) == "STEEP_DROP"
        assert _annotate_descending(-6.0, _PRESSURE_DROP_RATE_BANDS) == "CLIFF"

    def test_assess_channeling_risk_low(self):
        # All indicators below primary thresholds → LOW
        assert _assess_channeling_risk(
            flow_jitter=0.02,
            flow_vs_tgt=0.1,
            pressure_max_drop_rate=-0.3,
            flow_acceleration_late=0.01,
            pressure_jitter=0.03,
        ) == "LOW"

    def test_assess_channeling_risk_moderate(self):
        # flow_jitter 0.15 → +2 = MODERATE
        assert _assess_channeling_risk(
            flow_jitter=0.15,
            flow_vs_tgt=0.1,
            pressure_max_drop_rate=-0.3,
            flow_acceleration_late=0.01,
            pressure_jitter=0.03,
        ) == "MODERATE"

    def test_assess_channeling_risk_high(self):
        # jitter 0.15 (+2) + vs_tgt 0.4 (+1) + drop -2.0 (+1) = 4 → HIGH
        assert _assess_channeling_risk(
            flow_jitter=0.15,
            flow_vs_tgt=0.4,
            pressure_max_drop_rate=-2.0,
            flow_acceleration_late=0.01,
            pressure_jitter=0.03,
        ) == "HIGH"

    def test_assess_channeling_risk_very_high(self):
        # All indicators past second threshold → 8 → VERY_HIGH
        assert _assess_channeling_risk(
            flow_jitter=0.25,
            flow_vs_tgt=0.9,
            pressure_max_drop_rate=-4.0,
            flow_acceleration_late=0.15,
            pressure_jitter=0.5,
        ) == "VERY_HIGH"


class TestBrewPhaseExtraction:
    """Tests for brew phase sample extraction."""

    def test_with_phases_skips_preinfusion(self):
        shot = ShotData(
            id='1', version=5, fields_mask=0xFF, sample_count=6,
            sample_interval=100, profile_id='test', profile_name='Test',
            timestamp=1640000000, rating=0, duration=30000, weight=40.0,
            samples=[
                {'t': 0, 'ct': 90.0, 'cp': 2.0, 'pf': 0.5, 'phase': 0},
                {'t': 100, 'ct': 91.0, 'cp': 3.0, 'pf': 0.8, 'phase': 0},
                {'t': 200, 'ct': 92.0, 'cp': 4.0, 'pf': 1.0, 'phase': 0},
                {'t': 300, 'ct': 93.0, 'cp': 9.0, 'pf': 2.5, 'phase': 1},
                {'t': 400, 'ct': 93.0, 'cp': 8.5, 'pf': 2.0, 'phase': 1},
                {'t': 500, 'ct': 93.0, 'cp': 8.0, 'pf': 1.5, 'phase': 1},
            ],
            phases=[
                PhaseTransition(sample_index=0, phase_number=0, phase_name='Preinfusion'),
                PhaseTransition(sample_index=3, phase_number=1, phase_name='Extraction'),
            ],
        )
        brew = _get_brew_phase_samples(shot)
        assert len(brew) == 3
        assert brew[0]['cp'] == 9.0  # First brew sample

    def test_without_phases_uses_pressure_threshold(self):
        shot = ShotData(
            id='1', version=4, fields_mask=0xFF, sample_count=5,
            sample_interval=100, profile_id='test', profile_name='Test',
            timestamp=1640000000, rating=0, duration=25000, weight=None,
            samples=[
                {'t': 0, 'cp': 0.0, 'pf': 0.0},
                {'t': 100, 'cp': 2.0, 'pf': 0.5},
                {'t': 200, 'cp': 5.0, 'pf': 1.5},
                {'t': 300, 'cp': 9.0, 'pf': 2.5},
                {'t': 400, 'cp': 8.0, 'pf': 2.0},
            ],
            phases=[],
        )
        brew = _get_brew_phase_samples(shot)
        # 50% of peak (9.0) = 4.5, first sample >= 4.5 is at index 2 (5.0)
        assert len(brew) == 3
        assert brew[0]['cp'] == 5.0

    def test_empty_samples(self):
        shot = ShotData(
            id='1', version=4, fields_mask=0xFF, sample_count=0,
            sample_interval=100, profile_id='test', profile_name='Test',
            timestamp=1640000000, rating=0, duration=0, weight=None,
            samples=[], phases=[],
        )
        assert _get_brew_phase_samples(shot) == []


class TestShotDiagnostics:
    """Tests for complete shot diagnostics computation."""

    def _make_shot(self, samples, phases=None, **kwargs):
        """Helper to create a ShotData with defaults."""
        defaults = dict(
            id='000100', version=5, fields_mask=0xFF,
            sample_count=len(samples), sample_interval=100,
            profile_id='test', profile_name='Test Profile',
            timestamp=1640000000, rating=0, duration=30000,
            weight=40.0,
        )
        defaults.update(kwargs)
        return ShotData(
            samples=samples,
            phases=phases or [],
            **defaults,
        )

    def test_returns_none_for_too_few_samples(self):
        shot = self._make_shot([
            {'t': 0, 'ct': 90.0, 'cp': 2.0, 'pf': 0.5},
            {'t': 100, 'ct': 93.0, 'cp': 9.0, 'pf': 2.0},
        ])
        assert compute_shot_diagnostics(shot) is None

    def test_healthy_shot_diagnostics(self):
        """Test diagnostics for a well-extracted shot (stable, no channeling)."""
        samples = [
            # Pre-infusion (ramp)
            {'t': 0, 'ct': 92.0, 'tt': 93.0, 'cp': 3.0, 'pf': 0.5, 'v': 0.0, 'phase': 0},
            {'t': 100, 'ct': 92.5, 'tt': 93.0, 'cp': 5.0, 'pf': 1.2, 'v': 0.1, 'phase': 0},
            # Brew phase - very stable pressure and flow
            {'t': 200, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'pf': 2.0, 'v': 0.3, 'phase': 1},
            {'t': 300, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'pf': 2.0, 'v': 0.5, 'phase': 1},
            {'t': 400, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'pf': 2.0, 'v': 0.7, 'phase': 1},
            {'t': 500, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'pf': 2.0, 'v': 0.9, 'phase': 1},
            {'t': 600, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'pf': 2.0, 'v': 1.1, 'phase': 1},
            {'t': 700, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'pf': 2.0, 'v': 1.3, 'phase': 1},
        ]
        shot = self._make_shot(
            samples,
            phases=[
                PhaseTransition(sample_index=0, phase_number=0, phase_name='Preinfusion'),
                PhaseTransition(sample_index=2, phase_number=1, phase_name='Extraction'),
            ],
        )
        diag = compute_shot_diagnostics(shot)

        assert diag is not None

        # Resistance should be computed
        assert diag['resistance']['avg'] > 0
        assert diag['resistance']['peak'] > 0
        assert 0.0 <= diag['resistance']['peak_timing_pct'] <= 1.0

        # Annotations should be present
        assert 'level' in diag['resistance']['annotations']
        assert 'stability' in diag['resistance']['annotations']
        assert 'erosion' in diag['resistance']['annotations']

        # Channeling should be LOW for this stable shot
        assert diag['channeling']['channeling_risk'] == 'LOW'
        assert 'flow_jitter' in diag['channeling']['annotations']
        assert 'pressure_jitter' in diag['channeling']['annotations']
        assert 'guidance' in diag['channeling']['annotations']

        # Temperature should be stable
        assert diag['temperature']['stability_std_c'] < 1.0
        assert 'stability' in diag['temperature']['annotations']

        # Extraction metrics should be present
        assert diag['extraction']['pressure_auc_bar_s'] > 0
        assert 'pressure_trend' in diag['extraction']['annotations']
        assert 'flow_trend' in diag['extraction']['annotations']

        # Weight should detect scale
        assert diag['weight']['scale_connected'] is True
        assert diag['weight']['rate_avg_g_s'] is not None

    def test_channeling_shot_diagnostics(self):
        """Test diagnostics detect volatile pressure/flow (channeling)."""
        # Create a shot with jittery pressure and flow
        samples = [
            {'t': 0, 'ct': 93.0, 'tt': 93.0, 'cp': 3.0, 'pf': 0.5},
            {'t': 100, 'ct': 93.0, 'tt': 93.0, 'cp': 5.0, 'pf': 1.0},
            {'t': 200, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'pf': 2.0},
            {'t': 300, 'ct': 93.0, 'tt': 93.0, 'cp': 6.0, 'pf': 3.5},
            {'t': 400, 'ct': 93.0, 'tt': 93.0, 'cp': 9.5, 'pf': 1.5},
            {'t': 500, 'ct': 93.0, 'tt': 93.0, 'cp': 5.0, 'pf': 4.0},
            {'t': 600, 'ct': 93.0, 'tt': 93.0, 'cp': 8.0, 'pf': 2.0},
            {'t': 700, 'ct': 93.0, 'tt': 93.0, 'cp': 4.0, 'pf': 5.0},
        ]
        shot = self._make_shot(samples)
        diag = compute_shot_diagnostics(shot)

        assert diag is not None
        # Alternating flow and pressure produces high jitter on both variables
        assert diag['channeling']['flow_jitter_ml_s'] > 0.20
        assert diag['channeling']['pressure_jitter_bar'] > 0.20
        # Channeling risk should be elevated
        assert diag['channeling']['channeling_risk'] in ('MODERATE', 'HIGH', 'VERY_HIGH')

    def test_no_scale_data(self):
        """Test diagnostics when scale data is absent."""
        samples = [
            {'t': 0, 'ct': 93.0, 'tt': 93.0, 'cp': 3.0, 'pf': 0.5},
            {'t': 100, 'ct': 93.0, 'tt': 93.0, 'cp': 5.0, 'pf': 1.0},
            {'t': 200, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'pf': 2.0},
            {'t': 300, 'ct': 93.0, 'tt': 93.0, 'cp': 8.5, 'pf': 2.0},
            {'t': 400, 'ct': 93.0, 'tt': 93.0, 'cp': 8.0, 'pf': 2.1},
        ]
        shot = self._make_shot(samples)
        diag = compute_shot_diagnostics(shot)

        assert diag is not None
        assert diag['weight']['scale_connected'] is False
        assert diag['weight']['rate_avg_g_s'] is None
        assert diag['weight']['rate_std_g_s'] is None
        assert diag['weight']['annotations'].get('note') == 'No scale data available'

    def test_diagnostics_with_phases(self):
        """Test that diagnostics use brew phase when phases are defined."""
        samples = [
            # Pre-infusion (should be excluded from brew diagnostics)
            {'t': 0, 'ct': 90.0, 'tt': 93.0, 'cp': 1.0, 'pf': 0.2, 'phase': 0},
            {'t': 100, 'ct': 91.0, 'tt': 93.0, 'cp': 2.0, 'pf': 0.3, 'phase': 0},
            {'t': 200, 'ct': 92.0, 'tt': 93.0, 'cp': 3.0, 'pf': 0.5, 'phase': 0},
            # Extraction (brew phase)
            {'t': 300, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'pf': 2.0, 'phase': 1},
            {'t': 400, 'ct': 93.0, 'tt': 93.0, 'cp': 8.5, 'pf': 2.0, 'phase': 1},
            {'t': 500, 'ct': 93.0, 'tt': 93.0, 'cp': 8.0, 'pf': 2.1, 'phase': 1},
            {'t': 600, 'ct': 93.0, 'tt': 93.0, 'cp': 7.5, 'pf': 2.1, 'phase': 1},
        ]
        shot = self._make_shot(
            samples,
            phases=[
                PhaseTransition(sample_index=0, phase_number=0, phase_name='Preinfusion'),
                PhaseTransition(sample_index=3, phase_number=1, phase_name='Extraction'),
            ],
        )
        diag = compute_shot_diagnostics(shot)

        assert diag is not None
        # Resistance should be computed from brew-phase only
        # At brew phase: flow is around 2.0-2.1, so resistance should be calculable
        assert diag['resistance']['avg'] > 0
        # Only 4 brew samples — fewer than the 5-sample minimum for steady-state
        # channeling assessment, so we expect INSUFFICIENT_DATA
        assert diag['channeling']['channeling_risk'] == 'INSUFFICIENT_DATA'

    def test_transform_includes_diagnostics(self):
        """Test that transform_shot_for_ai includes full diagnostics at per_phase level."""
        samples = [
            {'t': 0, 'ct': 92.0, 'tt': 93.0, 'cp': 3.0, 'pf': 0.5},
            {'t': 100, 'ct': 93.0, 'tt': 93.0, 'cp': 5.0, 'pf': 1.0},
            {'t': 200, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'pf': 2.0},
            {'t': 300, 'ct': 93.0, 'tt': 93.0, 'cp': 8.5, 'pf': 2.0},
            {'t': 400, 'ct': 93.0, 'tt': 93.0, 'cp': 8.0, 'pf': 2.1},
            {'t': 500, 'ct': 93.0, 'tt': 93.0, 'cp': 7.5, 'pf': 2.1},
        ]
        shot = self._make_shot(samples)
        transformed = transform_shot_for_ai(shot, detail="per_phase")

        assert 'diagnostics' in transformed
        assert transformed['diagnostics'] is not None
        assert 'resistance' in transformed['diagnostics']
        assert 'channeling' in transformed['diagnostics']
        assert 'temperature' in transformed['diagnostics']
        assert 'extraction' in transformed['diagnostics']
        assert 'weight' in transformed['diagnostics']
        assert 'profile_compliance' in transformed['diagnostics']

    def test_transform_summary_diagnostics(self):
        """Test that default (summary) returns SummaryDiagnostics keys."""
        samples = [
            {'t': 0, 'ct': 92.0, 'tt': 93.0, 'cp': 3.0, 'pf': 0.5},
            {'t': 100, 'ct': 93.0, 'tt': 93.0, 'cp': 5.0, 'pf': 1.0},
            {'t': 200, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'pf': 2.0},
            {'t': 300, 'ct': 93.0, 'tt': 93.0, 'cp': 8.5, 'pf': 2.0},
            {'t': 400, 'ct': 93.0, 'tt': 93.0, 'cp': 8.0, 'pf': 2.1},
            {'t': 500, 'ct': 93.0, 'tt': 93.0, 'cp': 7.5, 'pf': 2.1},
        ]
        shot = self._make_shot(samples)
        transformed = transform_shot_for_ai(shot)  # default = summary

        assert transformed['detail_level'] == 'summary'
        diag = transformed['diagnostics']
        assert diag is not None
        assert 'resistance_avg' in diag
        assert 'channeling_risk' in diag
        assert 'temperature_stability_c' in diag
        assert 'annotations' in diag
        # Summary should NOT have full sub-dicts
        assert 'resistance' not in diag

    def test_transform_diagnostics_none_for_short_shot(self):
        """Test that diagnostics is None for a very short shot."""
        shot = self._make_shot([
            {'t': 0, 'ct': 90.0, 'cp': 0.0, 'pf': 0.0},
            {'t': 100, 'ct': 93.0, 'cp': 9.0, 'pf': 2.0},
        ])
        transformed = transform_shot_for_ai(shot)
        assert transformed['diagnostics'] is None

    def test_resistance_near_zero_flow_handling(self):
        """Test that near-zero flow doesn't cause division errors."""
        samples = [
            {'t': 0, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'pf': 0.0},  # zero flow
            {'t': 100, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'pf': 0.05},  # near-zero
            {'t': 200, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'pf': 2.0},
            {'t': 300, 'ct': 93.0, 'tt': 93.0, 'cp': 8.5, 'pf': 2.0},
            {'t': 400, 'ct': 93.0, 'tt': 93.0, 'cp': 8.0, 'pf': 2.1},
        ]
        shot = self._make_shot(samples)
        # Should not raise any errors
        diag = compute_shot_diagnostics(shot)
        assert diag is not None
        # Zero/near-zero flow samples should be excluded from resistance calc
        assert diag['resistance']['avg'] > 0

    def test_temperature_overshoot_and_undershoot(self):
        """Test temperature deviation detection."""
        samples = [
            {'t': 0, 'ct': 91.0, 'tt': 93.0, 'cp': 9.0, 'pf': 2.0},   # -2°C under
            {'t': 100, 'ct': 93.0, 'tt': 93.0, 'cp': 8.5, 'pf': 2.0},  # on target
            {'t': 200, 'ct': 95.0, 'tt': 93.0, 'cp': 8.0, 'pf': 2.1},  # +2°C over
            {'t': 300, 'ct': 93.0, 'tt': 93.0, 'cp': 7.5, 'pf': 2.1},  # on target
            {'t': 400, 'ct': 93.0, 'tt': 93.0, 'cp': 7.0, 'pf': 2.2},  # on target
        ]
        shot = self._make_shot(samples)
        diag = compute_shot_diagnostics(shot)

        assert diag is not None
        assert diag['temperature']['overshoot_c'] == 2.0
        assert diag['temperature']['undershoot_c'] == 2.0
        assert diag['temperature']['annotations']['overshoot'] == 'SIGNIFICANT'
        assert diag['temperature']['annotations']['undershoot'] == 'SIGNIFICANT'


class TestDetailLevels:
    """Tests for the 3-level detail system."""

    def _make_shot(self, samples, phases=None, **kwargs):
        defaults = dict(
            id='000100', version=5, fields_mask=0xFF,
            sample_count=len(samples), sample_interval=100,
            profile_id='test', profile_name='Test Profile',
            timestamp=1640000000, rating=0, duration=30000,
            weight=40.0,
        )
        defaults.update(kwargs)
        return ShotData(samples=samples, phases=phases or [], **defaults)

    def _standard_shot(self):
        """A shot with preinfusion and extraction phases."""
        samples = [
            {'t': 0, 'ct': 91.0, 'tt': 93.0, 'cp': 2.0, 'tp': 3.0, 'pf': 0.3, 'tf': 0.5, 'v': 0.0, 'phase': 0},
            {'t': 100, 'ct': 92.0, 'tt': 93.0, 'cp': 3.0, 'tp': 3.0, 'pf': 0.5, 'tf': 0.5, 'v': 0.0, 'phase': 0},
            {'t': 200, 'ct': 92.5, 'tt': 93.0, 'cp': 4.0, 'tp': 9.0, 'pf': 1.0, 'tf': 2.0, 'v': 0.1, 'phase': 0},
            {'t': 300, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'tp': 9.0, 'pf': 2.0, 'tf': 2.0, 'v': 0.3, 'phase': 1},
            {'t': 400, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'tp': 9.0, 'pf': 2.0, 'tf': 2.0, 'v': 0.5, 'phase': 1},
            {'t': 500, 'ct': 93.0, 'tt': 93.0, 'cp': 8.8, 'tp': 9.0, 'pf': 2.1, 'tf': 2.0, 'v': 0.7, 'phase': 1},
            {'t': 600, 'ct': 93.0, 'tt': 93.0, 'cp': 8.5, 'tp': 9.0, 'pf': 2.1, 'tf': 2.0, 'v': 0.9, 'phase': 1},
            {'t': 700, 'ct': 93.0, 'tt': 93.0, 'cp': 8.3, 'tp': 9.0, 'pf': 2.2, 'tf': 2.0, 'v': 1.1, 'phase': 1},
        ]
        phases = [
            PhaseTransition(sample_index=0, phase_number=0, phase_name='Preinfusion'),
            PhaseTransition(sample_index=3, phase_number=1, phase_name='Extraction'),
        ]
        return self._make_shot(samples, phases=phases)

    def test_valid_detail_levels(self):
        assert VALID_DETAIL_LEVELS == ("summary", "per_phase", "per_phase_detailed")

    def test_invalid_detail_falls_back_to_summary(self):
        shot = self._standard_shot()
        t = transform_shot_for_ai(shot, detail="bogus")
        assert t['detail_level'] == 'summary'

    def test_summary_no_samples_in_phases(self):
        shot = self._standard_shot()
        t = transform_shot_for_ai(shot, detail="summary")
        for phase in t['phases']:
            assert 'samples' not in phase

    def test_summary_no_phase_diagnostics(self):
        shot = self._standard_shot()
        t = transform_shot_for_ai(shot, detail="summary")
        for phase in t['phases']:
            assert 'diagnostics' not in phase

    def test_per_phase_has_diagnostics_no_samples(self):
        shot = self._standard_shot()
        t = transform_shot_for_ai(shot, detail="per_phase")
        for phase in t['phases']:
            assert 'samples' not in phase
            if phase['sample_count'] >= 3:
                assert 'diagnostics' in phase
                assert phase['diagnostics']['phase_type'] in ('preinfusion', 'brew', 'decline')

    def test_per_phase_detailed_has_samples_and_diagnostics(self):
        shot = self._standard_shot()
        t = transform_shot_for_ai(shot, detail="per_phase_detailed")
        for phase in t['phases']:
            assert 'samples' in phase
            assert len(phase['samples']) > 0
            if phase['sample_count'] >= 3:
                assert 'diagnostics' in phase
                assert phase['diagnostics']['phase_type'] in ('preinfusion', 'brew', 'decline')

    def test_per_phase_detailed_representative_samples(self):
        shot = self._standard_shot()
        t = transform_shot_for_ai(shot, detail="per_phase_detailed")
        # 5-sample extraction phase should have 5 evenly-spaced averaged samples
        ext_phase = [p for p in t['phases'] if p['name'] == 'Extraction'][0]
        assert len(ext_phase['samples']) == 5

    def test_per_phase_detailed_full_diagnostics(self):
        shot = self._standard_shot()
        t = transform_shot_for_ai(shot, detail="per_phase_detailed")
        diag = t['diagnostics']
        assert 'resistance' in diag
        assert 'channeling' in diag
        assert 'profile_compliance' in diag

    def test_summary_diagnostics_keys(self):
        shot = self._standard_shot()
        t = transform_shot_for_ai(shot, detail="summary")
        diag = t['diagnostics']
        assert 'resistance_avg' in diag
        assert 'channeling_risk' in diag
        assert 'temperature_stability_c' in diag
        assert 'pressure_rmse_bar' in diag
        assert 'max_overshoot_bar' in diag
        assert 'flow_rmse_ml_s' in diag
        assert 'max_flow_overshoot_ml_s' in diag
        assert 'scale_connected' in diag


class TestPhaseClassification:
    """Tests for phase name classification."""

    def test_preinfusion_exact_names(self):
        for name in ['Preinfusion', 'pre-infusion', 'PI', 'soak', 'Bloom', 'fill', 'Preinfuse']:
            assert _classify_phase(name) == 'preinfusion', f"Failed for: {name}"

    def test_preinfusion_substring_names(self):
        """Creative names containing preinfusion keywords should match."""
        for name in ['Gentle Pre-infusion', 'Blooming Phase', 'Long Soak', 'Quick Fill']:
            assert _classify_phase(name) == 'preinfusion', f"Failed for: {name}"

    def test_decline_exact_names(self):
        for name in ['Decline', 'taper', 'ramp-down', 'Ramp Down', 'cool down', 'cooldown']:
            assert _classify_phase(name) == 'decline', f"Failed for: {name}"

    def test_decline_substring_names(self):
        """Creative names containing decline keywords should match."""
        for name in ['Gentle Decline', 'Smooth Taper', 'Final Cooldown']:
            assert _classify_phase(name) == 'decline', f"Failed for: {name}"

    def test_brew_names(self):
        """Known brew-ish names fall through to brew without telemetry."""
        for name in ['Extraction', 'Brew', 'Main', 'Hold', 'flat', 'Step 2']:
            assert _classify_phase(name) == 'brew', f"Failed for: {name}"

    def test_unrecognised_name_defaults_to_brew(self):
        """Without telemetry samples, unrecognised names default to brew."""
        assert _classify_phase('My Custom Phase') == 'brew'
        assert _classify_phase('Phase 3') == 'brew'

    def test_telemetry_fallback_preinfusion(self):
        """First phase with low pressure and rising trend -> preinfusion."""
        samples = [
            {'cp': 1.0, 'pf': 0.5},
            {'cp': 2.0, 'pf': 1.0},
            {'cp': 3.0, 'pf': 1.5},
            {'cp': 4.0, 'pf': 2.0},
        ]
        result = _classify_phase(
            'My Custom Phase', phase_samples=samples,
            phase_index=0, total_phases=3,
        )
        assert result == 'preinfusion'

    def test_telemetry_fallback_decline(self):
        """Last phase with declining pressure -> decline."""
        samples = [
            {'cp': 8.0, 'pf': 2.0},
            {'cp': 7.0, 'pf': 2.1},
            {'cp': 6.0, 'pf': 2.2},
            {'cp': 4.5, 'pf': 2.3},
        ]
        result = _classify_phase(
            'Finish', phase_samples=samples,
            phase_index=2, total_phases=3,
        )
        assert result == 'decline'

    def test_telemetry_fallback_brew(self):
        """Middle phase with stable high pressure -> brew."""
        samples = [
            {'cp': 9.0, 'pf': 2.0},
            {'cp': 9.0, 'pf': 2.1},
            {'cp': 8.9, 'pf': 2.0},
            {'cp': 9.0, 'pf': 2.1},
        ]
        result = _classify_phase(
            'My Custom Phase', phase_samples=samples,
            phase_index=1, total_phases=3,
        )
        assert result == 'brew'

    def test_name_match_takes_priority_over_telemetry(self):
        """Even if telemetry looks like brew, name match wins."""
        # High pressure samples that look like brew
        samples = [
            {'cp': 9.0, 'pf': 2.0},
            {'cp': 9.0, 'pf': 2.1},
        ]
        result = _classify_phase(
            'Preinfusion', phase_samples=samples,
            phase_index=1, total_phases=3,
        )
        assert result == 'preinfusion'


class TestProfileCompliance:
    """Tests for profile compliance (RMSE and overshoot)."""

    def _make_shot(self, samples, phases=None, **kwargs):
        defaults = dict(
            id='000100', version=5, fields_mask=0xFF,
            sample_count=len(samples), sample_interval=100,
            profile_id='test', profile_name='Test Profile',
            timestamp=1640000000, rating=0, duration=30000,
            weight=40.0,
        )
        defaults.update(kwargs)
        return ShotData(samples=samples, phases=phases or [], **defaults)

    def test_compute_rmse_perfect(self):
        assert _compute_rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_compute_rmse_known(self):
        # RMSE of [1, 2, 3] vs [2, 3, 4] = sqrt((1+1+1)/3) = 1.0
        result = _compute_rmse([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        assert abs(result - 1.0) < 0.01

    def test_compute_rmse_empty(self):
        assert _compute_rmse([], []) == 0.0

    def test_profile_compliance_present_in_diagnostics(self):
        """Profile compliance computed when tp data available."""
        samples = [
            {'t': 0, 'ct': 93.0, 'tt': 93.0, 'cp': 3.0, 'tp': 3.0, 'pf': 0.5},
            {'t': 100, 'ct': 93.0, 'tt': 93.0, 'cp': 5.0, 'tp': 5.0, 'pf': 1.0},
            {'t': 200, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'tp': 9.0, 'pf': 2.0},
            {'t': 300, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'tp': 9.0, 'pf': 2.0},
            {'t': 400, 'ct': 93.0, 'tt': 93.0, 'cp': 8.5, 'tp': 9.0, 'pf': 2.0},
            {'t': 500, 'ct': 93.0, 'tt': 93.0, 'cp': 8.0, 'tp': 9.0, 'pf': 2.1},
        ]
        shot = self._make_shot(samples)
        diag = compute_shot_diagnostics(shot)
        assert diag is not None
        pc = diag['profile_compliance']
        assert pc is not None
        assert pc['pressure_rmse_bar'] >= 0
        assert 'pressure_adherence' in pc['annotations']
        assert 'pressure_overshoot' in pc['annotations']

    def test_profile_compliance_none_without_tp(self):
        """Profile compliance is None when no tp data."""
        samples = [
            {'t': 0, 'ct': 93.0, 'tt': 93.0, 'cp': 3.0, 'pf': 0.5},
            {'t': 100, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'pf': 2.0},
            {'t': 200, 'ct': 93.0, 'tt': 93.0, 'cp': 8.5, 'pf': 2.0},
            {'t': 300, 'ct': 93.0, 'tt': 93.0, 'cp': 8.0, 'pf': 2.0},
            {'t': 400, 'ct': 93.0, 'tt': 93.0, 'cp': 7.5, 'pf': 2.1},
        ]
        shot = self._make_shot(samples)
        diag = compute_shot_diagnostics(shot)
        assert diag is not None
        assert diag['profile_compliance'] is None

    def test_overshoot_detected(self):
        """Overshoot correctly detected when actual exceeds target."""
        samples = [
            {'t': 0, 'ct': 93.0, 'tt': 93.0, 'cp': 3.0, 'tp': 3.0, 'pf': 0.5},
            {'t': 100, 'ct': 93.0, 'tt': 93.0, 'cp': 11.0, 'tp': 9.0, 'pf': 1.0},  # +2 bar
            {'t': 200, 'ct': 93.0, 'tt': 93.0, 'cp': 10.5, 'tp': 9.0, 'pf': 2.0},  # +1.5 bar
            {'t': 300, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'tp': 9.0, 'pf': 2.0},
            {'t': 400, 'ct': 93.0, 'tt': 93.0, 'cp': 8.5, 'tp': 9.0, 'pf': 2.0},
            {'t': 500, 'ct': 93.0, 'tt': 93.0, 'cp': 8.0, 'tp': 9.0, 'pf': 2.1},
        ]
        shot = self._make_shot(samples)
        diag = compute_shot_diagnostics(shot)
        pc = diag['profile_compliance']
        assert pc['max_pressure_overshoot_bar'] == 2.0
        assert pc['annotations']['pressure_overshoot'] == 'SEVERE_OVERSHOOT'

    def test_flow_rmse_when_tf_available(self):
        """Flow RMSE computed when tf data available."""
        samples = [
            {'t': 0, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'tp': 9.0, 'pf': 2.0, 'tf': 2.0},
            {'t': 100, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'tp': 9.0, 'pf': 2.5, 'tf': 2.0},
            {'t': 200, 'ct': 93.0, 'tt': 93.0, 'cp': 8.5, 'tp': 9.0, 'pf': 2.0, 'tf': 2.0},
            {'t': 300, 'ct': 93.0, 'tt': 93.0, 'cp': 8.0, 'tp': 9.0, 'pf': 2.1, 'tf': 2.0},
            {'t': 400, 'ct': 93.0, 'tt': 93.0, 'cp': 7.5, 'tp': 9.0, 'pf': 2.2, 'tf': 2.0},
        ]
        shot = self._make_shot(samples)
        diag = compute_shot_diagnostics(shot)
        pc = diag['profile_compliance']
        assert pc['flow_rmse_ml_s'] is not None
        assert pc['flow_rmse_ml_s'] >= 0
        assert 'flow_adherence' in pc['annotations']

    def test_flow_overshoot_detected(self):
        """Flow overshoot correctly detected when actual exceeds target."""
        samples = [
            {'t': 0, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'tp': 9.0, 'pf': 2.0, 'tf': 1.0},
            {'t': 100, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'tp': 9.0, 'pf': 3.0, 'tf': 1.0},  # +2.0 ml/s
            {'t': 200, 'ct': 93.0, 'tt': 93.0, 'cp': 8.5, 'tp': 9.0, 'pf': 2.5, 'tf': 1.0},  # +1.5 ml/s
            {'t': 300, 'ct': 93.0, 'tt': 93.0, 'cp': 8.0, 'tp': 9.0, 'pf': 1.2, 'tf': 1.0},
            {'t': 400, 'ct': 93.0, 'tt': 93.0, 'cp': 7.5, 'tp': 9.0, 'pf': 1.0, 'tf': 1.0},
        ]
        shot = self._make_shot(samples)
        diag = compute_shot_diagnostics(shot)
        pc = diag['profile_compliance']
        assert pc['max_flow_overshoot_ml_s'] == 2.0
        assert pc['annotations']['flow_overshoot'] == 'SEVERE_DEVIATION'

    def test_flow_undershoot_detected(self):
        """Flow undershoot correctly detected when actual is below target."""
        samples = [
            {'t': 0, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'tp': 9.0, 'pf': 2.0, 'tf': 2.0},
            {'t': 100, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'tp': 9.0, 'pf': 1.0, 'tf': 2.0},  # -1.0 ml/s
            {'t': 200, 'ct': 93.0, 'tt': 93.0, 'cp': 8.5, 'tp': 9.0, 'pf': 1.2, 'tf': 2.0},  # -0.8 ml/s
            {'t': 300, 'ct': 93.0, 'tt': 93.0, 'cp': 8.0, 'tp': 9.0, 'pf': 1.8, 'tf': 2.0},
            {'t': 400, 'ct': 93.0, 'tt': 93.0, 'cp': 7.5, 'tp': 9.0, 'pf': 1.9, 'tf': 2.0},
        ]
        shot = self._make_shot(samples)
        diag = compute_shot_diagnostics(shot)
        pc = diag['profile_compliance']
        assert pc['max_flow_undershoot_ml_s'] == 1.0
        assert pc['annotations']['flow_undershoot'] == 'NOTABLE_DEVIATION'

    def test_flow_deviation_none_without_tf(self):
        """Flow overshoot/undershoot are None when no tf data."""
        samples = [
            {'t': 0, 'ct': 93.0, 'tt': 93.0, 'cp': 3.0, 'tp': 3.0, 'pf': 0.5},
            {'t': 100, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'tp': 9.0, 'pf': 2.0},
            {'t': 200, 'ct': 93.0, 'tt': 93.0, 'cp': 8.5, 'tp': 9.0, 'pf': 2.0},
            {'t': 300, 'ct': 93.0, 'tt': 93.0, 'cp': 8.0, 'tp': 9.0, 'pf': 2.0},
            {'t': 400, 'ct': 93.0, 'tt': 93.0, 'cp': 7.5, 'tp': 9.0, 'pf': 2.1},
        ]
        shot = self._make_shot(samples)
        diag = compute_shot_diagnostics(shot)
        pc = diag['profile_compliance']
        assert pc['max_flow_overshoot_ml_s'] is None
        assert pc['max_flow_undershoot_ml_s'] is None
        assert 'flow_overshoot' not in pc['annotations']
        assert 'flow_undershoot' not in pc['annotations']

    def test_flow_within_tolerance(self):
        """Small flow deviations annotated as WITHIN_TOLERANCE."""
        samples = [
            {'t': 0, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'tp': 9.0, 'pf': 2.1, 'tf': 2.0},
            {'t': 100, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'tp': 9.0, 'pf': 2.2, 'tf': 2.0},
            {'t': 200, 'ct': 93.0, 'tt': 93.0, 'cp': 8.5, 'tp': 9.0, 'pf': 1.9, 'tf': 2.0},
            {'t': 300, 'ct': 93.0, 'tt': 93.0, 'cp': 8.0, 'tp': 9.0, 'pf': 2.0, 'tf': 2.0},
            {'t': 400, 'ct': 93.0, 'tt': 93.0, 'cp': 7.5, 'tp': 9.0, 'pf': 2.05, 'tf': 2.0},
        ]
        shot = self._make_shot(samples)
        diag = compute_shot_diagnostics(shot)
        pc = diag['profile_compliance']
        assert pc['max_flow_overshoot_ml_s'] <= 0.3
        assert pc['annotations']['flow_overshoot'] == 'WITHIN_TOLERANCE'


class TestPerPhaseDiagnostics:
    """Tests for per-phase diagnostic computation."""

    def test_preinfusion_diagnostics(self):
        samples = [
            {'t': 0, 'cp': 0.5, 'pf': 0.0, 'tp': 3.0},
            {'t': 100, 'cp': 1.5, 'pf': 0.2, 'tp': 3.0},
            {'t': 200, 'cp': 2.5, 'pf': 0.4, 'tp': 3.0},
            {'t': 300, 'cp': 3.0, 'pf': 0.5, 'tp': 3.0},
        ]
        diag = _compute_phase_diagnostics(samples, "preinfusion", 0.1)
        assert diag['phase_type'] == 'preinfusion'
        assert 'ramp_rate_bar_s' in diag
        assert 'saturation_time_s' in diag
        assert diag['ramp_rate_bar_s'] > 0
        assert 'ramp_rate' in diag['annotations']

    def test_brew_diagnostics(self):
        samples = [
            {'t': 0, 'cp': 9.0, 'pf': 2.0, 'tp': 9.0},
            {'t': 100, 'cp': 9.0, 'pf': 2.0, 'tp': 9.0},
            {'t': 200, 'cp': 8.8, 'pf': 2.1, 'tp': 9.0},
            {'t': 300, 'cp': 8.5, 'pf': 2.1, 'tp': 9.0},
            {'t': 400, 'cp': 8.3, 'pf': 2.2, 'tp': 9.0},
        ]
        diag = _compute_phase_diagnostics(samples, "brew", 0.1)
        assert diag['phase_type'] == 'brew'
        assert 'resistance_avg' in diag
        assert 'resistance_slope' in diag
        assert 'channeling_risk' in diag
        assert 'flow_jitter_ml_s' in diag
        assert 'pressure_jitter_bar' in diag
        assert 'resistance_level' in diag['annotations']
        assert 'channeling' in diag['annotations']
        # Namespaced channeling annotations from the shared builder
        assert 'channeling_flow_jitter' in diag['annotations']
        assert 'channeling_guidance' in diag['annotations']

    def test_decline_diagnostics(self):
        samples = [
            {'t': 0, 'cp': 8.0, 'pf': 2.0, 'tp': 6.0},
            {'t': 100, 'cp': 7.0, 'pf': 2.2, 'tp': 5.0},
            {'t': 200, 'cp': 6.0, 'pf': 2.3, 'tp': 4.0},
            {'t': 300, 'cp': 5.0, 'pf': 2.5, 'tp': 3.0},
        ]
        diag = _compute_phase_diagnostics(samples, "decline", 0.1)
        assert diag['phase_type'] == 'decline'
        assert 'taper_rate_bar_s' in diag
        assert 'taper_smoothness' in diag
        assert diag['taper_rate_bar_s'] < 0  # Declining pressure
        assert 'taper_smoothness' in diag['annotations']

    def test_per_phase_rmse(self):
        """All phase types get RMSE vs target."""
        samples = [
            {'t': 0, 'cp': 9.0, 'pf': 2.0, 'tp': 9.0},
            {'t': 100, 'cp': 8.5, 'pf': 2.0, 'tp': 9.0},
            {'t': 200, 'cp': 8.0, 'pf': 2.1, 'tp': 9.0},
        ]
        for phase_type in ('preinfusion', 'brew', 'decline'):
            diag = _compute_phase_diagnostics(samples, phase_type, 0.1)
            assert 'pressure_rmse_bar' in diag
            assert 'flow_rmse_ml_s' in diag
            assert diag['pressure_rmse_bar'] >= 0


class TestRampExclusion:
    """Tests for ramp-up trimming and steady-state channeling assessment."""

    def test_trim_ramp_up_basic(self):
        """Ramp portion excluded when pressure climbs to target."""
        pressures = [1.0, 3.0, 5.0, 7.0, 8.5, 9.0, 9.0, 8.9, 9.0]
        flows = [0.1, 0.5, 1.0, 1.5, 1.9, 2.0, 2.0, 2.1, 2.0]
        samples = [{'cp': p, 'pf': f} for p, f in zip(pressures, flows)]
        ss_p, ss_f, ss_s = _trim_ramp_up(pressures, flows, samples)
        # Peak is 9.0, threshold = 8.1.  First sample >= 8.1 is index 4 (8.5)
        assert len(ss_p) == 5
        assert ss_p[0] == 8.5

    def test_trim_ramp_up_already_at_target(self):
        """No ramp to exclude — all samples already at target."""
        pressures = [9.0, 9.0, 8.9, 9.0, 8.8]
        flows = [2.0, 2.0, 2.1, 2.0, 2.1]
        samples = [{'cp': p, 'pf': f} for p, f in zip(pressures, flows)]
        ss_p, ss_f, _ = _trim_ramp_up(pressures, flows, samples)
        assert len(ss_p) == 5  # All returned

    def test_trim_ramp_up_empty(self):
        """Empty lists returned unchanged."""
        ss_p, ss_f, ss_s = _trim_ramp_up([], [], [])
        assert ss_p == []

    def test_insufficient_data_for_short_brew(self):
        """Short brew phase → INSUFFICIENT_DATA channeling risk."""
        # Only 3 brew samples at 9 bar — fewer than _MIN_STEADY_STATE_SAMPLES
        samples = [
            {'t': 0, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'pf': 2.0, 'tp': 9.0},
            {'t': 100, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'pf': 2.0, 'tp': 9.0},
            {'t': 200, 'ct': 93.0, 'tt': 93.0, 'cp': 8.8, 'pf': 2.1, 'tp': 9.0},
        ]
        diag = _compute_phase_diagnostics(samples, "brew", 0.1)
        assert diag['channeling_risk'] == 'INSUFFICIENT_DATA'
        assert 'channeling_note' in diag['annotations']

    def test_sufficient_data_gives_risk_label(self):
        """Brew phase with enough steady-state samples gets a real risk."""
        # 8 stable samples at ~9 bar
        samples = [
            {'t': i * 100, 'cp': 9.0 - i * 0.05, 'pf': 2.0, 'tp': 9.0}
            for i in range(8)
        ]
        diag = _compute_phase_diagnostics(samples, "brew", 0.1)
        assert diag['channeling_risk'] in ('LOW', 'MODERATE', 'HIGH', 'VERY_HIGH')


class TestCVNormalization:
    """Tests for coefficient-of-variation pressure volatility labelling."""

    def test_cv_used_at_high_pressure(self):
        """CV bands used when mean pressure >= 1.0 bar."""
        # std 0.1 at mean 9.0 → CV = 0.011 → VERY_STABLE
        assert _pressure_volatility_label(0.1, 9.0) == "VERY_STABLE"
        # std 0.5 at mean 9.0 → CV = 0.056 → MODERATE_JITTER
        assert _pressure_volatility_label(0.5, 9.0) == "MODERATE_JITTER"
        # std 2.0 at mean 9.0 → CV = 0.222 → VOLATILE
        assert _pressure_volatility_label(2.0, 9.0) == "VOLATILE"

    def test_absolute_fallback_at_low_pressure(self):
        """Absolute bands used when mean pressure < 1.0 bar."""
        # std 0.1 at mean 0.5 → absolute band → VERY_STABLE
        assert _pressure_volatility_label(0.1, 0.5) == "VERY_STABLE"
        # std 0.5 at mean 0.5 → absolute band → MODERATE_JITTER
        assert _pressure_volatility_label(0.5, 0.5) == "MODERATE_JITTER"

    def test_low_pressure_not_free_pass(self):
        """Low-pressure profiles don't automatically get VERY_STABLE."""
        # std 0.15 at mean 2.0 → CV = 0.075 → MODERATE_JITTER (not VERY_STABLE
        # as it would be under absolute bands where 0.15 < 0.35 → STABLE)
        assert _pressure_volatility_label(0.15, 2.0) == "MODERATE_JITTER"
        # std 0.4 at mean 2.0 → CV = 0.2 → VOLATILE
        assert _pressure_volatility_label(0.4, 2.0) == "VOLATILE"


class TestSampledDataPoints:
    """Tests for 5-point evenly-spaced averaged sampling."""

    def _make_shot(self, samples, phases=None, **kwargs):
        defaults = dict(
            id='000100', version=5, fields_mask=0xFF,
            sample_count=len(samples), sample_interval=100,
            profile_id='test', profile_name='Test Profile',
            timestamp=1640000000, rating=0, duration=30000,
            weight=40.0,
        )
        defaults.update(kwargs)
        return ShotData(samples=samples, phases=phases or [], **defaults)

    def test_per_phase_detailed_returns_5_samples_for_long_phase(self):
        """Phase with >5 samples returns exactly 5 evenly-spaced points."""
        # 20-sample single phase
        samples = [
            {'t': i * 100, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'pf': 2.0, 'tp': 9.0}
            for i in range(20)
        ]
        shot = self._make_shot(samples)
        t = transform_shot_for_ai(shot, detail="per_phase_detailed")
        phase = t['phases'][0]
        assert len(phase['samples']) == 5

    def test_per_phase_detailed_returns_all_for_short_phase(self):
        """Phase with <=5 samples returns all of them."""
        samples = [
            {'t': i * 100, 'ct': 93.0, 'tt': 93.0, 'cp': 9.0, 'pf': 2.0}
            for i in range(4)
        ]
        shot = self._make_shot(samples)
        t = transform_shot_for_ai(shot, detail="per_phase_detailed")
        phase = t['phases'][0]
        assert len(phase['samples']) == 4

    def test_averaged_samples_smooth_noise(self):
        """Sampled points use ±1 window averaging to smooth noise."""
        # Create samples with a spike at index 10
        samples = []
        for i in range(20):
            p = 9.0 if i != 10 else 12.0  # spike at index 10
            samples.append({'t': i * 100, 'ct': 93.0, 'cp': p, 'pf': 2.0})
        shot = self._make_shot(samples)
        t = transform_shot_for_ai(shot, detail="per_phase_detailed")
        phase_samples = t['phases'][0]['samples']
        # The middle sample (index 10 → anchor ~10) should be averaged:
        # (9.0 + 12.0 + 9.0) / 3 = 10.0, not 12.0
        mid = phase_samples[2]  # 3rd of 5 points → anchor at ~index 10
        assert mid['pressure_bar'] == 10.0


# ═══════════════════════════════════════════════════════════════════════
# Channeling indicators v2 — V4 (edge trim) + V5 (jitter) + V6 (residual)
# ═══════════════════════════════════════════════════════════════════════


class TestJitterStd:
    """_jitter_std measures first-difference std — noise around trend."""

    def test_flat_signal_has_zero_jitter(self):
        from gaggimate_mcp.transformers.shot import _jitter_std
        assert _jitter_std([4.0, 4.0, 4.0, 4.0, 4.0]) == 0.0

    def test_smooth_ramp_has_near_zero_jitter(self):
        """A linear ramp has identical first differences → std(diffs) ≈ 0."""
        from gaggimate_mcp.transformers.shot import _jitter_std
        ramp = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        assert _jitter_std(ramp) < 0.001

    def test_jittery_signal_has_high_jitter(self):
        """Alternating values produce large first differences."""
        from gaggimate_mcp.transformers.shot import _jitter_std
        jittery = [2.0, 4.0, 2.0, 4.0, 2.0, 4.0]
        assert _jitter_std(jittery) > 1.5

    def test_too_few_values_returns_zero(self):
        from gaggimate_mcp.transformers.shot import _jitter_std
        assert _jitter_std([]) == 0.0
        assert _jitter_std([1.0]) == 0.0
        assert _jitter_std([1.0, 2.0]) == 0.0


class TestResidualStdVsTarget:
    """_residual_std_vs_target measures how far actual flow strayed from target."""

    def test_perfect_tracking_returns_zero(self):
        from gaggimate_mcp.transformers.shot import _residual_std_vs_target
        samples = [{'pf': 2.0, 'tf': 2.0}] * 5
        assert _residual_std_vs_target(samples) == 0.0

    def test_systematic_constant_offset_returns_zero(self):
        """Systematic offset has zero std (no variation in residual)."""
        from gaggimate_mcp.transformers.shot import _residual_std_vs_target
        samples = [{'pf': 2.5, 'tf': 2.0}] * 5
        assert _residual_std_vs_target(samples) == 0.0

    def test_jitter_around_target_elevates_residual(self):
        from gaggimate_mcp.transformers.shot import _residual_std_vs_target
        samples = [
            {'pf': 1.5, 'tf': 2.0},
            {'pf': 2.5, 'tf': 2.0},
            {'pf': 1.5, 'tf': 2.0},
            {'pf': 2.5, 'tf': 2.0},
            {'pf': 1.5, 'tf': 2.0},
            {'pf': 2.5, 'tf': 2.0},
        ]
        # Residuals alternate ±0.5 → population std = 0.5
        assert abs(_residual_std_vs_target(samples) - 0.5) < 0.01

    def test_no_target_flow_returns_none(self):
        """Pure pressure-led profile — no tf field → not applicable."""
        from gaggimate_mcp.transformers.shot import _residual_std_vs_target
        samples = [{'pf': 2.0}] * 5
        assert _residual_std_vs_target(samples) is None

    def test_zero_target_samples_ignored(self):
        """Samples with tf=0 are not part of the commanded trajectory."""
        from gaggimate_mcp.transformers.shot import _residual_std_vs_target
        samples = [
            {'pf': 0.0, 'tf': 0.0},      # ignored
            {'pf': 2.0, 'tf': 2.0},
            {'pf': 2.0, 'tf': 2.0},
            {'pf': 2.0, 'tf': 2.0},
        ]
        # Only 3 valid pairs, all perfect → 0.0
        assert _residual_std_vs_target(samples) == 0.0

    def test_too_few_valid_pairs_returns_none(self):
        from gaggimate_mcp.transformers.shot import _residual_std_vs_target
        samples = [{'pf': 2.0, 'tf': 2.0}, {'pf': 2.0, 'tf': 0.0}]
        assert _residual_std_vs_target(samples) is None


class TestStripFlowEdges:
    """_strip_flow_edges removes leading/trailing near-zero flow samples."""

    def test_no_zero_edges_returned_unchanged(self):
        from gaggimate_mcp.transformers.shot import _strip_flow_edges
        p = [8.0, 8.5, 9.0, 8.5, 8.0]
        f = [2.0, 2.5, 3.0, 2.5, 2.0]
        s = [{'cp': pi, 'pf': fi} for pi, fi in zip(p, f)]
        rp, rf, rs, trimmed = _strip_flow_edges(p, f, s)
        assert rp == p
        assert rf == f
        assert rs == s
        assert trimmed == (0, 0)

    def test_strips_trailing_zero_flow_with_trapped_pressure(self):
        """Volumetric cutoff pattern: flow drops to 0 while pressure stays elevated."""
        from gaggimate_mcp.transformers.shot import _strip_flow_edges
        p = [8.0, 8.0, 8.0, 7.2, 7.2, 7.2]
        f = [3.0, 3.0, 3.0, 0.0, 0.0, 0.0]
        s = [{'cp': pi, 'pf': fi} for pi, fi in zip(p, f)]
        rp, rf, _, trimmed = _strip_flow_edges(p, f, s)
        assert rp == [8.0, 8.0, 8.0]
        assert rf == [3.0, 3.0, 3.0]
        assert trimmed == (0, 3)

    def test_strips_leading_zero_flow(self):
        """Valve-closed pattern: pressure ramping but flow hasn't opened yet."""
        from gaggimate_mcp.transformers.shot import _strip_flow_edges
        p = [8.0, 8.5, 8.8, 9.0, 9.0]
        f = [0.0, 0.0, 2.0, 2.5, 3.0]
        s = [{'cp': pi, 'pf': fi} for pi, fi in zip(p, f)]
        rp, rf, _, trimmed = _strip_flow_edges(p, f, s)
        assert rf == [2.0, 2.5, 3.0]
        assert rp == [8.8, 9.0, 9.0]
        assert trimmed == (2, 0)

    def test_strips_both_edges(self):
        from gaggimate_mcp.transformers.shot import _strip_flow_edges
        p = [7, 8, 9, 9, 9, 8, 7]
        f = [0.0, 0.0, 2.0, 3.0, 2.0, 0.0, 0.0]
        s = [{'cp': pi, 'pf': fi} for pi, fi in zip(p, f)]
        rp, rf, _, trimmed = _strip_flow_edges(p, f, s)
        assert rf == [2.0, 3.0, 2.0]
        assert rp == [9, 9, 9]
        assert trimmed == (2, 2)

    def test_all_zeros_returns_empty(self):
        """When every sample is below threshold, caller decides what to do."""
        from gaggimate_mcp.transformers.shot import _strip_flow_edges
        p = [5.0] * 5
        f = [0.0] * 5
        s = [{'cp': pi, 'pf': fi} for pi, fi in zip(p, f)]
        rp, rf, _, trimmed = _strip_flow_edges(p, f, s)
        assert rp == [] and rf == []
        assert trimmed == (5, 0)

    def test_threshold_is_configurable(self):
        from gaggimate_mcp.transformers.shot import _strip_flow_edges
        p = [8, 9, 9, 8]
        f = [0.2, 2.0, 2.0, 0.2]
        s = [{'cp': pi, 'pf': fi} for pi, fi in zip(p, f)]
        # Default threshold 0.1: nothing trimmed
        _, rf, _, t = _strip_flow_edges(p, f, s, thr=0.1)
        assert len(rf) == 4
        # Threshold 0.5: edges trimmed
        _, rf, _, t = _strip_flow_edges(p, f, s, thr=0.5)
        assert rf == [2.0, 2.0]
        assert t == (1, 1)


class TestFlowShapeLabel:
    """_flow_shape_label classifies the trajectory of flow over a window."""

    def test_flat_flow(self):
        from gaggimate_mcp.transformers.shot import _flow_shape_label
        assert _flow_shape_label([3.0, 3.0, 3.0, 3.0, 3.0], dt=0.25) == "FLAT"

    def test_ramping_up(self):
        from gaggimate_mcp.transformers.shot import _flow_shape_label
        flows = [1.0, 1.5, 2.0, 2.5, 3.0]  # +2 ml/s over 1s → 2 ml/s²
        assert _flow_shape_label(flows, dt=0.25) == "RAMPING_UP"

    def test_ramping_down(self):
        from gaggimate_mcp.transformers.shot import _flow_shape_label
        flows = [3.0, 2.5, 2.0, 1.5, 1.0]
        assert _flow_shape_label(flows, dt=0.25) == "RAMPING_DOWN"

    def test_insufficient_samples_returns_flat(self):
        from gaggimate_mcp.transformers.shot import _flow_shape_label
        assert _flow_shape_label([2.0], dt=0.25) == "FLAT"
        assert _flow_shape_label([], dt=0.25) == "FLAT"


class TestAssessChannelingRiskV2:
    """4-indicator scoring with flow_vs_target primary / pressure_jitter fallback."""

    def _risk(self, **kw):
        from gaggimate_mcp.transformers.shot import _assess_channeling_risk
        return _assess_channeling_risk(
            flow_jitter=kw.get('flow_jitter', 0.0),
            flow_vs_tgt=kw.get('flow_vs_tgt', 0.0),
            pressure_max_drop_rate=kw.get('pressure_max_drop_rate', 0.0),
            flow_acceleration_late=kw.get('flow_acceleration_late', 0.0),
            pressure_jitter=kw.get('pressure_jitter', 0.0),
        )

    # --- basic bands ---

    def test_all_zeros_is_low(self):
        assert self._risk() == "LOW"

    def test_flow_jitter_alone_moderate(self):
        """flow_jitter >= 0.10 adds 2 points → MODERATE."""
        assert self._risk(flow_jitter=0.15) == "MODERATE"

    def test_two_aligned_indicators_is_high(self):
        """flow_jitter (2) + pressure_cliff (2) = 4 → HIGH."""
        assert self._risk(flow_jitter=0.15, pressure_max_drop_rate=-3.5) == "HIGH"

    def test_all_indicators_blown_is_very_high(self):
        assert self._risk(
            flow_jitter=0.30,
            flow_vs_tgt=1.0,
            pressure_max_drop_rate=-4.0,
            flow_acceleration_late=0.15,
        ) == "VERY_HIGH"

    # --- fallback behavior ---

    def test_vs_target_none_falls_back_to_pressure_jitter(self):
        """When no target_flow commanded, pressure_jitter fills the indicator slot."""
        # flow_jitter 0.15 → +2, pressure_jitter 0.25 (>=0.20) → +2 = 4 → HIGH
        assert self._risk(
            flow_vs_tgt=None,
            flow_jitter=0.15,
            pressure_jitter=0.25,
        ) == "HIGH"

    def test_vs_target_present_ignores_pressure_jitter(self):
        """Flow-led profile: residual signal is used, not pressure_jitter."""
        # pressure_jitter irrelevant; flow_vs_tgt 0.0 + flow_jitter 0.15 → 2 → MODERATE
        assert self._risk(
            flow_vs_tgt=0.0,
            flow_jitter=0.15,
            pressure_jitter=0.5,  # would be alarming, but ignored
        ) == "MODERATE"

    # --- threshold boundaries ---

    def test_flow_jitter_boundary_stable(self):
        """Just below STABLE threshold → no contribution."""
        assert self._risk(flow_jitter=0.049) == "LOW"

    def test_flow_jitter_boundary_moderate_jitter(self):
        """At MODERATE_JITTER threshold → +1 point → still LOW (score=1)."""
        assert self._risk(flow_jitter=0.05) == "LOW"

    def test_flow_jitter_boundary_jittery(self):
        """At JITTERY threshold → +2 points → MODERATE."""
        assert self._risk(flow_jitter=0.10) == "MODERATE"


class TestLateFlowRunawayDetrended:
    """_late_flow_runaway must compare late slope against overall slope.

    Raw f_accel_late fires on any ramping-flow profile because the
    late-window slope equals the overall ramp.  The indicator should
    measure *excess* acceleration over what the profile designed for.
    """

    def test_flat_flow_no_runaway(self):
        from gaggimate_mcp.transformers.shot import _late_flow_runaway
        flows = [2.0] * 20
        assert abs(_late_flow_runaway(flows, dt=0.25)) < 0.005

    def test_linear_ramp_no_runaway(self):
        """A clean linear ramp has late_slope == overall_slope → 0 runaway."""
        from gaggimate_mcp.transformers.shot import _late_flow_runaway
        flows = [1.0 + 0.1 * i for i in range(20)]  # steady 0.1 per step
        assert abs(_late_flow_runaway(flows, dt=0.25)) < 0.01

    def test_late_acceleration_on_flat_profile(self):
        """Flat early, then kicks up → positive runaway."""
        from gaggimate_mcp.transformers.shot import _late_flow_runaway
        flows = [2.0] * 12 + [2.0, 2.3, 2.7, 3.2, 3.8, 4.5, 5.3, 6.2]
        result = _late_flow_runaway(flows, dt=0.25)
        assert result > 0.5

    def test_late_acceleration_exceeds_existing_ramp(self):
        """A ramping profile with an extra kick at the end registers runaway."""
        from gaggimate_mcp.transformers.shot import _late_flow_runaway
        # Overall slope ~0.1 per step; late window jumps to +0.5 per step
        flows = [1.0 + 0.1 * i for i in range(12)]
        flows += [flows[-1] + 0.5 * (i + 1) for i in range(8)]
        result = _late_flow_runaway(flows, dt=0.25)
        assert result > 0.5

    def test_too_few_samples_returns_zero(self):
        from gaggimate_mcp.transformers.shot import _late_flow_runaway
        assert _late_flow_runaway([1.0, 2.0, 3.0], dt=0.25) == 0.0


class TestChannelingRegressionFixtures:
    """Regression tests against real shot data.

    Shot 222 reproduced the 'Extraction Hold tail' false-positive that
    motivated this refactor.  204 and 196 caught secondary false-positives
    from flow-ramp profiles tripping late_flow_runaway.
    """

    def _run(self, path: str):
        from pathlib import Path
        from gaggimate_mcp.parsers.shot import parse_binary_shot
        from gaggimate_mcp.transformers.shot import compute_shot_diagnostics
        raw = Path(path).read_bytes()
        shot_id = Path(path).stem.split('_')[1]
        return compute_shot_diagnostics(parse_binary_shot(raw, f"000{shot_id}"))

    def test_shot_222_extraction_hold_tail_not_flagged(self):
        """Old pipeline: HIGH (f_vol=2.0 from trapped-pressure tail).
        New pipeline: LOW with stable flow jitter."""
        d = self._run('tests/fixtures/shot_222_hold_false_positive.slog')
        c = d['channeling']
        assert c['channeling_risk'] == "LOW"
        assert c['flow_jitter_ml_s'] < 0.025   # VERY_STABLE
        assert c['annotations']['flow_shape'] == "FLAT"
        # V4 trim must have stripped the problematic tail
        assert "trailing zero-flow" in c['annotations']['note']

    def test_shot_204_ramping_profile_not_flagged(self):
        """Flow ramping 1.6→3.3 on designed profile should not trip
        late_flow_runaway (V5b fix: detrended vs overall slope)."""
        d = self._run('tests/fixtures/shot_204_ramping_flow.slog')
        c = d['channeling']
        assert c['channeling_risk'] == "LOW"
        assert c['annotations']['flow_shape'] == "RAMPING_UP"
        # flow_spread will be elevated (it's raw std) but jitter stays low
        assert c['flow_jitter_ml_s'] < 0.05
        assert c['flow_spread_ml_s'] > 0.3  # intended ramp — descriptor only

    def test_shot_196_baseline_high_cleared(self):
        """Previously HIGH under baseline algorithm due to tail artifact."""
        d = self._run('tests/fixtures/shot_196_baseline_high.slog')
        c = d['channeling']
        assert c['channeling_risk'] == "LOW"
        assert c['flow_jitter_ml_s'] < 0.025


# ═══════════════════════════════════════════════════════════════════════
# Issue #18 — phase classification misses 'saturate'/'saturation'
# ═══════════════════════════════════════════════════════════════════════


class TestSaturatePhaseClassification:
    """Stock GaggiMate uses a pre-infusion phase literally named 'Saturate'.

    The keyword 'saturate' was missing from _PREINFUSION_KEYWORDS, so the
    phase was bucketed as brew.  Both 'saturate' and 'saturation' must now
    classify as preinfusion.  'saturate' is NOT a substring of 'saturation'
    (they differ at the 8th character), so both keywords must be present.
    """

    def test_saturate_exact_name_classifies_as_preinfusion(self):
        """`_classify_phase_by_name('Saturate')` must return 'preinfusion'."""
        assert _classify_phase_by_name('Saturate') == 'preinfusion'

    def test_saturation_exact_name_classifies_as_preinfusion(self):
        """`_classify_phase_by_name('Saturation')` must return 'preinfusion'."""
        assert _classify_phase_by_name('Saturation') == 'preinfusion'

    def test_saturate_substring_in_creative_name(self):
        """Creative names that contain 'saturate' should also match."""
        assert _classify_phase_by_name('Pre-Saturate') == 'preinfusion'
        assert _classify_phase_by_name('long saturation phase') == 'preinfusion'

    def test_saturate_keyword_in_preinfusion_keywords_tuple(self):
        """Both keywords must appear in the exported tuple."""
        assert 'saturate' in _PREINFUSION_KEYWORDS
        assert 'saturation' in _PREINFUSION_KEYWORDS

    def test_saturate_phase_excluded_from_brew_samples(self):
        """A phase named 'Saturate' must be skipped by _get_brew_phase_samples,
        so brew samples come only from the subsequent extraction phase."""
        samples = [
            # Saturate phase — should be excluded
            {'t': 0, 'ct': 90.0, 'cp': 2.0, 'pf': 0.3, 'phase': 0},
            {'t': 100, 'ct': 91.0, 'cp': 3.5, 'pf': 0.5, 'phase': 0},
            {'t': 200, 'ct': 92.0, 'cp': 4.0, 'pf': 0.6, 'phase': 0},
            # Extraction phase — should be included
            {'t': 300, 'ct': 93.0, 'cp': 9.0, 'pf': 2.0, 'phase': 1},
            {'t': 400, 'ct': 93.0, 'cp': 8.8, 'pf': 2.1, 'phase': 1},
            {'t': 500, 'ct': 93.0, 'cp': 8.5, 'pf': 2.1, 'phase': 1},
        ]
        shot = ShotData(
            id='001', version=5, fields_mask=0xFF, sample_count=6,
            sample_interval=100, profile_id='test', profile_name='Test',
            timestamp=1640000000, rating=0, duration=30000, weight=40.0,
            samples=samples,
            phases=[
                PhaseTransition(sample_index=0, phase_number=0, phase_name='Saturate'),
                PhaseTransition(sample_index=3, phase_number=1, phase_name='Extraction'),
            ],
        )
        brew = _get_brew_phase_samples(shot)
        assert len(brew) == 3, (
            f"Expected 3 brew samples (extraction only), got {len(brew)}; "
            "'Saturate' phase was not excluded"
        )
        assert brew[0]['cp'] == 9.0, "First brew sample should be from the Extraction phase"


# ═══════════════════════════════════════════════════════════════════════
# Issue #17 — summary resistance inflated by whole-brew window
# ═══════════════════════════════════════════════════════════════════════


class TestSummaryResistanceSteadyState:
    """compute_summary_diagnostics must compute resistance over the
    steady-state extraction window (after _trim_ramp_up + _strip_flow_edges)
    rather than the naive whole-brew window.

    A low-flow ramp at the start of the brew phase has very high P/flow²,
    inflating the summary resistance_avg if not trimmed.
    """

    def _make_shot(self, samples, phases=None, **kwargs):
        defaults = dict(
            id='000100', version=5, fields_mask=0xFF,
            sample_count=len(samples), sample_interval=100,
            profile_id='test', profile_name='Test Profile',
            timestamp=1640000000, rating=0, duration=30000,
            weight=40.0,
        )
        defaults.update(kwargs)
        return ShotData(samples=samples, phases=phases or [], **defaults)

    def test_ramp_inflates_naive_resistance_but_not_steady_state(self):
        """When the brew window opens with a pressure ramp at low flow,
        resistance_avg must be the steady-state value, not the inflated
        whole-window mean.

        Ramp samples (cp < 90 % of peak=9.0, i.e. < 8.1):
          cp=3.0, pf=0.5 → R = 3.0 / 0.25 = 12.0
          cp=5.0, pf=0.5 → R = 5.0 / 0.25 = 20.0
          cp=7.0, pf=0.5 → R = 7.0 / 0.25 = 28.0
        Steady-state samples (cp=9.0, pf=2.0) → R = 9.0 / 4.0 = 2.25 each

        Naive whole-window mean ≈ (12+20+28+2.25×8)/11 ≈ 7.09
        Steady-state mean = 2.25 — markedly lower.
        """
        preinfusion = [
            {'t': i * 100, 'ct': 91.0, 'cp': 1.0, 'pf': 0.1, 'phase': 0}
            for i in range(3)
        ]
        ramp = [
            {'t': (i + 3) * 100, 'ct': 93.0, 'cp': float(cp), 'pf': 0.5, 'phase': 1}
            for i, cp in enumerate([3.0, 5.0, 7.0])
        ]
        steady = [
            {'t': (i + 6) * 100, 'ct': 93.0, 'cp': 9.0, 'pf': 2.0, 'phase': 1}
            for i in range(8)
        ]
        samples = preinfusion + ramp + steady
        shot = self._make_shot(
            samples,
            phases=[
                PhaseTransition(sample_index=0, phase_number=0, phase_name='Preinfusion'),
                PhaseTransition(sample_index=3, phase_number=1, phase_name='Extraction'),
            ],
        )

        diag = compute_summary_diagnostics(shot)
        assert diag is not None

        # Compute the naive whole-window resistance for reference.
        # Brew samples are ramp (3) + steady (8) = 11 total; all pf > 0.1.
        naive_r_values = (
            [3.0 / 0.25, 5.0 / 0.25, 7.0 / 0.25]  # ramp
            + [9.0 / 4.0] * 8                        # steady
        )
        naive_mean = sum(naive_r_values) / len(naive_r_values)  # ≈ 7.09

        # Steady-state resistance (after trim) should equal 2.25.
        assert diag['resistance_avg'] == 2.25, (
            f"Expected steady-state resistance_avg=2.25, got {diag['resistance_avg']}"
        )
        # And it must be markedly lower than the naive whole-window figure.
        assert diag['resistance_avg'] < naive_mean * 0.5, (
            f"resistance_avg ({diag['resistance_avg']}) is not markedly lower than "
            f"naive whole-window mean ({naive_mean:.2f})"
        )

    def test_fallback_when_trimmed_window_too_short(self):
        """When the steady-state window has fewer than _MIN_STEADY_STATE_SAMPLES
        samples, resistance must fall back to the whole brew window and still
        return a finite, positive value (not None or NaN).

        This shot has only 2 steady samples after trim, which is below the
        threshold of 5, so the fallback activates.
        """
        from gaggimate_mcp.transformers.shot import _MIN_STEADY_STATE_SAMPLES

        # Construct a brew window: 4 low-pressure ramp + 2 steady samples
        # _trim_ramp_up keeps only samples with cp >= 90%*9 = 8.1 → just 2
        ramp = [
            {'t': i * 100, 'ct': 93.0, 'cp': float(cp), 'pf': 0.5, 'phase': 0}
            for i, cp in enumerate([3.0, 5.0, 6.0, 7.0])
        ]
        steady = [
            {'t': (i + 4) * 100, 'ct': 93.0, 'cp': 9.0, 'pf': 2.0, 'phase': 0}
            for i in range(2)
        ]
        samples = ramp + steady
        assert 2 < _MIN_STEADY_STATE_SAMPLES, "test premise: trimmed window is too short"

        shot = self._make_shot(samples)

        diag = compute_summary_diagnostics(shot)
        assert diag is not None

        # Fallback to whole-brew resistance — must be a sensible positive float.
        assert diag['resistance_avg'] is not None
        assert diag['resistance_avg'] > 0, (
            f"Expected positive resistance_avg from fallback, got {diag['resistance_avg']}"
        )
        import math
        assert not math.isnan(diag['resistance_avg']), "resistance_avg must not be NaN"

        # Compute the brew samples as _get_brew_phase_samples would (no phases →
        # pressure-threshold fallback: keep samples with cp >= 50% of peak=9.0).
        brew = [s for s in samples if s['cp'] >= 0.5 * 9.0]
        all_r = [s['cp'] / (s['pf'] ** 2) for s in brew if s['pf'] > 0.1]
        expected = round(sum(all_r) / len(all_r), 2)
        assert diag['resistance_avg'] == expected, (
            f"Fallback resistance_avg ({diag['resistance_avg']}) does not match "
            f"whole-window mean ({expected})"
        )
