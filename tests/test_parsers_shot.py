"""Tests for .slog binary shot file parser."""

import struct
from pathlib import Path

from gaggimate_mcp.parsers.shot import parse_binary_shot


# Bit -> field name, mirroring shot_log_format.h. This is the on-disk spec,
# not the (buggy) implementation under test, so tests may hardcode it.
_BIT_NAMES = {
    0: 't', 1: 'tt', 2: 'ct', 3: 'tp', 4: 'cp', 5: 'fl', 6: 'tf', 7: 'pf',
    8: 'vf', 9: 'v', 10: 'ev', 11: 'pr', 12: 'si', 13: 'wp', 14: 'unknown14',
}
_SIGNED_FIELDS = {'fl', 'tf', 'pf', 'vf'}


def _encode_sample(fields_mask: int, values: dict, elapsed_ms: bool) -> bytes:
    """Encode one sample record per shot_log_format.h field order."""
    parts = []
    for bit in range(32):
        if not (fields_mask & (1 << bit)):
            continue
        name = _BIT_NAMES[bit]
        value = values[name]
        if bit == 0 and elapsed_ms:
            parts.append(struct.pack('<I', value))
        elif name in _SIGNED_FIELDS:
            parts.append(struct.pack('<h', value))
        else:
            parts.append(struct.pack('<H', value))
    return b''.join(parts)


def _encode_header(
    *, version: int, reserved0: int, header_size: int, sample_interval: int,
    fields_mask: int, sample_count: int, duration: int, timestamp: int,
    profile_id: str = 'test', profile_name: str = 'Test', weight: float = 0.0,
    phase_transitions: tuple = (), final_exit_reason: int = 0, brew_delay_ms: int = 0,
) -> bytes:
    """Encode a v1-v8 .slog header per shot_log_format.h."""
    magic = 0x544F4853
    profile_id_bytes = profile_id.encode().ljust(32, b'\x00')[:32]
    profile_name_bytes = profile_name.encode().ljust(48, b'\x00')[:48]
    weight_raw = int(round(weight * 10))

    part1 = struct.pack(
        '<IBBHHHIIII32s48sH',
        magic, version, reserved0, header_size, sample_interval, 0,
        fields_mask, sample_count, duration, timestamp,
        profile_id_bytes, profile_name_bytes, weight_raw,
    )

    if header_size == 128:
        return part1 + b'\x00' * (128 - len(part1))

    transitions_bytes = bytearray()
    for sample_index, phase_number, reason, name in phase_transitions:
        name_bytes = name.encode().ljust(25, b'\x00')[:25]
        transitions_bytes += struct.pack('<HBB25s', sample_index, phase_number, reason, name_bytes)
    transitions_bytes += b'\x00' * (12 * 29 - len(transitions_bytes))

    tail = struct.pack('<BBH', len(phase_transitions), final_exit_reason, brew_delay_ms)
    header = part1 + bytes(transitions_bytes) + tail
    return header + b'\x00' * (512 - len(header))


class TestBinaryShotParser:
    """Test binary shot file parsing."""

    def test_parse_minimal_shot_v4(self):
        """Test parsing a minimal V4 shot file."""
        # Create minimal valid V4 shot file
        magic = 0x544F4853  # 'SHOT'
        version = 4
        reserved0 = 0
        header_size = 128
        sample_interval = 100  # milliseconds
        reserved1 = 0
        fields_mask = 0b1111  # T, TT, CT, TP (4 fields)
        sample_count = 2
        duration = 20000  # 20 seconds
        timestamp = 1640000000
        profile_id = b'test_profile\x00' + b'\x00' * 19
        profile_name = b'Test Profile\x00' + b'\x00' * 35
        weight = 360  # 36.0 grams

        # Build header
        header = struct.pack(
            '<IB B H H H I I I I 32s 48s H',
            magic, version, reserved0, header_size,
            sample_interval, reserved1,
            fields_mask, sample_count, duration, timestamp,
            profile_id, profile_name, weight
        )

        # Pad header to 128 bytes
        header = header + b'\x00' * (128 - len(header))

        # Sample data: 2 samples, 4 fields each (8 bytes per sample)
        # Sample 1: t=1000ms, tt=93.0°C, ct=92.5°C, tp=9.0 bar
        sample1 = struct.pack('<HHHH', 10, 930, 925, 90)

        # Sample 2: t=2000ms, tt=93.0°C, ct=93.0°C, tp=9.0 bar
        sample2 = struct.pack('<HHHH', 20, 930, 930, 90)

        binary_data = header + sample1 + sample2

        # Parse
        shot = parse_binary_shot(binary_data, "000001")

        # Verify
        assert shot.id == "000001"
        assert shot.version == 4
        assert shot.fields_mask == 0b1111
        assert shot.sample_count == 2
        assert shot.sample_interval == 100
        assert shot.profile_id == "test_profile"
        assert shot.profile_name == "Test Profile"
        assert shot.timestamp == 1640000000
        assert shot.duration == 20000
        assert shot.weight == 36.0
        assert shot.incomplete == False
        assert len(shot.samples) == 2
        assert len(shot.phases) == 0  # V4 has no phase data

        # Check first sample
        assert shot.samples[0]['t'] == 1000  # 10 * 100ms
        assert shot.samples[0]['tt'] == 93.0  # 930 / 10
        assert shot.samples[0]['ct'] == 92.5  # 925 / 10
        assert shot.samples[0]['tp'] == 9.0  # 90 / 10

    def test_parse_shot_invalid_magic(self):
        """Test parsing shot file with invalid magic number."""
        invalid_data = struct.pack('<I', 0xDEADBEEF) + b'\x00' * 124

        try:
            parse_binary_shot(invalid_data, "000001")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid shot magic" in str(e)

    def test_parse_shot_html_response_gives_friendly_error(self):
        """Test that HTML response (firmware 1.8.0 issue) gives a clear error."""
        html_data = b'<!DOCTYPE html><html><body>Not Found</body></html>'
        html_data = html_data + b'\x00' * (128 - len(html_data))

        try:
            parse_binary_shot(html_data, "000195")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "HTML" in str(e)
            assert "overloaded" in str(e)

    def test_parse_shot_short_html_detected_before_size_check(self):
        """Short HTML responses should get HTML error, not 'too small' error."""
        short_html = b'<html>err</html>'  # < 128 bytes
        try:
            parse_binary_shot(short_html, "000195")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "HTML" in str(e)
            assert "too small" not in str(e)

    def test_parse_shot_firmware_1_8_0_magic_bytes(self):
        """Test the specific magic bytes from the 1.8.0 error report."""
        # 0x6f64213c = "<!do" in little-endian ASCII
        data = b'<!doctype html>' + b'\x00' * 113

        try:
            parse_binary_shot(data, "000195")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "HTML" in str(e)

    def test_parse_shot_too_small(self):
        """Test parsing shot file that's too small."""
        too_small = b'\x00' * 50

        try:
            parse_binary_shot(too_small, "000001")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "too small" in str(e)

    def test_parse_shot_with_phase_transitions(self):
        """Test parsing V5 shot file with phase transitions."""
        magic = 0x544F4853
        version = 5
        reserved0 = 0
        header_size = 512
        sample_interval = 100
        reserved1 = 0
        fields_mask = 0b11  # T, TT (2 fields)
        sample_count = 3
        duration = 30000
        timestamp = 1640000000
        profile_id = b'v5_profile\x00' + b'\x00' * 21
        profile_name = b'V5 Profile\x00' + b'\x00' * 37
        weight = 400  # 40.0 grams

        # Build header up to weight (110 bytes)
        header_part1 = struct.pack(
            '<IB B H H H I I I I 32s 48s H',
            magic, version, reserved0, header_size,
            sample_interval, reserved1,
            fields_mask, sample_count, duration, timestamp,
            profile_id, profile_name, weight
        )

        # Phase transitions: 2 transitions
        # Transition 1: sample 0, phase 0, "Preinfusion"
        transition1 = struct.pack('<HB x 25s', 0, 0, b'Preinfusion\x00' + b'\x00' * 13)

        # Transition 2: sample 2, phase 1, "Extraction"
        transition2 = struct.pack('<HB x 25s', 2, 1, b'Extraction\x00' + b'\x00' * 14)

        # Pad to transition count position (458 bytes)
        padding = b'\x00' * (458 - len(header_part1) - len(transition1) - len(transition2))

        # Transition count
        transition_count = struct.pack('B', 2)

        # Pad to 512 bytes
        header = header_part1 + transition1 + transition2 + padding + transition_count
        header = header + b'\x00' * (512 - len(header))

        # Sample data: 3 samples, 2 fields each
        sample1 = struct.pack('<HH', 10, 900)  # t=1000ms, tt=90.0°C
        sample2 = struct.pack('<HH', 20, 920)  # t=2000ms, tt=92.0°C
        sample3 = struct.pack('<HH', 30, 930)  # t=3000ms, tt=93.0°C

        binary_data = header + sample1 + sample2 + sample3

        # Parse
        shot = parse_binary_shot(binary_data, "000002")

        # Verify
        assert shot.version == 5
        assert len(shot.phases) == 2
        assert shot.phases[0].sample_index == 0
        assert shot.phases[0].phase_number == 0
        assert shot.phases[0].phase_name == "Preinfusion"
        assert shot.phases[1].sample_index == 2
        assert shot.phases[1].phase_number == 1
        assert shot.phases[1].phase_name == "Extraction"

        # Check phase numbers in samples
        assert shot.samples[0].get('phase') == 0
        assert shot.samples[1].get('phase') == 0
        assert shot.samples[2].get('phase') == 1

    def test_parse_incomplete_shot(self):
        """Test parsing a shot file with incomplete samples."""
        magic = 0x544F4853
        version = 4
        reserved0 = 0
        header_size = 128
        sample_interval = 100
        reserved1 = 0
        fields_mask = 0b11  # T, TT (2 fields)
        sample_count = 10  # Claims 10 samples
        duration = 10000
        timestamp = 1640000000
        profile_id = b'incomplete\x00' + b'\x00' * 21
        profile_name = b'Incomplete\x00' + b'\x00' * 37
        weight = 0

        header = struct.pack(
            '<IB B H H H I I I I 32s 48s H',
            magic, version, reserved0, header_size,
            sample_interval, reserved1,
            fields_mask, sample_count, duration, timestamp,
            profile_id, profile_name, weight
        )
        header = header + b'\x00' * (128 - len(header))

        # Only provide 2 samples worth of data (but claims 10)
        sample1 = struct.pack('<HH', 10, 900)
        sample2 = struct.pack('<HH', 20, 920)

        binary_data = header + sample1 + sample2

        # Parse
        shot = parse_binary_shot(binary_data, "000003")

        # Should only parse the 2 samples that exist
        assert shot.sample_count == 2
        assert shot.incomplete == True
        assert len(shot.samples) == 2


class TestFirmwareV6V7V8ShotParser:
    """Regression tests for GaggiMate firmware v1.9.0 (shot log v6/v7/v8).

    Firmware v1.9.0 widened the first sample field (`t`) from uint16 to
    uint32 and added a `wp` (water pumped) field, growing the per-sample
    record from 26 to 30 bytes. The parser used to compute the stride as
    ``popcount(fields_mask) * 2`` and ignore the on-disk record size
    (header byte 5), so every v6+ file was decoded 2-4 bytes misaligned —
    producing temperatures in the thousands, pressures to match, and
    phase timestamps in the thousands of seconds.
    """

    FIELDS_MASK_V7 = 0x3FFF  # 14 fields: T..SI (0-12) + WP (13)
    FIELDS_MASK_LEGACY = 0x1FFF  # 13 fields: T..SI (0-12), no WP

    def _v7_sample_values(self, i: int) -> dict:
        return {
            't': i * 251,
            'tt': 930, 'ct': 934 + (i % 3), 'tp': 90, 'cp': 89 - (i % 2),
            'fl': 210, 'tf': 200, 'pf': 190, 'vf': 185,
            'v': 281, 'ev': 275, 'pr': 150, 'si': 0x0024, 'wp': 1200 + i * 42,
        }

    def test_v7_shot_decodes_plausible_values_not_thousands(self):
        """The exact bug: a v7 file must no longer decode into the 1000s."""
        n = 30
        header = _encode_header(
            version=7, reserved0=30, header_size=512, sample_interval=250,
            fields_mask=self.FIELDS_MASK_V7, sample_count=n, duration=n * 251,
            timestamp=1775112285, profile_name='Adaptive v2', weight=28.1,
            phase_transitions=[(0, 0, 0, 'Pre-infusion'), (12, 1, 5, 'Extraction')],
        )
        samples = b''.join(
            _encode_sample(self.FIELDS_MASK_V7, self._v7_sample_values(i), elapsed_ms=True)
            for i in range(n)
        )

        shot = parse_binary_shot(header + samples, "000493")

        assert shot.version == 7
        assert len(shot.samples) == n
        assert shot.incomplete is False

        ct_values = [s['ct'] for s in shot.samples]
        cp_values = [s['cp'] for s in shot.samples]
        v_values = [s['v'] for s in shot.samples]
        pf_values = [s['pf'] for s in shot.samples]
        t_values = [s['t'] for s in shot.samples]

        # These are the exact symptoms reported against firmware v1.9.0:
        # thousands-of-degrees temperatures, pressure == temperature + 50.0,
        # a scale reading over a kilogram, negative flow, phase times in
        # the thousands of seconds.
        assert max(ct_values) < 150.0
        assert max(cp_values) < 20.0
        assert max(cp_values) - max(ct_values) != 50.0
        assert max(v_values) < 500.0
        assert min(pf_values) >= 0.0
        assert max(t_values) < shot.duration * 1.5

        # Values should match what was encoded, not a shifted column.
        assert ct_values[0] == 93.4
        assert cp_values[0] == 8.9
        assert v_values[0] == 28.1
        assert t_values[0] == 0
        assert t_values[1] == 251

    def test_v7_wp_field_is_decoded(self):
        """wp (cumulative water pumped) is new in v7 and must be exposed."""
        header = _encode_header(
            version=7, reserved0=30, header_size=512, sample_interval=250,
            fields_mask=self.FIELDS_MASK_V7, sample_count=2, duration=502,
            timestamp=1775112285,
        )
        samples = b''.join(
            _encode_sample(self.FIELDS_MASK_V7, self._v7_sample_values(i), elapsed_ms=True)
            for i in range(2)
        )

        shot = parse_binary_shot(header + samples, "000001")

        assert shot.samples[0]['wp'] == 120.0  # 1200 / WEIGHT_SCALE(10)
        assert shot.samples[1]['wp'] == 124.2  # 1242 / 10

    def test_v7_phase_transition_reason_and_final_exit_reason(self):
        """transition_reason (per phase) and final_exit_reason are new in v7."""
        header = _encode_header(
            version=7, reserved0=30, header_size=512, sample_interval=250,
            fields_mask=self.FIELDS_MASK_V7, sample_count=2, duration=502,
            timestamp=1775112285,
            phase_transitions=[(0, 0, 0, 'Pre-infusion'), (1, 1, 5, 'Extraction')],
            final_exit_reason=1, brew_delay_ms=500,
        )
        samples = b''.join(
            _encode_sample(self.FIELDS_MASK_V7, self._v7_sample_values(i), elapsed_ms=True)
            for i in range(2)
        )

        shot = parse_binary_shot(header + samples, "000001")

        assert shot.phases[0].transition_reason == 0
        assert shot.phases[1].transition_reason == 5
        assert shot.phases[1].transition_reason_label == "Duration"
        assert shot.final_exit_reason == 1
        assert shot.final_exit_reason_label == "Volumetric target"
        assert shot.brew_delay_ms == 500

    def test_v6_upstream_elapsed_ms_28_byte_record(self):
        """Upstream v6: 28-byte record, no wp, but t is already uint32 ms."""
        mask = self.FIELDS_MASK_LEGACY  # 13 fields, no WP -> 26 + 2(wide t) = 28
        header = _encode_header(
            version=6, reserved0=28, header_size=512, sample_interval=250,
            fields_mask=mask, sample_count=3, duration=753, timestamp=1775112285,
        )
        values = lambda i: {
            't': i * 251, 'tt': 930, 'ct': 934, 'tp': 90, 'cp': 89,
            'fl': 210, 'tf': 200, 'pf': 190, 'vf': 185,
            'v': 281, 'ev': 275, 'pr': 150, 'si': 0x0004,
        }
        samples = b''.join(_encode_sample(mask, values(i), elapsed_ms=True) for i in range(3))

        shot = parse_binary_shot(header + samples, "000001")

        assert shot.version == 6
        assert shot.samples[0]['t'] == 0
        assert shot.samples[1]['t'] == 251
        assert shot.samples[0]['ct'] == 93.4

    def test_v6_legacy_hardware_scale_26_byte_record(self):
        """Legacy hardware-scale v6 fork: 26-byte record, t is still a tick."""
        mask = self.FIELDS_MASK_LEGACY
        sample_interval = 250
        header = _encode_header(
            version=6, reserved0=26, header_size=512, sample_interval=sample_interval,
            fields_mask=mask, sample_count=3, duration=750, timestamp=1775112285,
        )
        values = lambda tick: {
            't': tick, 'tt': 930, 'ct': 934, 'tp': 90, 'cp': 89,
            'fl': 210, 'tf': 200, 'pf': 190, 'vf': 185,
            'v': 281, 'ev': 275, 'pr': 150, 'si': 0x0004,
        }
        samples = b''.join(_encode_sample(mask, values(tick), elapsed_ms=False) for tick in (0, 1, 2))

        shot = parse_binary_shot(header + samples, "000001")

        assert shot.version == 6
        # t = tick * sample_interval, the legacy transform -- NOT raw ms.
        assert shot.samples[0]['t'] == 0
        assert shot.samples[1]['t'] == sample_interval
        assert shot.samples[2]['t'] == sample_interval * 2

    def test_v8_new_system_info_bits_and_exit_reasons(self):
        """v8 adds si bits 0x20-0x100 and exit reasons 8 (hold released) and 9 (ratio)."""
        header = _encode_header(
            version=8, reserved0=30, header_size=512, sample_interval=250,
            fields_mask=self.FIELDS_MASK_V7, sample_count=1, duration=251,
            timestamp=1775112285,
            phase_transitions=[(0, 0, 8, 'Hold')], final_exit_reason=9,
        )
        values = {
            't': 0, 'tt': 930, 'ct': 934, 'tp': 90, 'cp': 89,
            'fl': 210, 'tf': 200, 'pf': 190, 'vf': 185,
            'v': 281, 'ev': 275, 'pr': 150, 'si': 0x0124,  # BT scale + active scale connected
            'wp': 1200,
        }
        samples = _encode_sample(self.FIELDS_MASK_V7, values, elapsed_ms=True)

        shot = parse_binary_shot(header + samples, "000001")

        si = shot.samples[0]['systemInfo']
        assert si['bluetooth_scale_connected'] is True
        assert si['active_scale_connected'] is True
        assert shot.phases[0].transition_reason_label == "Hold released"
        assert shot.final_exit_reason_label == "Ratio target"

    def test_v7_active_scale_connected_mirrors_bluetooth_bit(self):
        """v6/v7 have no source-neutral scale bit -- mirror BT bit per upstream."""
        header = _encode_header(
            version=7, reserved0=30, header_size=512, sample_interval=250,
            fields_mask=self.FIELDS_MASK_V7, sample_count=1, duration=251,
            timestamp=1775112285,
        )
        values = {
            't': 0, 'tt': 930, 'ct': 934, 'tp': 90, 'cp': 89,
            'fl': 210, 'tf': 200, 'pf': 190, 'vf': 185,
            'v': 281, 'ev': 275, 'pr': 150, 'si': 0x0004,  # BT scale connected, no 0x100 bit
            'wp': 1200,
        }
        samples = _encode_sample(self.FIELDS_MASK_V7, values, elapsed_ms=True)

        shot = parse_binary_shot(header + samples, "000001")

        si = shot.samples[0]['systemInfo']
        assert si['bluetooth_scale_connected'] is True
        assert si['active_scale_connected'] is True  # mirrored, not left False

    def test_unknown_future_version_decodes_known_fields_and_warns(self):
        """A future v9 with an extra unknown field must degrade gracefully."""
        mask = self.FIELDS_MASK_V7 | (1 << 14)  # 15 fields: known 14 + 1 unknown
        header = _encode_header(
            version=9, reserved0=32, header_size=512, sample_interval=250,
            fields_mask=mask, sample_count=2, duration=502, timestamp=1775112285,
        )
        values = lambda i: {**self._v7_sample_values(i), 'unknown14': 0xBEEF}
        samples = b''.join(_encode_sample(mask, values(i), elapsed_ms=True) for i in range(2))

        shot = parse_binary_shot(header + samples, "000001")

        assert shot.version == 9
        assert shot.samples[0]['ct'] == 93.4
        assert shot.samples[0]['cp'] == 8.9
        assert 'unknown14' not in shot.samples[0]
        assert any("version 9" in w for w in shot.warnings)

    def test_reserved0_mismatch_is_trusted_and_warned(self):
        """When the device-declared size disagrees with the field mask,
        trust the device (it knows its own layout) but surface a warning."""
        header = _encode_header(
            version=7, reserved0=34, header_size=512, sample_interval=250,
            fields_mask=self.FIELDS_MASK_V7, sample_count=2, duration=502,
            timestamp=1775112285,
        )
        # Pad each sample to 34 bytes (30 known + 4 trailing unknown bytes)
        values = {**self._v7_sample_values(0), 't': 0}
        samples = b''.join(
            _encode_sample(self.FIELDS_MASK_V7, {**values, 't': i * 251}, elapsed_ms=True) + b'\x00\x00\x00\x00'
            for i in range(2)
        )

        shot = parse_binary_shot(header + samples, "000001")

        assert shot.samples[0]['ct'] == 93.4
        assert shot.samples[1]['ct'] == 93.4
        assert any("mismatch" in w.lower() for w in shot.warnings)

    def test_real_v5_fixtures_still_decode_within_plausible_ranges(self):
        """Guard against regressions on the three real-device v5 fixtures."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        for path in sorted(fixtures_dir.glob("*.slog")):
            shot = parse_binary_shot(path.read_bytes(), path.stem)
            assert shot.version == 5
            ct_values = [s['ct'] for s in shot.samples if 'ct' in s]
            cp_values = [s['cp'] for s in shot.samples if 'cp' in s]
            assert max(ct_values) < 150.0, f"{path.name}: implausible ct"
            assert max(cp_values) < 20.0, f"{path.name}: implausible cp"
            assert shot.warnings == []
