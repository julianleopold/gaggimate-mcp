"""Tests for .slog binary shot file parser."""

import struct
from gaggimate_mcp.parsers.shot import parse_binary_shot


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


def _build_v5plus_header(version, reserved0, fields_mask, sample_count, duration=40000):
    """Build a 512-byte v5+ header with no phase transitions."""
    header = struct.pack(
        '<IB B H H H I I I I 32s 48s H',
        0x544F4853, version, reserved0, 512,
        250, 0,
        fields_mask, sample_count, duration, 1792181285,
        b'prof\x00'.ljust(32, b'\x00'), b'Profile\x00'.ljust(48, b'\x00'), 433,
    )
    return header + b'\x00' * (512 - len(header))


def _full_sample_v7(t_ms, ct, cp, wp):
    """One 30-byte v7/v8 sample: uint32 t followed by 13 uint16/int16 fields."""
    return struct.pack(
        '<I HHHH hhhh HHHHH',
        t_ms,
        960, int(ct * 10),            # tt, ct
        90, int(cp * 10),             # tp, cp
        250, 0, 180, 0,               # fl, tf, pf, vf
        0, 215, 1234,                 # v, ev, pr
        0x0020,                       # si (process is brew)
        int(wp * 10),                 # wp
    )


class TestBinaryShotParserV6Plus:
    """Firmware 1.9.0+ logs (v6 uint32 timestamps, v7/v8 water-pumped field)."""

    def test_parse_v7_full_mask(self):
        mask = 0x3FFF
        samples = _full_sample_v7(250, 95.4, 1.2, 3.5) + _full_sample_v7(30217, 89.1, 9.0, 61.0)
        data = _build_v5plus_header(7, 30, mask, 2) + samples

        shot = parse_binary_shot(data, "000112")

        assert shot.version == 7
        assert shot.sample_count == 2
        s0, s1 = shot.samples
        assert s0['t'] == 250
        assert s1['t'] == 30217  # real ms, not index * interval
        assert s0['tt'] == 96.0
        assert s0['ct'] == 95.4
        assert s1['cp'] == 9.0
        assert s1['pf'] == 1.8
        assert s1['ev'] == 21.5
        assert s1['pr'] == 12.34
        assert s1['wp'] == 61.0
        assert s1['systemInfo']['raw'] == 0x0020

    def test_parse_v8_same_layout_as_v7(self):
        data = _build_v5plus_header(8, 30, 0x3FFF, 1) + _full_sample_v7(1000, 93.0, 5.0, 10.0)
        shot = parse_binary_shot(data, "000200")
        assert shot.samples[0]['t'] == 1000
        assert shot.samples[0]['ct'] == 93.0
        assert shot.samples[0]['wp'] == 10.0

    def test_parse_v6_uint32_timestamp(self):
        mask = 0x1FFF  # no wp field
        sample = _full_sample_v7(5000, 92.0, 8.5, 0)[:-2]  # drop wp -> 28 bytes
        data = _build_v5plus_header(6, 28, mask, 1) + sample
        shot = parse_binary_shot(data, "000150")
        assert shot.samples[0]['t'] == 5000
        assert shot.samples[0]['ct'] == 92.0
        assert shot.samples[0]['cp'] == 8.5
        assert 'wp' not in shot.samples[0]

    def test_parse_legacy_hardware_scale_v6_uses_uint16_tick(self):
        # Legacy hardware-scale v6 wrote 26-byte samples with a uint16 tick
        mask = 0x1FFF
        sample = struct.pack('<H HHHH hhhh HHHH', 4, 930, 925, 90, 88, 0, 0, 0, 0, 0, 0, 0, 0)
        data = _build_v5plus_header(6, 26, mask, 1) + sample
        shot = parse_binary_shot(data, "000151")
        assert shot.samples[0]['t'] == 4 * 250
        assert shot.samples[0]['ct'] == 92.5

    def test_sample_size_mismatch_raises(self):
        data = _build_v5plus_header(7, 32, 0x3FFF, 1) + _full_sample_v7(0, 90, 0, 0) + b'\x00\x00'
        try:
            parse_binary_shot(data, "000999")
        except ValueError as e:
            assert "sample size mismatch" in str(e)
        else:
            raise AssertionError("expected ValueError")
