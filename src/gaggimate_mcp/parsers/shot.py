"""Parser for .slog binary shot files.

Mirrors shot_log_format.h from the Gaggimate firmware.

Firmware v1.9.0 (shot log version 7) widened the first sample field (``t``)
from a uint16 tick index to a uint32 elapsed-milliseconds counter and added
a ``wp`` (cumulative water pumped) field, growing the per-sample record from
26 to 30 bytes. The decoder below trusts the on-disk record size
(``reserved0``) and header size rather than hardcoding them, so it degrades
gracefully on versions newer than it was written against instead of silently
misaligning the byte stream (see ``ShotData.warnings``).
"""

import struct
from dataclasses import dataclass, field
from typing import Optional


# Constants
HEADER_SIZE_V4 = 128
HEADER_SIZE_V5 = 512
MAGIC = 0x544F4853  # 'SHOT'

# Highest shot log version this parser has an authoritative field layout
# for. Anything newer is still decoded (known fields, trusted record size
# from `reserved0`) but produces a warning rather than failing outright.
SUPPORTED_MAX_VERSION = 8

# Scaling factors
TEMP_SCALE = 10
PRESSURE_SCALE = 10
FLOW_SCALE = 100
WEIGHT_SCALE = 10
RESISTANCE_SCALE = 100

# Plausibility bounds used for the defence-in-depth sanity pass below.
# Wider than the brewing safety limits in config.py (which bound what is
# safe to *command*) because telemetry can legitimately show e.g. a flush
# or a cooling flush outside the normal brew temperature band.
PLAUSIBLE_TEMP_RANGE_C = (0.0, 150.0)
PLAUSIBLE_PRESSURE_RANGE_BAR = (-1.0, 20.0)
PLAUSIBLE_WEIGHT_MAX_G = 500.0

# Field bit positions (must match shot_log_format.h)
FIELD_BITS = {
    'T': 0,    # tick (v1-v5) / elapsed ms (v6+)
    'TT': 1,   # target temp
    'CT': 2,   # current temp
    'TP': 3,   # target pressure
    'CP': 4,   # current pressure
    'FL': 5,   # pump flow
    'TF': 6,   # target flow
    'PF': 7,   # puck flow
    'VF': 8,   # volumetric flow
    'V': 9,    # volumetric weight
    'EV': 10,  # estimated weight
    'PR': 11,  # puck resistance
    'SI': 12,  # system info (v2+)
    'WP': 13,  # cumulative water pumped (v7+)
}

# Phase exit reason codes stored in PhaseTransition.transition_reason /
# ShotData.final_exit_reason (v7+; byte was reserved padding through v5).
PHASE_EXIT_REASON_LABELS = {
    0: "Unknown",
    1: "Volumetric target",
    2: "Pressure target",
    3: "Flow target",
    4: "Pumped target",
    5: "Duration",
    6: "Safety timeout",
    7: "Aborted",
    8: "Hold released",
    9: "Ratio target",
}


@dataclass
class FieldDef:
    """Field definition with parsing info."""
    name: str
    field_type: str  # 'uint8', 'uint16', 'int16'
    scale: Optional[float] = None
    transform: Optional[callable] = None


def _decode_system_info(val: int, _sample_interval: int) -> dict:
    """Decode the `si` bitfield (see SYSTEM_INFO_* in shot_log_format.h)."""
    return {
        'raw': val,
        'shot_started_volumetric': bool(val & 0x0001),
        'currently_volumetric': bool(val & 0x0002),
        'bluetooth_scale_connected': bool(val & 0x0004),
        'volumetric_available': bool(val & 0x0008),
        'extended_recording': bool(val & 0x0010),
        'process_is_brew': bool(val & 0x0020),            # v8+
        'target_is_volumetric': bool(val & 0x0040),        # v8+
        'phase_has_volumetric': bool(val & 0x0080),        # v8+
        'active_scale_connected': bool(val & 0x0100),      # v8+
    }


# Field definitions
FIELD_DEFS = {
    FIELD_BITS['T']: FieldDef(
        name='t',
        field_type='uint16',
        transform=lambda val, sample_interval: val * sample_interval
    ),
    FIELD_BITS['TT']: FieldDef(name='tt', field_type='uint16', scale=TEMP_SCALE),
    FIELD_BITS['CT']: FieldDef(name='ct', field_type='uint16', scale=TEMP_SCALE),
    FIELD_BITS['TP']: FieldDef(name='tp', field_type='uint16', scale=PRESSURE_SCALE),
    FIELD_BITS['CP']: FieldDef(name='cp', field_type='uint16', scale=PRESSURE_SCALE),
    FIELD_BITS['FL']: FieldDef(name='fl', field_type='int16', scale=FLOW_SCALE),
    FIELD_BITS['TF']: FieldDef(name='tf', field_type='int16', scale=FLOW_SCALE),
    FIELD_BITS['PF']: FieldDef(name='pf', field_type='int16', scale=FLOW_SCALE),
    FIELD_BITS['VF']: FieldDef(name='vf', field_type='int16', scale=FLOW_SCALE),
    FIELD_BITS['V']: FieldDef(name='v', field_type='uint16', scale=WEIGHT_SCALE),
    FIELD_BITS['EV']: FieldDef(name='ev', field_type='uint16', scale=WEIGHT_SCALE),
    FIELD_BITS['PR']: FieldDef(name='pr', field_type='uint16', scale=RESISTANCE_SCALE),
    FIELD_BITS['SI']: FieldDef(name='systemInfo', field_type='uint16', transform=_decode_system_info),
    FIELD_BITS['WP']: FieldDef(name='wp', field_type='uint16', scale=WEIGHT_SCALE),
}


@dataclass
class PhaseTransition:
    """Phase transition marker."""
    sample_index: int
    phase_number: int
    phase_name: str
    transition_reason: int = 0  # PHASE_EXIT_REASON_* code (v7+; 0 on older files)
    transition_reason_label: str = "Unknown"


@dataclass
class ShotData:
    """Parsed shot data."""
    id: str
    version: int
    fields_mask: int
    sample_count: int
    sample_interval: int
    profile_id: str
    profile_name: str
    timestamp: int
    rating: int
    duration: int
    weight: Optional[float]
    samples: list[dict] = field(default_factory=list)
    phases: list[PhaseTransition] = field(default_factory=list)
    incomplete: bool = False
    final_exit_reason: int = 0  # PHASE_EXIT_REASON_* code (v7+; 0 on older files)
    final_exit_reason_label: str = "Unknown"
    brew_delay_ms: int = 0  # Predictive volumetric lead time the shot ran with (v7+)
    warnings: list[str] = field(default_factory=list)


def is_html_response(data: bytes) -> bool:
    """Check if data looks like HTML instead of binary shot data.

    Some firmware versions return HTML error pages instead of binary data
    when the device is overloaded or the endpoint is temporarily unavailable.
    """
    if len(data) < 4:
        return False
    prefix = data[:15].lower()
    return prefix.startswith(b'<!doc') or prefix.startswith(b'<html')


def decode_c_string(data: bytes) -> str:
    """Decode null-terminated C string."""
    null_pos = data.find(b'\x00')
    if null_pos == -1:
        return data.decode('utf-8', errors='ignore')
    return data[:null_pos].decode('utf-8', errors='ignore')


def count_set_bits(n: int) -> int:
    """Count number of set bits in integer."""
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count


def parse_binary_shot(data: bytes, shot_id: str) -> ShotData:
    """Parse binary shot file (.slog format).

    Args:
        data: Binary data from .slog file
        shot_id: Shot identifier

    Returns:
        Parsed shot data

    Raises:
        ValueError: If file format is invalid
    """
    # Check for HTML responses before size validation — the device may return
    # a short HTML error page that would otherwise trigger "Shot file too small"
    if is_html_response(data):
        raise ValueError(
            "Device returned an HTML page instead of binary shot data. "
            "This usually means the device is overloaded or the firmware's "
            "web server is serving its UI instead of the API endpoint."
        )

    if len(data) < HEADER_SIZE_V4:
        raise ValueError(f"Shot file too small: {len(data)} bytes")

    # Parse header
    magic = struct.unpack_from('<I', data, 0)[0]
    if magic != MAGIC:
        raise ValueError(f"Invalid shot magic: 0x{magic:08x} (expected 0x{MAGIC:08x})")

    version = struct.unpack_from('<B', data, 4)[0]
    # reserved0 (byte 5) carries the device's own per-sample record size.
    # The parser used to never read it, assuming every file was 26-byte-
    # stride v5 — on a v6+ file that misaligns every sample by 2-4 bytes.
    reserved0 = struct.unpack_from('<B', data, 5)[0]
    on_disk_header_size = struct.unpack_from('<H', data, 6)[0]

    expected_header_size = HEADER_SIZE_V4 if version <= 4 else HEADER_SIZE_V5

    if len(data) < expected_header_size:
        raise ValueError(f"Shot file too small for version {version}: {len(data)} bytes")

    if on_disk_header_size != expected_header_size:
        raise ValueError(
            f"Header size mismatch for version {version}: expected "
            f"{expected_header_size}, file declares {on_disk_header_size}"
        )

    header_size = expected_header_size

    # Parse header fields
    sample_interval = struct.unpack_from('<H', data, 8)[0]
    fields_mask = struct.unpack_from('<I', data, 12)[0]
    sample_count = struct.unpack_from('<I', data, 16)[0]
    duration = struct.unpack_from('<I', data, 20)[0]
    timestamp = struct.unpack_from('<I', data, 24)[0]

    profile_id_bytes = data[28:60]
    profile_name_bytes = data[60:108]
    weight_raw = struct.unpack_from('<H', data, 108)[0]

    profile_id = decode_c_string(profile_id_bytes)
    profile_name = decode_c_string(profile_name_bytes)
    weight = weight_raw / WEIGHT_SCALE if weight_raw > 0 else None

    warnings: list[str] = []

    if version > SUPPORTED_MAX_VERSION:
        warnings.append(
            f"Shot log version {version} is newer than the supported maximum "
            f"({SUPPORTED_MAX_VERSION}); unknown fields will be skipped."
        )

    # Parse phase transitions (V5+)
    phases: list[PhaseTransition] = []
    final_exit_reason = 0
    brew_delay_ms = 0
    if version >= 5:
        transition_count = struct.unpack_from('<B', data, 458)[0]
        base_offset = 110

        for i in range(min(transition_count, 12)):
            offset = base_offset + i * 29
            sample_index = struct.unpack_from('<H', data, offset)[0]
            phase_number = struct.unpack_from('<B', data, offset + 2)[0]
            # transition_reason (v7+) was reserved padding through v5; a
            # legacy file reads 0 here, which maps to "Unknown" below.
            transition_reason = struct.unpack_from('<B', data, offset + 3)[0]
            phase_name_bytes = data[offset + 4:offset + 29]
            phase_name = decode_c_string(phase_name_bytes)

            phases.append(PhaseTransition(
                sample_index=sample_index,
                phase_number=phase_number,
                phase_name=phase_name,
                transition_reason=transition_reason,
                transition_reason_label=PHASE_EXIT_REASON_LABELS.get(transition_reason, "Unknown"),
            ))

        # final_exit_reason / brew_delay_ms (v7+) were reserved padding
        # through v6; legacy files read 0 here.
        final_exit_reason = struct.unpack_from('<B', data, 459)[0]
        brew_delay_ms = struct.unpack_from('<H', data, 460)[0]

    # Determine the per-sample record layout.
    #
    # v1-v5: t is a uint16 tick index (multiplied by sample_interval).
    # v6+:   t is a uint32 elapsed-millisecond counter -- EXCEPT the legacy
    #        hardware-scale v6 fork, which kept the 26-byte/tick-index
    #        layout. The on-disk reserved0 + fields_mask combination
    #        disambiguates the two, mirroring the firmware's own reference
    #        decoder (web/src/pages/ShotHistory/parseBinaryShot.js).
    fields_per_sample = count_set_bits(fields_mask)
    if fields_per_sample == 0:
        raise ValueError("Invalid shot file: fields_mask is 0 (no fields recorded)")

    is_legacy_hardware_scale_v6 = (
        version == 6 and reserved0 == 26 and fields_mask == 0x1FFF
    )
    elapsed_ms_timestamp = version >= 6 and not is_legacy_hardware_scale_v6
    expected_sample_size = fields_per_sample * 2 + (2 if elapsed_ms_timestamp else 0)

    # Trust the device's own declared record size over our derived
    # expectation -- it is the ground truth for *this* file. Only fall
    # back to the derived size when the device didn't report one (older
    # synthetic/test data, or a genuinely zeroed reserved0).
    if reserved0:
        sample_data_size = reserved0
        if reserved0 != expected_sample_size:
            warnings.append(
                f"Sample size mismatch: fields_mask implies {expected_sample_size} "
                f"bytes/sample, device reports {reserved0}. Trusting the "
                f"device-reported size; any extra bytes are treated as unknown."
            )
    else:
        sample_data_size = expected_sample_size

    # Determine actual samples (handle truncated files)
    available_data = len(data) - header_size
    actual_sample_count = min(sample_count, available_data // sample_data_size)
    incomplete = actual_sample_count < sample_count

    trailing_bytes = available_data - actual_sample_count * sample_data_size
    if trailing_bytes:
        warnings.append(
            f"{trailing_bytes} trailing byte(s) after the last complete "
            f"sample were ignored."
        )

    # Build field order based on mask. Range covers every bit so a future
    # firmware's unknown high bits are still walked over (and their width
    # accounted for in the stride) even though we don't know their meaning.
    field_order = []
    for bit in range(32):
        if fields_mask & (1 << bit):
            field_order.append(bit)

    # Parse samples
    samples = []
    for i in range(actual_sample_count):
        sample = {}
        sample_offset = header_size + i * sample_data_size

        # Add phase number based on transitions
        for phase in reversed(phases):
            if i >= phase.sample_index:
                sample['phase'] = phase.phase_number
                break

        offset = sample_offset
        for field_bit in field_order:
            field_def = FIELD_DEFS.get(field_bit)
            is_wide_t = elapsed_ms_timestamp and field_bit == FIELD_BITS['T']
            width = 4 if is_wide_t else 2

            if field_def is None:
                # Unknown field (future firmware). Assumed 2 bytes wide,
                # per the firmware's own reference decoder -- skip it but
                # still advance the cursor so later known fields stay
                # aligned.
                offset += width
                continue

            if is_wide_t:
                value = struct.unpack_from('<I', data, offset)[0]
                sample[field_def.name] = value  # already milliseconds
            else:
                if field_def.field_type == 'int16':
                    value = struct.unpack_from('<h', data, offset)[0]
                else:  # uint16 or uint8
                    value = struct.unpack_from('<H', data, offset)[0]

                if field_def.transform:
                    sample[field_def.name] = field_def.transform(value, sample_interval)
                elif field_def.scale:
                    sample[field_def.name] = value / field_def.scale
                else:
                    sample[field_def.name] = value

            offset += width

        # v6/v7 have no source-neutral "active scale" bit -- only the
        # legacy hardware-scale fork and v8+ record it. Mirror the
        # Bluetooth-scale bit onto it so downstream code has one field to
        # read regardless of firmware version, per the upstream reference.
        if 'systemInfo' in sample and version < 8 and not is_legacy_hardware_scale_v6:
            sample['systemInfo']['active_scale_connected'] = (
                sample['systemInfo']['bluetooth_scale_connected']
            )

        samples.append(sample)

    # Defence in depth: this exact bug produced silently impossible values
    # for weeks. Flag (don't clamp -- that would hide genuine sensor
    # faults too) any sample outside physically plausible bounds, and any
    # timestamp far beyond the header's own duration.
    implausible_count = sum(
        1 for s in samples if (
            ('ct' in s and not (PLAUSIBLE_TEMP_RANGE_C[0] <= s['ct'] <= PLAUSIBLE_TEMP_RANGE_C[1]))
            or ('cp' in s and not (PLAUSIBLE_PRESSURE_RANGE_BAR[0] <= s['cp'] <= PLAUSIBLE_PRESSURE_RANGE_BAR[1]))
            or ('v' in s and s['v'] > PLAUSIBLE_WEIGHT_MAX_G)
        )
    )
    if implausible_count:
        warnings.append(
            f"{implausible_count} sample(s) have temperature, pressure, or "
            f"weight values outside physically plausible ranges -- the "
            f"decode may be misaligned."
        )

    if samples and duration:
        max_t = max(s.get('t', 0) for s in samples)
        if max_t > duration * 2:
            warnings.append(
                f"Sample timestamps (up to {max_t} ms) exceed the shot "
                f"duration ({duration} ms) by more than 2x -- the decode "
                f"may be misaligned."
            )

    return ShotData(
        id=shot_id,
        version=version,
        fields_mask=fields_mask,
        sample_count=actual_sample_count,
        sample_interval=sample_interval,
        profile_id=profile_id,
        profile_name=profile_name,
        timestamp=timestamp,
        rating=0,  # Rating not stored in binary file
        duration=duration,
        weight=weight,
        samples=samples,
        phases=phases,
        incomplete=incomplete,
        final_exit_reason=final_exit_reason,
        final_exit_reason_label=PHASE_EXIT_REASON_LABELS.get(final_exit_reason, "Unknown"),
        brew_delay_ms=brew_delay_ms,
        warnings=warnings,
    )
