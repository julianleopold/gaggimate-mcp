"""Parser for .slog binary shot files.

Mirrors shot_log_format.h from the Gaggimate firmware.
"""

import struct
from dataclasses import dataclass, field
from typing import Optional


# Constants
HEADER_SIZE_V4 = 128
HEADER_SIZE_V5 = 512
MAGIC = 0x544F4853  # 'SHOT'

# Scaling factors
TEMP_SCALE = 10
PRESSURE_SCALE = 10
FLOW_SCALE = 100
WEIGHT_SCALE = 10
RESISTANCE_SCALE = 100
WATER_PUMPED_SCALE = 10

# Field bit positions (must match shot_log_format.h)
FIELD_BITS = {
    'T': 0,    # tick
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


@dataclass
class FieldDef:
    """Field definition with parsing info."""
    name: str
    field_type: str  # 'uint8', 'uint16', 'int16'
    scale: Optional[float] = None
    transform: Optional[callable] = None


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
    FIELD_BITS['SI']: FieldDef(
        name='systemInfo',
        field_type='uint16',
        transform=lambda val, _: {
            'raw': val,
            'shot_started_volumetric': bool(val & 0x0001),
            'currently_volumetric': bool(val & 0x0002),
            'bluetooth_scale_connected': bool(val & 0x0004),
            'volumetric_available': bool(val & 0x0008),
            'extended_recording': bool(val & 0x0010),
        }
    ),
    FIELD_BITS['WP']: FieldDef(name='wp', field_type='uint16', scale=WATER_PUMPED_SCALE),
}


def has_elapsed_timestamp(version: int, reserved0: int, fields_mask: int) -> bool:
    """Whether samples store `t` as a uint32 of elapsed milliseconds.

    v1-v5 store `t` as a uint16 sample index (multiply by sample_interval).
    v6+ store `t` as a uint32 of real elapsed ms. One exception mirrors
    shotLogHasElapsedTimestamp() in the firmware: a legacy hardware-scale v6
    build wrote 26-byte samples with a uint16 `t` (reserved0 == 26, mask 0x1FFF).
    """
    if version < 6:
        return False
    return not (version == 6 and reserved0 == 26 and fields_mask == 0x1FFF)


@dataclass
class PhaseTransition:
    """Phase transition marker."""
    sample_index: int
    phase_number: int
    phase_name: str


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
    header_size = HEADER_SIZE_V5 if version >= 5 else HEADER_SIZE_V4

    if len(data) < header_size:
        raise ValueError(f"Shot file too small for version {version}: {len(data)} bytes")

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

    # Parse phase transitions (V5+)
    phases: list[PhaseTransition] = []
    if version >= 5:
        transition_count = struct.unpack_from('<B', data, 458)[0]
        base_offset = 110

        for i in range(min(transition_count, 12)):
            offset = base_offset + i * 29
            sample_index = struct.unpack_from('<H', data, offset)[0]
            phase_number = struct.unpack_from('<B', data, offset + 2)[0]
            phase_name_bytes = data[offset + 4:offset + 29]
            phase_name = decode_c_string(phase_name_bytes)

            phases.append(PhaseTransition(
                sample_index=sample_index,
                phase_number=phase_number,
                phase_name=phase_name
            ))

    # Build field order and per-field byte widths based on mask
    reserved0 = struct.unpack_from('<B', data, 5)[0]
    elapsed_t = has_elapsed_timestamp(version, reserved0, fields_mask)

    field_order = []
    field_widths = []
    for bit in range(32):
        if not fields_mask & (1 << bit):
            continue
        if bit not in FIELD_DEFS:
            raise ValueError(
                f"Unsupported shot log field bit {bit} in fields_mask "
                f"0x{fields_mask:x} (version {version})"
            )
        field_order.append(bit)
        field_widths.append(4 if bit == FIELD_BITS['T'] and elapsed_t else 2)

    if not field_order:
        raise ValueError(f"Invalid shot file: fields_mask is 0 (no fields recorded)")
    sample_data_size = sum(field_widths)

    # v6+ firmware records the sample size in reserved0; a mismatch means we
    # don't understand this layout, so fail rather than return shifted values.
    if version >= 6 and reserved0 not in (0, sample_data_size):
        raise ValueError(
            f"Shot log v{version} sample size mismatch: header says {reserved0} "
            f"bytes, fields_mask 0x{fields_mask:x} implies {sample_data_size}"
        )

    # Determine actual samples (handle truncated files)
    available_data = len(data) - header_size
    actual_sample_count = min(sample_count, available_data // sample_data_size)
    incomplete = actual_sample_count < sample_count

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

        # Parse each field in this sample
        field_offset = sample_offset
        for field_bit, width in zip(field_order, field_widths):
            field_def = FIELD_DEFS[field_bit]

            if field_bit == FIELD_BITS['T'] and elapsed_t:
                # v6+: uint32 elapsed milliseconds, no interval scaling
                sample[field_def.name] = struct.unpack_from('<I', data, field_offset)[0]
                field_offset += width
                continue

            # Read value based on type
            if field_def.field_type == 'int16':
                value = struct.unpack_from('<h', data, field_offset)[0]
            else:  # uint16 or uint8
                value = struct.unpack_from('<H', data, field_offset)[0]
            field_offset += width

            # Apply transform or scale
            if field_def.transform:
                sample[field_def.name] = field_def.transform(value, sample_interval)
            elif field_def.scale:
                sample[field_def.name] = value / field_def.scale
            else:
                sample[field_def.name] = value

        samples.append(sample)

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
    )
