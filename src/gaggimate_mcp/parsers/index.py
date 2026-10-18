"""Parser for index.bin binary shot index files.

Mirrors shot_log_format.h ShotIndexHeader and ShotIndexEntry.
"""

import struct
from dataclasses import dataclass
from typing import Optional


# Constants
INDEX_HEADER_SIZE = 32
INDEX_ENTRY_SIZE = 128
INDEX_MAGIC = 0x58444953  # 'SIDX'

# Index entry flags
SHOT_FLAG_COMPLETED = 0x01
SHOT_FLAG_DELETED = 0x02
SHOT_FLAG_HAS_NOTES = 0x04

WEIGHT_SCALE = 10
TEMP_SCALE = 10
PRESSURE_SCALE = 10
FLOW_SCALE = 100


def decode_c_string(data: bytes) -> str:
    """Decode null-terminated C string."""
    null_pos = data.find(b'\x00')
    if null_pos == -1:
        return data.decode('utf-8', errors='ignore')
    return data[:null_pos].decode('utf-8', errors='ignore')


@dataclass
class IndexEntry:
    """Shot index entry."""
    id: int
    timestamp: int
    duration: int
    volume: Optional[float]
    rating: int
    flags: int
    profile_id: str
    profile_name: str
    completed: bool
    deleted: bool
    has_notes: bool
    incomplete: bool
    # Per-shot aggregates (firmware v1.9.0+; carved from former reserved
    # padding at offsets 96/98/100, so entry size is unchanged). None on
    # entries written by older firmware, which read back as 0.
    avg_temp: Optional[float] = None
    max_pressure: Optional[float] = None
    avg_flow: Optional[float] = None


@dataclass
class IndexHeader:
    """Index file header."""
    magic: int
    version: int
    entry_size: int
    entry_count: int
    next_id: int


@dataclass
class IndexData:
    """Complete index data."""
    header: IndexHeader
    entries: list[IndexEntry]


def parse_binary_index(data: bytes) -> IndexData:
    """Parse binary shot index file.

    Args:
        data: Binary data from index.bin file

    Returns:
        Parsed index data

    Raises:
        ValueError: If file format is invalid
    """
    if len(data) < INDEX_HEADER_SIZE:
        raise ValueError("Index file too small")

    # Parse header
    magic = struct.unpack_from('<I', data, 0)[0]
    if magic != INDEX_MAGIC:
        raise ValueError(f"Invalid index magic: 0x{magic:08x} (expected 0x{INDEX_MAGIC:08x})")

    version = struct.unpack_from('<H', data, 4)[0]
    entry_size = struct.unpack_from('<H', data, 6)[0]
    entry_count = struct.unpack_from('<I', data, 8)[0]
    next_id = struct.unpack_from('<I', data, 12)[0]

    if entry_size != INDEX_ENTRY_SIZE:
        raise ValueError(f"Unsupported entry size {entry_size} (expected {INDEX_ENTRY_SIZE})")

    expected_size = INDEX_HEADER_SIZE + entry_count * INDEX_ENTRY_SIZE
    if len(data) < expected_size:
        raise ValueError(f"Index file truncated: {len(data)} bytes (expected {expected_size})")

    # Parse entries
    entries = []
    for i in range(entry_count):
        base = INDEX_HEADER_SIZE + i * INDEX_ENTRY_SIZE

        entry_id = struct.unpack_from('<I', data, base + 0)[0]
        timestamp = struct.unpack_from('<I', data, base + 4)[0]
        duration = struct.unpack_from('<I', data, base + 8)[0]
        volume_raw = struct.unpack_from('<H', data, base + 12)[0]
        rating = struct.unpack_from('<B', data, base + 14)[0]
        flags = struct.unpack_from('<B', data, base + 15)[0]

        profile_id_bytes = data[base + 16:base + 48]
        profile_name_bytes = data[base + 48:base + 96]

        profile_id = decode_c_string(profile_id_bytes)
        profile_name = decode_c_string(profile_name_bytes)

        volume = volume_raw / WEIGHT_SCALE if volume_raw > 0 else None

        avg_temp_raw = struct.unpack_from('<H', data, base + 96)[0]
        max_pressure_raw = struct.unpack_from('<H', data, base + 98)[0]
        avg_flow_raw = struct.unpack_from('<H', data, base + 100)[0]

        entries.append(IndexEntry(
            id=entry_id,
            timestamp=timestamp,
            duration=duration,
            volume=volume,
            rating=rating,
            flags=flags,
            profile_id=profile_id,
            profile_name=profile_name,
            completed=bool(flags & SHOT_FLAG_COMPLETED),
            deleted=bool(flags & SHOT_FLAG_DELETED),
            has_notes=bool(flags & SHOT_FLAG_HAS_NOTES),
            incomplete=not bool(flags & SHOT_FLAG_COMPLETED),
            avg_temp=avg_temp_raw / TEMP_SCALE if avg_temp_raw > 0 else None,
            max_pressure=max_pressure_raw / PRESSURE_SCALE if max_pressure_raw > 0 else None,
            avg_flow=avg_flow_raw / FLOW_SCALE if avg_flow_raw > 0 else None,
        ))

    header = IndexHeader(
        magic=magic,
        version=version,
        entry_size=entry_size,
        entry_count=entry_count,
        next_id=next_id,
    )

    return IndexData(header=header, entries=entries)


def index_to_shot_list(index_data: IndexData) -> list[dict]:
    """Filter deleted entries and convert to shot list.

    Args:
        index_data: Parsed index data

    Returns:
        List of shot items sorted by timestamp (most recent first)
    """
    shots = []
    for entry in index_data.entries:
        if entry.deleted:
            continue

        shots.append({
            'id': str(entry.id),
            'profile': entry.profile_name,
            'profile_id': entry.profile_id,
            'timestamp': entry.timestamp,
            'duration': entry.duration,
            'volume': entry.volume,
            'rating': entry.rating if entry.rating > 0 else None,
            'incomplete': entry.incomplete,
            'has_notes': entry.has_notes,
            'avg_temp': entry.avg_temp,
            'max_pressure': entry.max_pressure,
            'avg_flow': entry.avg_flow,
        })

    # Sort by timestamp, most recent first
    shots.sort(key=lambda x: x['timestamp'], reverse=True)
    return shots
