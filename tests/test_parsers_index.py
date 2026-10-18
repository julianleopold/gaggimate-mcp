"""Tests for index.bin binary index file parser."""

import struct
from gaggimate_mcp.parsers.index import (
    parse_binary_index,
    index_to_shot_list,
    INDEX_MAGIC,
    SHOT_FLAG_COMPLETED,
    SHOT_FLAG_DELETED,
    SHOT_FLAG_HAS_NOTES,
)


class TestBinaryIndexParser:
    """Test binary index file parsing."""

    def test_parse_minimal_index(self):
        """Test parsing a minimal index file with one entry."""
        # Header
        magic = INDEX_MAGIC
        version = 1
        entry_size = 128
        entry_count = 1
        next_id = 2

        header = struct.pack('<I H H I I', magic, version, entry_size, entry_count, next_id)
        header = header + b'\x00' * (32 - len(header))

        # Entry
        entry_id = 1
        timestamp = 1640000000
        duration = 25000  # 25 seconds
        volume = 400  # 40.0 grams
        rating = 4
        flags = SHOT_FLAG_COMPLETED
        profile_id = b'test_profile\x00' + b'\x00' * 19
        profile_name = b'Test Profile\x00' + b'\x00' * 35

        entry = struct.pack('<I I I H B B', entry_id, timestamp, duration, volume, rating, flags)
        entry = entry + profile_id + profile_name
        entry = entry + b'\x00' * (128 - len(entry))

        data = header + entry

        # Parse
        index = parse_binary_index(data)

        # Verify header
        assert index.header.magic == INDEX_MAGIC
        assert index.header.version == 1
        assert index.header.entry_size == 128
        assert index.header.entry_count == 1
        assert index.header.next_id == 2

        # Verify entry
        assert len(index.entries) == 1
        assert index.entries[0].id == 1
        assert index.entries[0].timestamp == 1640000000
        assert index.entries[0].duration == 25000
        assert index.entries[0].volume == 40.0
        assert index.entries[0].rating == 4
        assert index.entries[0].flags == SHOT_FLAG_COMPLETED
        assert index.entries[0].profile_id == "test_profile"
        assert index.entries[0].profile_name == "Test Profile"
        assert index.entries[0].completed == True
        assert index.entries[0].deleted == False
        assert index.entries[0].has_notes == False
        assert index.entries[0].incomplete == False

    def test_parse_index_with_flags(self):
        """Test parsing index entries with various flags."""
        header = struct.pack('<I H H I I', INDEX_MAGIC, 1, 128, 3, 4)
        header = header + b'\x00' * (32 - len(header))

        # Entry 1: Completed
        profile_id1 = b'profile1\x00' + b'\x00' * 23
        profile_name1 = b'Profile 1\x00' + b'\x00' * 38
        entry1 = struct.pack('<I I I H B B', 1, 1640000000, 25000, 400, 4, SHOT_FLAG_COMPLETED)
        entry1 = entry1 + profile_id1 + profile_name1 + b'\x00' * (128 - len(entry1) - len(profile_id1) - len(profile_name1))

        # Entry 2: Deleted
        profile_id2 = b'profile2\x00' + b'\x00' * 23
        profile_name2 = b'Profile 2\x00' + b'\x00' * 38
        entry2 = struct.pack('<I I I H B B', 2, 1640000100, 26000, 420, 3, SHOT_FLAG_DELETED)
        entry2 = entry2 + profile_id2 + profile_name2 + b'\x00' * (128 - len(entry2) - len(profile_id2) - len(profile_name2))

        # Entry 3: Has notes
        profile_id3 = b'profile3\x00' + b'\x00' * 23
        profile_name3 = b'Profile 3\x00' + b'\x00' * 38
        entry3 = struct.pack('<I I I H B B', 3, 1640000200, 27000, 0, 0, SHOT_FLAG_HAS_NOTES)
        entry3 = entry3 + profile_id3 + profile_name3 + b'\x00' * (128 - len(entry3) - len(profile_id3) - len(profile_name3))

        data = header + entry1 + entry2 + entry3

        index = parse_binary_index(data)

        assert len(index.entries) == 3
        assert index.entries[0].completed == True
        assert index.entries[0].deleted == False
        assert index.entries[1].completed == False
        assert index.entries[1].deleted == True
        assert index.entries[2].has_notes == True

    def test_index_to_shot_list(self):
        """Test converting index to shot list."""
        header = struct.pack('<I H H I I', INDEX_MAGIC, 1, 128, 3, 4)
        header = header + b'\x00' * (32 - len(header))

        # Entry 1: timestamp 100 (middle)
        profile_id1 = b'prof1\x00' + b'\x00' * 26
        profile_name1 = b'Profile 1\x00' + b'\x00' * 38
        entry1 = struct.pack('<I I I H B B', 1, 100, 25000, 400, 4, SHOT_FLAG_COMPLETED)
        entry1 = entry1 + profile_id1 + profile_name1 + b'\x00' * (128 - len(entry1) - len(profile_id1) - len(profile_name1))

        # Entry 2: timestamp 50 (oldest) - DELETED
        profile_id2 = b'prof2\x00' + b'\x00' * 26
        profile_name2 = b'Profile 2\x00' + b'\x00' * 38
        entry2 = struct.pack('<I I I H B B', 2, 50, 26000, 420, 0, SHOT_FLAG_DELETED)
        entry2 = entry2 + profile_id2 + profile_name2 + b'\x00' * (128 - len(entry2) - len(profile_id2) - len(profile_name2))

        # Entry 3: timestamp 150 (newest)
        profile_id3 = b'prof3\x00' + b'\x00' * 26
        profile_name3 = b'Profile 3\x00' + b'\x00' * 38
        entry3 = struct.pack('<I I I H B B', 3, 150, 27000, 0, 5, SHOT_FLAG_COMPLETED)
        entry3 = entry3 + profile_id3 + profile_name3 + b'\x00' * (128 - len(entry3) - len(profile_id3) - len(profile_name3))

        data = header + entry1 + entry2 + entry3

        index = parse_binary_index(data)
        shot_list = index_to_shot_list(index)

        # Should have 2 shots (deleted one filtered out)
        assert len(shot_list) == 2

        # Should be sorted by timestamp descending (newest first)
        assert shot_list[0]['id'] == '3'
        assert shot_list[0]['timestamp'] == 150
        assert shot_list[1]['id'] == '1'
        assert shot_list[1]['timestamp'] == 100

        # Verify structure
        assert shot_list[0]['profile'] == 'Profile 3'
        assert shot_list[0]['rating'] == 5
        assert shot_list[0]['volume'] is None
        assert shot_list[1]['volume'] == 40.0
        assert shot_list[1]['rating'] == 4

    def test_parse_index_entry_with_v9_aggregates(self):
        """Firmware v1.9.0 carves avg_temp/max_pressure/avg_flow out of the
        former reserved padding at entry offsets 96/98/100."""
        header = struct.pack('<I H H I I', INDEX_MAGIC, 1, 128, 1, 2)
        header = header + b'\x00' * (32 - len(header))

        profile_id = b'profile\x00' + b'\x00' * 24
        profile_name = b'Profile\x00' + b'\x00' * 40
        entry = struct.pack('<I I I H B B', 1, 1640000000, 25000, 400, 4, SHOT_FLAG_COMPLETED)
        entry = entry + profile_id + profile_name
        assert len(entry) == 96
        entry = entry + struct.pack('<HHH', 934, 89, 180)  # avg_temp, max_pressure, avg_flow
        entry = entry + b'\x00' * (128 - len(entry))

        data = header + entry
        index = parse_binary_index(data)

        assert index.entries[0].avg_temp == 93.4
        assert index.entries[0].max_pressure == 8.9
        assert index.entries[0].avg_flow == 1.8

    def test_parse_index_entry_without_aggregates_reads_none(self):
        """Older firmware never wrote these fields -- 0 means not recorded."""
        header = struct.pack('<I H H I I', INDEX_MAGIC, 1, 128, 1, 2)
        header = header + b'\x00' * (32 - len(header))

        profile_id = b'profile\x00' + b'\x00' * 24
        profile_name = b'Profile\x00' + b'\x00' * 40
        entry = struct.pack('<I I I H B B', 1, 1640000000, 25000, 400, 4, SHOT_FLAG_COMPLETED)
        entry = entry + profile_id + profile_name + b'\x00' * (128 - len(entry) - len(profile_id) - len(profile_name))

        data = header + entry
        index = parse_binary_index(data)

        assert index.entries[0].avg_temp is None
        assert index.entries[0].max_pressure is None
        assert index.entries[0].avg_flow is None

    def test_index_to_shot_list_surfaces_aggregates(self):
        """list_recent_shots should get the aggregates without a full .slog fetch."""
        header = struct.pack('<I H H I I', INDEX_MAGIC, 1, 128, 1, 2)
        header = header + b'\x00' * (32 - len(header))

        profile_id = b'profile\x00' + b'\x00' * 24
        profile_name = b'Profile\x00' + b'\x00' * 40
        entry = struct.pack('<I I I H B B', 1, 1640000000, 25000, 400, 4, SHOT_FLAG_COMPLETED)
        entry = entry + profile_id + profile_name
        entry = entry + struct.pack('<HHH', 934, 89, 180)
        entry = entry + b'\x00' * (128 - len(entry))

        data = header + entry
        shot_list = index_to_shot_list(parse_binary_index(data))

        assert shot_list[0]['avg_temp'] == 93.4
        assert shot_list[0]['max_pressure'] == 8.9
        assert shot_list[0]['avg_flow'] == 1.8

    def test_parse_index_invalid_magic(self):
        """Test parsing with invalid magic number."""
        data = struct.pack('<I', 0xDEADBEEF) + b'\x00' * 28

        try:
            parse_binary_index(data)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid index magic" in str(e)

    def test_parse_index_too_small(self):
        """Test parsing truncated index file."""
        data = b'\x00' * 20

        try:
            parse_binary_index(data)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "too small" in str(e)

    def test_parse_index_truncated_entries(self):
        """Test parsing index with truncated entry data."""
        header = struct.pack('<I H H I I', INDEX_MAGIC, 1, 128, 5, 6)
        header = header + b'\x00' * (32 - len(header))

        # Only provide 1 entry instead of claimed 5
        entry = b'\x00' * 128

        data = header + entry

        try:
            parse_binary_index(data)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "truncated" in str(e)
