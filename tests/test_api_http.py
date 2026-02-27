"""Tests for HTTP client including list_recent_shots and timeout handling."""

import asyncio
import struct

import aiohttp
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gaggimate_mcp.api.http import GaggimateHTTPClient
from gaggimate_mcp.config import GaggimateConfig
from gaggimate_mcp.errors import GaggimateError, ErrorCode
from gaggimate_mcp.parsers.index import (
    INDEX_MAGIC,
    SHOT_FLAG_COMPLETED,
)


def _build_index_binary(entries: list[dict]) -> bytes:
    """Build a minimal binary index for testing.

    Args:
        entries: List of dicts with keys: id, timestamp, duration, volume, rating, flags,
                 profile_id, profile_name
    """
    entry_size = 128
    entry_count = len(entries)
    next_id = max(e["id"] for e in entries) + 1 if entries else 1

    # Header (32 bytes)
    header = struct.pack("<I H H I I", INDEX_MAGIC, 1, entry_size, entry_count, next_id)
    header += b"\x00" * (32 - len(header))

    data = header
    for e in entries:
        profile_id = e.get("profile_id", "default").encode()[:32]
        profile_id += b"\x00" * (32 - len(profile_id))
        profile_name = e.get("profile_name", "Default").encode()[:48]
        profile_name += b"\x00" * (48 - len(profile_name))

        entry = struct.pack(
            "<I I I H B B",
            e["id"],
            e["timestamp"],
            e.get("duration", 25000),
            int(e.get("volume", 40.0) * 10),
            e.get("rating", 0),
            e.get("flags", SHOT_FLAG_COMPLETED),
        )
        entry += profile_id + profile_name
        entry += b"\x00" * (entry_size - len(entry))
        data += entry

    return data


def _make_entries(count: int) -> list[dict]:
    """Create N test index entries with descending timestamps."""
    return [
        {
            "id": i + 1,
            "timestamp": 1700000000 - (i * 3600),
            "duration": 25000 + i * 1000,
            "volume": 36.0 + i,
            "rating": 0,
            "flags": SHOT_FLAG_COMPLETED,
            "profile_id": f"profile_{i}",
            "profile_name": f"Profile {i}",
        }
        for i in range(count)
    ]


class TestListRecentShots:
    """Tests for list_recent_shots / fetch_shot_index."""

    @pytest.fixture
    def client(self):
        config = GaggimateConfig()
        return GaggimateHTTPClient(config)

    @pytest.mark.asyncio
    async def test_list_recent_shots_limit_3(self, client):
        """list_recent_shots with limit=3 should return exactly 3 shots."""
        entries = _make_entries(5)
        binary_data = _build_index_binary(entries)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=binary_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.list_recent_shots(limit=3)

        assert len(result) == 3
        # Should be sorted newest-first (by parser)
        for shot in result:
            assert "id" in shot
            assert "timestamp" in shot

    @pytest.mark.asyncio
    async def test_list_recent_shots_limit_1(self, client):
        """list_recent_shots with limit=1 should return exactly 1 shot."""
        entries = _make_entries(5)
        binary_data = _build_index_binary(entries)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=binary_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.list_recent_shots(limit=1)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_recent_shots_empty_index(self, client):
        """list_recent_shots returns empty list when index is 404."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.list_recent_shots(limit=3)

        assert result == []

    @pytest.mark.asyncio
    async def test_list_recent_shots_http_error(self, client):
        """list_recent_shots raises GaggimateError on HTTP 500."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.reason = "Internal Server Error"
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(GaggimateError) as exc_info:
                await client.list_recent_shots(limit=3)

        assert exc_info.value.code == ErrorCode.API_ERROR
        assert "500" in exc_info.value.message


class TestTimeoutHandling:
    """Tests for asyncio.TimeoutError handling — the root cause of the bug."""

    @pytest.fixture
    def client(self):
        config = GaggimateConfig()
        return GaggimateHTTPClient(config)

    @pytest.mark.asyncio
    async def test_fetch_shot_index_timeout_raises_gaggimate_error(self, client):
        """asyncio.TimeoutError should be caught and wrapped as GaggimateError(TIMEOUT)."""
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=asyncio.TimeoutError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(GaggimateError) as exc_info:
                await client.fetch_shot_index(limit=3)

        assert exc_info.value.code == ErrorCode.TIMEOUT
        assert exc_info.value.retryable is True
        assert "timed out" in exc_info.value.message.lower()
        assert "shot index" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_list_recent_shots_timeout_raises_gaggimate_error(self, client):
        """list_recent_shots timeout should propagate as GaggimateError(TIMEOUT)."""
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=asyncio.TimeoutError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(GaggimateError) as exc_info:
                await client.list_recent_shots(limit=3)

        assert exc_info.value.code == ErrorCode.TIMEOUT
        assert exc_info.value.retryable is True

    @pytest.mark.asyncio
    async def test_fetch_shot_timeout_raises_gaggimate_error(self, client):
        """fetch_shot timeout should be caught and wrapped as GaggimateError(TIMEOUT)."""
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=asyncio.TimeoutError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(GaggimateError) as exc_info:
                await client.fetch_shot("123")

        assert exc_info.value.code == ErrorCode.TIMEOUT
        assert exc_info.value.retryable is True
        assert "123" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_timeout_error_has_nonempty_message(self, client):
        """Verify the bug fix: TimeoutError (empty str) should produce a useful message."""
        # asyncio.TimeoutError() has str() == ""
        timeout_err = asyncio.TimeoutError()
        assert str(timeout_err) == "", "Precondition: TimeoutError has empty str()"

        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=timeout_err)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(GaggimateError) as exc_info:
                await client.list_recent_shots(limit=3)

        # The error message must NOT be empty
        assert len(exc_info.value.message) > 0
        assert "timed out" in exc_info.value.message.lower()


class TestConnectionErrorHandling:
    """Tests for aiohttp.ClientError handling."""

    @pytest.fixture
    def client(self):
        config = GaggimateConfig()
        return GaggimateHTTPClient(config)

    @pytest.mark.asyncio
    async def test_connection_error_raises_device_unreachable(self, client):
        """aiohttp.ClientError should map to DEVICE_UNREACHABLE."""
        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(), os_error=OSError("Connection refused")
            )
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(GaggimateError) as exc_info:
                await client.list_recent_shots(limit=3)

        assert exc_info.value.code == ErrorCode.DEVICE_UNREACHABLE
