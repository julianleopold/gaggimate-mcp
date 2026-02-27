"""HTTP client for Gaggimate API.

Handles shot history operations via HTTP:
- Fetch index.bin (shot history index)
- Fetch specific shot .slog files
"""

import asyncio
from typing import Optional
import aiohttp

from gaggimate_mcp.config import GaggimateConfig
from gaggimate_mcp.errors import GaggimateError, ErrorCode
from gaggimate_mcp.logging_config import get_logger
from gaggimate_mcp.parsers.index import parse_binary_index, index_to_shot_list
from gaggimate_mcp.parsers.shot import parse_binary_shot, ShotData

logger = get_logger(__name__)


class GaggimateHTTPClient:
    """HTTP client for Gaggimate API."""

    def __init__(self, config: Optional[GaggimateConfig] = None):
        """Initialize HTTP client.

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or GaggimateConfig()
        self.timeout = aiohttp.ClientTimeout(total=5.0)  # 5 second timeout

    @property
    def base_url(self) -> str:
        """Get base HTTP URL from config."""
        protocol = "https" if self.config.use_https else "http"
        return f"{protocol}://{self.config.host}/api/history"

    async def fetch_shot_index(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> list[dict]:
        """Fetch shot history index.

        Args:
            limit: Maximum number of shots to return
            offset: Number of shots to skip

        Returns:
            List of shot metadata dictionaries (sorted newest first)

        Raises:
            GaggimateError: If request fails
        """
        url = f"{self.base_url}/index.bin"
        logger.info("fetching_shot_index", url=url)

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(
                    url,
                    headers={"Accept": "application/octet-stream"}
                ) as response:
                    if response.status == 404:
                        # Index doesn't exist - empty history
                        logger.info("shot_index_not_found", url=url)
                        return []

                    if response.status != 200:
                        logger.error("http_error", status=response.status, url=url)
                        raise GaggimateError(
                            code=ErrorCode.API_ERROR,
                            message=f"HTTP {response.status}: {response.reason}"
                        )

                    # Parse binary index
                    binary_data = await response.read()
                    index_data = parse_binary_index(binary_data)
                    shot_list = index_to_shot_list(index_data)

                    # Apply offset and limit
                    if offset and offset > 0:
                        shot_list = shot_list[offset:]
                    if limit and limit > 0:
                        shot_list = shot_list[:limit]

                    logger.info("shot_index_fetched", count=len(shot_list))
                    return shot_list

        except asyncio.TimeoutError as e:
            logger.error("http_timeout", url=url, timeout=self.timeout.total)
            raise GaggimateError(
                code=ErrorCode.TIMEOUT,
                message=f"Request timed out after {self.timeout.total}s fetching shot index",
                retryable=True,
            ) from e
        except aiohttp.ClientError as e:
            logger.error("http_connection_error", error=str(e), url=url)
            raise GaggimateError(
                code=ErrorCode.DEVICE_UNREACHABLE,
                message=f"HTTP connection error: {str(e)}"
            ) from e
        except ValueError as e:
            logger.error("parse_error", error=str(e), url=url)
            raise GaggimateError(
                code=ErrorCode.PARSE_ERROR,
                message=f"Failed to parse index.bin: {str(e)}"
            ) from e

    async def fetch_shot(self, shot_id: str) -> Optional[ShotData]:
        """Fetch a specific shot by ID.

        Args:
            shot_id: Shot identifier (will be zero-padded to 6 digits)

        Returns:
            Parsed shot data, or None if not found

        Raises:
            GaggimateError: If request fails (excluding 404)
        """
        # Normalize ID to 6 digits for both filename and storage
        padded_id = shot_id.zfill(6)
        url = f"{self.base_url}/{padded_id}.slog"
        logger.info("fetching_shot", shot_id=shot_id, padded_id=padded_id, url=url)

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(
                    url,
                    headers={"Accept": "application/octet-stream"}
                ) as response:
                    if response.status == 404:
                        # Shot not found
                        logger.warning("shot_not_found", shot_id=shot_id, url=url)
                        return None

                    if response.status != 200:
                        logger.error("http_error", status=response.status, url=url)
                        raise GaggimateError(
                            code=ErrorCode.API_ERROR,
                            message=f"HTTP {response.status}: {response.reason}"
                        )

                    # Parse binary shot data - use padded_id as canonical ID
                    binary_data = await response.read()
                    shot_data = parse_binary_shot(binary_data, padded_id)

                    logger.info("shot_fetched", shot_id=shot_id, samples=shot_data.sample_count)
                    return shot_data

        except asyncio.TimeoutError as e:
            logger.error("http_timeout", url=url, timeout=self.timeout.total)
            raise GaggimateError(
                code=ErrorCode.TIMEOUT,
                message=f"Request timed out after {self.timeout.total}s fetching shot {shot_id}",
                retryable=True,
            ) from e
        except aiohttp.ClientError as e:
            logger.error("http_connection_error", error=str(e), url=url)
            raise GaggimateError(
                code=ErrorCode.DEVICE_UNREACHABLE,
                message=f"HTTP connection error: {str(e)}"
            ) from e
        except ValueError as e:
            logger.error("parse_error", error=str(e), url=url)
            raise GaggimateError(
                code=ErrorCode.PARSE_ERROR,
                message=f"Failed to parse shot file: {str(e)}"
            ) from e

    async def list_recent_shots(self, limit: int = 10) -> list[dict]:
        """List recent shots (convenience method).

        Args:
            limit: Maximum number of shots to return (default 10)

        Returns:
            List of recent shot metadata dictionaries

        Raises:
            GaggimateError: If request fails
        """
        return await self.fetch_shot_index(limit=limit)
