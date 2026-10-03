"""Configuration management for Gaggimate MCP server."""

from pathlib import Path
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GaggimateConfig(BaseSettings):
    """Gaggimate MCP server configuration with validation."""

    model_config = SettingsConfigDict(
        env_prefix="GAGGIMATE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_nested_delimiter="__",
        populate_by_name=True,
    )

    # Connection settings
    # GAGGIMATE_HOST is the documented name; the env_prefix-derived
    # GAGGIMATE_GAGGIMATE_HOST keeps working for existing setups
    gaggimate_host: str = Field(
        default="gaggimate.local",
        validation_alias=AliasChoices("gaggimate_host", "GAGGIMATE_HOST", "GAGGIMATE_GAGGIMATE_HOST"),
    )
    gaggimate_protocol: str = Field(
        default="ws",
        validation_alias=AliasChoices("gaggimate_protocol", "GAGGIMATE_PROTOCOL", "GAGGIMATE_GAGGIMATE_PROTOCOL"),
    )
    request_timeout: float = 15.0

    # Safety limits
    max_temperature: float = 100.0
    min_temperature: float = 25.0
    max_pressure: float = 12.0
    min_pressure: float = 0.0

    # Storage
    profiles_dir: Path = Path("./profiles")
    storage_path: Path = Path("./data")

    # Content directories (for MCP resources)
    knowledge_dir: Path = Path("./knowledge")
    coffees_dir: Path = Path("./coffees")
    user_dir: Path = Path("./user")

    # Agent settings
    ai_profile_suffix: str = " [AI]"
    ai_notes_prefix: str = "[Updated by AI]: "

    # Observability
    log_level: str = "INFO"
    enable_metrics: bool = True

    @property
    def host(self) -> str:
        """Get host (alias for gaggimate_host)."""
        return self.gaggimate_host

    @property
    def use_https(self) -> bool:
        """Check if HTTPS/WSS should be used."""
        return self.gaggimate_protocol == "wss"

    @property
    def websocket_url(self) -> str:
        """Generate WebSocket URL from configuration."""
        return f"{self.gaggimate_protocol}://{self.gaggimate_host}/ws"

    @property
    def http_base_url(self) -> str:
        """Generate HTTP base URL from configuration."""
        protocol = "https" if self.gaggimate_protocol == "wss" else "http"
        return f"{protocol}://{self.gaggimate_host}"

    def validate_temperature(self, temp: float) -> float:
        """
        Clamp temperature to safety limits.

        Args:
            temp: Temperature in Celsius

        Returns:
            Temperature clamped to [min_temperature, max_temperature]
        """
        return max(self.min_temperature, min(self.max_temperature, temp))

    def validate_pressure(self, pressure: float) -> float:
        """
        Clamp pressure to safety limits.

        Args:
            pressure: Pressure in bar

        Returns:
            Pressure clamped to [min_pressure, max_pressure]
        """
        return max(self.min_pressure, min(self.max_pressure, pressure))
