"""Tests for configuration management."""

from pathlib import Path
from gaggimate_mcp.config import GaggimateConfig


class TestGaggimateConfig:
    """Test configuration management."""

    def test_default_values(self):
        """Test default configuration values load correctly."""
        config = GaggimateConfig()
        assert config.gaggimate_host == "gaggimate.local"
        assert config.gaggimate_protocol == "ws"
        assert config.request_timeout == 15.0
        assert config.max_temperature == 96.0
        assert config.min_temperature == 60.0
        assert config.max_pressure == 12.0
        assert config.min_pressure == 0.0

    def test_config_override(self):
        """Test configuration values can be overridden."""
        config = GaggimateConfig(
            gaggimate_host="192.168.1.100",
            gaggimate_protocol="wss",
            max_temperature=95.0
        )
        assert config.gaggimate_host == "192.168.1.100"
        assert config.gaggimate_protocol == "wss"
        assert config.max_temperature == 95.0

    def test_websocket_url_generation(self):
        """Test WebSocket URL is generated correctly."""
        config = GaggimateConfig()
        assert config.websocket_url == "ws://gaggimate.local/ws"

    def test_websocket_url_wss(self):
        """Test WebSocket URL with wss protocol."""
        config = GaggimateConfig(gaggimate_protocol="wss")
        assert config.websocket_url == "wss://gaggimate.local/ws"

    def test_http_base_url_generation(self):
        """Test HTTP base URL is generated correctly."""
        config = GaggimateConfig()
        assert config.http_base_url == "http://gaggimate.local"

    def test_http_base_url_https(self):
        """Test HTTP base URL with https when using wss."""
        config = GaggimateConfig(gaggimate_protocol="wss")
        assert config.http_base_url == "https://gaggimate.local"

    def test_temperature_clamping_max(self):
        """Test temperature is clamped to maximum limit."""
        config = GaggimateConfig()
        assert config.validate_temperature(100.0) == 96.0
        assert config.validate_temperature(97.0) == 96.0

    def test_temperature_clamping_min(self):
        """Test temperature is clamped to minimum limit."""
        config = GaggimateConfig()
        assert config.validate_temperature(50.0) == 60.0
        assert config.validate_temperature(55.0) == 60.0

    def test_temperature_within_range(self):
        """Test temperature within range is unchanged."""
        config = GaggimateConfig()
        assert config.validate_temperature(93.0) == 93.0
        assert config.validate_temperature(85.5) == 85.5

    def test_pressure_clamping_max(self):
        """Test pressure is clamped to maximum limit."""
        config = GaggimateConfig()
        assert config.validate_pressure(15.0) == 12.0
        assert config.validate_pressure(13.0) == 12.0

    def test_pressure_clamping_min(self):
        """Test pressure is clamped to minimum limit."""
        config = GaggimateConfig()
        assert config.validate_pressure(-1.0) == 0.0
        assert config.validate_pressure(-5.0) == 0.0

    def test_pressure_within_range(self):
        """Test pressure within range is unchanged."""
        config = GaggimateConfig()
        assert config.validate_pressure(9.0) == 9.0
        assert config.validate_pressure(6.5) == 6.5

    def test_profiles_dir_is_path(self):
        """Test profiles_dir is a Path object."""
        config = GaggimateConfig()
        assert isinstance(config.profiles_dir, Path)

    def test_custom_profiles_dir(self):
        """Test custom profiles directory can be set."""
        config = GaggimateConfig(profiles_dir=Path("/custom/path/profiles"))
        assert config.profiles_dir == Path("/custom/path/profiles")

    def test_log_level_default(self):
        """Test default log level is INFO."""
        config = GaggimateConfig()
        assert config.log_level == "INFO"

    def test_metrics_enabled_default(self):
        """Test metrics are enabled by default."""
        config = GaggimateConfig()
        assert config.enable_metrics is True


def test_host_env_var_documented_and_legacy_names(monkeypatch):
    """GAGGIMATE_HOST / GAGGIMATE_PROTOCOL work as documented (#12); the long names still do."""
    monkeypatch.setenv("GAGGIMATE_HOST", "192.168.0.10")
    monkeypatch.setenv("GAGGIMATE_PROTOCOL", "wss")
    config = GaggimateConfig()
    assert config.gaggimate_host == "192.168.0.10"
    assert config.gaggimate_protocol == "wss"

    monkeypatch.delenv("GAGGIMATE_HOST")
    monkeypatch.setenv("GAGGIMATE_GAGGIMATE_HOST", "192.168.0.11")
    assert GaggimateConfig().gaggimate_host == "192.168.0.11"


def test_host_env_var_from_dotenv_file(tmp_path, monkeypatch):
    """The documented name is also read from a .env file, as in the issue's reproduction."""
    monkeypatch.delenv("GAGGIMATE_HOST", raising=False)
    monkeypatch.delenv("GAGGIMATE_GAGGIMATE_HOST", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("GAGGIMATE_HOST=192.168.0.12\n")
    assert GaggimateConfig(_env_file=env_file).gaggimate_host == "192.168.0.12"
