import pytest

from k3mcp.config import Settings, SettingsError


def test_settings_defaults() -> None:
    settings = Settings.from_env({"OPENROUTER_API_KEY": "secret"})

    assert settings.api_key == "secret"
    assert settings.model == "moonshotai/kimi-k3"
    assert settings.max_tokens == 256_000
    assert settings.max_retries == 7
    assert settings.timeout_seconds == 7_200
    assert settings.total_timeout_seconds == 15_000
    assert "secret" not in repr(settings)


def test_settings_require_key() -> None:
    with pytest.raises(SettingsError, match="OPENROUTER_API_KEY is required"):
        Settings.from_env({})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("K3MCP_MAX_TOKENS", "0"),
        ("K3MCP_MAX_INPUT_CHARS", "many"),
        ("K3MCP_TIMEOUT_SECONDS", "-1"),
        ("K3MCP_TIMEOUT_SECONDS", "nan"),
        ("K3MCP_TIMEOUT_SECONDS", "inf"),
        ("K3MCP_TOTAL_TIMEOUT_SECONDS", "0"),
        ("K3MCP_MAX_ATTEMPTS", "0"),
    ],
)
def test_settings_reject_invalid_positive_values(name: str, value: str) -> None:
    with pytest.raises(SettingsError):
        Settings.from_env({"OPENROUTER_API_KEY": "secret", name: value})
