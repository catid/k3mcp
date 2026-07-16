"""Environment-backed server configuration."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field


class SettingsError(ValueError):
    """Raised when environment configuration is missing or invalid."""


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise SettingsError(f"{name} must be positive, got {value}")
    return value


def _positive_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise SettingsError(f"{name} must be a positive finite number, got {raw!r}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    api_key: str = field(repr=False)
    model: str = "moonshotai/kimi-k3"
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: int = 256_000
    max_input_chars: int = 400_000
    timeout_seconds: float = 7_200.0
    total_timeout_seconds: float = 15_000.0
    max_retries: int = 7
    reasoning_effort: str = "max"
    app_name: str = "k3mcp"
    site_url: str = "https://github.com/catid/k3mcp"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source = os.environ if env is None else env
        api_key = source.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise SettingsError(
                "OPENROUTER_API_KEY is required; forward it to the MCP process with env_vars"
            )

        model = source.get("OPENROUTER_MODEL", "moonshotai/kimi-k3").strip()
        if not model:
            raise SettingsError("OPENROUTER_MODEL cannot be empty")

        base_url = source.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
        if not base_url.startswith(("https://", "http://")):
            raise SettingsError("OPENROUTER_BASE_URL must start with http:// or https://")

        retries = _positive_int(source, "K3MCP_MAX_ATTEMPTS", 8) - 1
        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url.rstrip("/"),
            max_tokens=_positive_int(source, "K3MCP_MAX_TOKENS", 256_000),
            max_input_chars=_positive_int(source, "K3MCP_MAX_INPUT_CHARS", 400_000),
            timeout_seconds=_positive_float(source, "K3MCP_TIMEOUT_SECONDS", 7_200.0),
            total_timeout_seconds=_positive_float(source, "K3MCP_TOTAL_TIMEOUT_SECONDS", 15_000.0),
            max_retries=retries,
            reasoning_effort=source.get("K3MCP_REASONING_EFFORT", "max").strip() or "max",
            app_name=source.get("OPENROUTER_APP_NAME", "k3mcp").strip() or "k3mcp",
            site_url=source.get("OPENROUTER_SITE_URL", "https://github.com/catid/k3mcp").strip(),
        )
