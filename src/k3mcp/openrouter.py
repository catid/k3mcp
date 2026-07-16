"""Small, defensive async client for OpenRouter chat completions."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from k3mcp.config import Settings


class OpenRouterError(RuntimeError):
    """A safe-to-display OpenRouter request or response failure."""


class OpenRouterResponseError(OpenRouterError):
    """An error envelope returned by OpenRouter, possibly inside an HTTP 200."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """A model answer plus operational metadata, excluding private reasoning."""

    analysis: str
    model: str
    provider: str | None
    request_id: str | None
    finish_reason: str | None
    truncated: bool
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    cost: float | None
    latency_seconds: float
    attempts: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis,
            "model": self.model,
            "provider": self.provider,
            "request_id": self.request_id,
            "finish_reason": self.finish_reason,
            "truncated": self.truncated,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "total_tokens": self.total_tokens,
                "cost_usd": self.cost,
            },
            "latency_seconds": self.latency_seconds,
            "attempts": self.attempts,
        }


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:1_000] if text else "no error body"

    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"][:1_000]
    if isinstance(error, str):
        return error[:1_000]
    return "unrecognized error response"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = [
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return "\n".join(chunks).strip()
    return ""


def _is_retryable_code(code: int | None, message: str) -> bool:
    if code in {408, 409, 425, 429} or (code is not None and 500 <= code < 600):
        return True
    return code == 400 and "not a valid model id" in message.lower()


def _in_body_error(payload: dict[str, Any]) -> OpenRouterResponseError | None:
    error = payload.get("error")
    if error is None:
        return None
    if isinstance(error, dict):
        message_value = error.get("message")
        message = message_value if isinstance(message_value, str) else "unrecognized error response"
        code_value = error.get("code")
    else:
        message = error if isinstance(error, str) else "unrecognized error response"
        code_value = None
    try:
        code = int(code_value) if code_value is not None else None
    except (TypeError, ValueError):
        code = None
    label = str(code) if code is not None else "unknown"
    return OpenRouterResponseError(
        f"OpenRouter in-body error {label}: {message[:1_000]}",
        retryable=_is_retryable_code(code, message),
    )


class OpenRouterClient:
    """Call the configured model without exposing credentials or chain of thought."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.settings = settings
        self._sleep = sleep
        headers = {
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
            "X-Title": settings.app_name,
        }
        if settings.site_url:
            headers["HTTP-Referer"] = settings.site_url
        self._http = httpx.AsyncClient(
            base_url=settings.base_url,
            headers=headers,
            timeout=httpx.Timeout(settings.timeout_seconds),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def complete(self, *, system: str, user: str) -> CompletionResult:
        input_chars = len(system) + len(user)
        if input_chars > self.settings.max_input_chars:
            raise OpenRouterError(
                f"request is {input_chars:,} characters; configured limit is "
                f"{self.settings.max_input_chars:,} (K3MCP_MAX_INPUT_CHARS)"
            )

        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.settings.max_tokens,
            "reasoning": {
                "effort": self.settings.reasoning_effort,
                "exclude": True,
            },
        }

        started = time.perf_counter()
        last_error = "request failed"
        total_attempts = self.settings.max_retries + 1
        for attempt in range(1, total_attempts + 1):
            try:
                response = await self._http.post("/chat/completions", json=payload)
            except httpx.RequestError as exc:
                last_error = f"network error: {exc.__class__.__name__}"
                if attempt < total_attempts:
                    await self._sleep(min(2 ** (attempt - 1), 8))
                    continue
                raise OpenRouterError(last_error) from exc

            if response.is_success:
                try:
                    return self._parse_response(
                        response,
                        latency_seconds=time.perf_counter() - started,
                        attempts=attempt,
                    )
                except OpenRouterResponseError as exc:
                    last_error = str(exc)
                    if exc.retryable and attempt < total_attempts:
                        await self._sleep(min(2 ** (attempt - 1), 8))
                        continue
                    raise
                except OpenRouterError as exc:
                    # A 2xx with no usable assistant message is still an upstream failure.
                    # Retrying covers truncated reasoning-only and malformed provider responses.
                    last_error = str(exc)
                    if attempt < total_attempts:
                        await self._sleep(min(2 ** (attempt - 1), 8))
                        continue
                    raise

            last_error = _error_message(response)
            retryable = _is_retryable_code(response.status_code, last_error)
            if retryable and attempt < total_attempts:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after is not None else 2 ** (attempt - 1)
                except ValueError:
                    delay = 2 ** (attempt - 1)
                await self._sleep(min(max(delay, 0), 30))
                continue
            raise OpenRouterError(f"OpenRouter HTTP {response.status_code}: {last_error}")

        raise OpenRouterError(last_error)  # pragma: no cover - loop always returns or raises

    def _parse_response(
        self, response: httpx.Response, *, latency_seconds: float, attempts: int
    ) -> CompletionResult:
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise TypeError("completion payload is not an object")
            in_body_error = _in_body_error(payload)
            if in_body_error is not None:
                raise in_body_error
            choice = payload["choices"][0]
            if not isinstance(choice, dict):
                raise TypeError("completion choice is not an object")
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError("completion message is not an object")
            finish_reason_value = choice.get("finish_reason")
            finish_reason = str(finish_reason_value) if finish_reason_value is not None else None
            content = _content_text(message.get("content"))
            if not content:
                if finish_reason == "length":
                    raise OpenRouterResponseError(
                        "OpenRouter exhausted max_tokens before producing a final answer; "
                        "increase K3MCP_MAX_TOKENS or reduce the submitted context",
                        retryable=False,
                    )
                raise OpenRouterError("OpenRouter returned an empty completion")

            usage = payload.get("usage") or {}
            if not isinstance(usage, dict):
                raise TypeError("usage is not an object")
            completion_details = usage.get("completion_tokens_details") or {}
            if not isinstance(completion_details, dict):
                raise TypeError("completion token details are not an object")
            cost_value = usage.get("cost")
            cost = float(cost_value) if cost_value is not None else None
            if cost is not None and not math.isfinite(cost):
                raise ValueError("cost is not finite")

            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or 0)
        except OpenRouterError:
            raise
        except (AttributeError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError("OpenRouter returned a malformed completion response") from exc

        return CompletionResult(
            analysis=content,
            model=str(payload.get("model") or self.settings.model),
            provider=payload.get("provider"),
            request_id=payload.get("id"),
            finish_reason=finish_reason,
            truncated=finish_reason == "length",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            cost=cost,
            latency_seconds=round(latency_seconds, 3),
            attempts=attempts,
        )
