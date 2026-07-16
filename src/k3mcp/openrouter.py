"""Small, defensive async client for OpenRouter chat completions."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC
from email.utils import parsedate_to_datetime
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


_MAX_AMBIGUOUS_ATTEMPTS = 2
_MAX_RETRY_AFTER_SECONDS = 300.0


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


def _retry_delay(
    attempt: int, retry_after: str | None = None, *, now: float | None = None
) -> float | None:
    fallback = 30 if attempt >= 6 else 2 ** (attempt - 1)
    if retry_after is None:
        return float(fallback)
    try:
        requested = float(retry_after)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(retry_after)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            requested = retry_at.timestamp() - (time.time() if now is None else now)
        except (OverflowError, TypeError, ValueError):
            return float(fallback)
        requested = max(requested, 0.0)
    if not math.isfinite(requested) or requested < 0:
        return float(fallback)
    if requested > _MAX_RETRY_AFTER_SECONDS:
        return None
    return requested


def _usage_int(value: Any) -> int:
    try:
        parsed = int(value or 0)
    except (OverflowError, TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _usage_cost(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


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
    except (OverflowError, TypeError, ValueError):
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
            timeout=httpx.Timeout(
                settings.timeout_seconds,
                connect=min(settings.timeout_seconds, 30.0),
                pool=min(settings.timeout_seconds, 30.0),
                write=min(settings.timeout_seconds, 60.0),
            ),
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
        deadline = started + self.settings.total_timeout_seconds
        total_timeout_error = (
            f"OpenRouter request exceeded the configured "
            f"{self.settings.total_timeout_seconds:g}-second total timeout"
        )

        async def wait_before_retry(delay: float) -> None:
            if delay >= deadline - time.perf_counter():
                raise OpenRouterError(total_timeout_error)
            await self._sleep(delay)

        last_error = "request failed"
        total_attempts = self.settings.max_retries + 1
        ambiguous_outcome_attempts = 0
        for attempt in range(1, total_attempts + 1):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise OpenRouterError(total_timeout_error)
            try:
                async with asyncio.timeout(remaining):
                    response = await self._http.post("/chat/completions", json=payload)
            except TimeoutError as exc:
                raise OpenRouterError(total_timeout_error) from exc
            except httpx.RequestError as exc:
                last_error = f"network error: {exc.__class__.__name__}"
                is_pre_request = isinstance(
                    exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
                )
                if not is_pre_request:
                    ambiguous_outcome_attempts += 1
                retryable = is_pre_request or (
                    not isinstance(exc, httpx.ReadTimeout)
                    and ambiguous_outcome_attempts < _MAX_AMBIGUOUS_ATTEMPTS
                )
                if retryable and attempt < total_attempts:
                    await wait_before_retry(_retry_delay(attempt))
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
                        delay = _retry_delay(attempt, response.headers.get("Retry-After"))
                        if delay is not None:
                            await wait_before_retry(delay)
                            continue
                    raise
                except OpenRouterError as exc:
                    # A 2xx with no usable assistant message is still an upstream failure.
                    # Retrying covers truncated reasoning-only and malformed provider responses.
                    last_error = str(exc)
                    ambiguous_outcome_attempts += 1
                    if (
                        attempt < total_attempts
                        and ambiguous_outcome_attempts < _MAX_AMBIGUOUS_ATTEMPTS
                    ):
                        await wait_before_retry(_retry_delay(attempt))
                        continue
                    raise

            last_error = _error_message(response)
            retryable = _is_retryable_code(response.status_code, last_error)
            if retryable and attempt < total_attempts:
                retry_after = response.headers.get("Retry-After")
                delay = _retry_delay(attempt, retry_after)
                if delay is not None:
                    await wait_before_retry(delay)
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

        except OpenRouterError:
            raise
        except (AttributeError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError("OpenRouter returned a malformed completion response") from exc

        # A usable answer is more valuable than perfect optional accounting metadata. Never
        # regenerate a potentially expensive completion merely because its usage shape is bad.
        usage_value = payload.get("usage") or {}
        usage = usage_value if isinstance(usage_value, dict) else {}
        details_value = usage.get("completion_tokens_details") or {}
        completion_details = details_value if isinstance(details_value, dict) else {}
        prompt_tokens = _usage_int(usage.get("prompt_tokens"))
        completion_tokens = _usage_int(usage.get("completion_tokens"))
        reasoning_tokens = _usage_int(completion_details.get("reasoning_tokens"))
        total_tokens = _usage_int(usage.get("total_tokens"))
        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens
        cost = _usage_cost(usage.get("cost"))

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
