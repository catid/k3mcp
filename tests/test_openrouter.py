from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from k3mcp.config import Settings
from k3mcp.openrouter import OpenRouterClient, OpenRouterError, _retry_delay, _usage_cost


def _success() -> dict[str, object]:
    return {
        "id": "gen-1",
        "model": "moonshotai/kimi-k3",
        "provider": "Moonshot AI",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "VERDICT: CORRECT"},
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost": 0.001,
            "completion_tokens_details": {"reasoning_tokens": 40},
        },
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 1.0), ("invalid", 1.0), ("nan", 1.0), ("-5", 1.0), ("45", 45.0)],
)
def test_retry_delay_validates_and_bounds_retry_after(value: str | None, expected: float) -> None:
    assert _retry_delay(1, value) == expected
    assert _retry_delay(1_000_000) == 30.0
    assert _retry_delay(1, "Thu, 01 Jan 1970 00:01:00 GMT", now=0) == 60.0
    assert _retry_delay(1, "301") is None


def test_usage_cost_overflow_is_ignored() -> None:
    assert _usage_cost(10**10_000) is None


@pytest.mark.asyncio
async def test_completion_payload_and_metadata() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers["Authorization"]
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=_success())

    settings = Settings.from_env({"OPENROUTER_API_KEY": "test-key"})
    client = OpenRouterClient(settings, transport=httpx.MockTransport(handler))
    try:
        assert client._http.timeout.connect == 30.0
        assert client._http.timeout.pool == 30.0
        assert client._http.timeout.write == 60.0
        assert client._http.timeout.read == 7_200.0
        result = await client.complete(system="system", user="user")
    finally:
        await client.aclose()

    assert seen["authorization"] == "Bearer test-key"
    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "moonshotai/kimi-k3"
    assert payload["max_tokens"] == 256_000
    assert payload["reasoning"] == {"effort": "max", "exclude": True}
    assert result.analysis == "VERDICT: CORRECT"
    assert result.reasoning_tokens == 40
    assert result.cost == 0.001
    assert result.truncated is False


@pytest.mark.asyncio
async def test_retries_transient_invalid_model_response() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                400,
                json={"error": {"message": "moonshotai/kimi-k3 is not a valid model ID"}},
            )
        return httpx.Response(200, json=_success())

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    settings = Settings.from_env({"OPENROUTER_API_KEY": "test-key"})
    client = OpenRouterClient(
        settings,
        transport=httpx.MockTransport(handler),
        sleep=sleep,
    )
    try:
        result = await client.complete(system="system", user="user")
    finally:
        await client.aclose()

    assert attempts == 2
    assert sleeps == [1]
    assert result.attempts == 2


@pytest.mark.asyncio
async def test_retries_any_server_error_and_respects_retry_after() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                529,
                headers={"Retry-After": "0.25"},
                json={"error": {"message": "provider overloaded"}},
            )
        return httpx.Response(200, json=_success())

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    settings = Settings.from_env({"OPENROUTER_API_KEY": "test-key"})
    client = OpenRouterClient(
        settings,
        transport=httpx.MockTransport(handler),
        sleep=sleep,
    )
    try:
        result = await client.complete(system="system", user="user")
    finally:
        await client.aclose()

    assert attempts == 2
    assert sleeps == [0.25]
    assert result.attempts == 2


@pytest.mark.asyncio
async def test_default_retry_budget_survives_six_provider_rate_limits() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts <= 6:
            return httpx.Response(429, json={"error": {"message": "provider overloaded"}})
        return httpx.Response(200, json=_success())

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    settings = Settings.from_env({"OPENROUTER_API_KEY": "test-key"})
    client = OpenRouterClient(
        settings,
        transport=httpx.MockTransport(handler),
        sleep=sleep,
    )
    try:
        result = await client.complete(system="system", user="user")
    finally:
        await client.aclose()

    assert attempts == 7
    assert sleeps == [1, 2, 4, 8, 16, 30]
    assert result.attempts == 7


@pytest.mark.asyncio
async def test_retries_empty_successful_response() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        payload = _success()
        if attempts == 1:
            payload["choices"][0]["message"]["content"] = None  # type: ignore[index]
        return httpx.Response(200, json=payload)

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    settings = Settings.from_env({"OPENROUTER_API_KEY": "test-key"})
    client = OpenRouterClient(
        settings,
        transport=httpx.MockTransport(handler),
        sleep=sleep,
    )
    try:
        result = await client.complete(system="system", user="user")
    finally:
        await client.aclose()

    assert attempts == 2
    assert sleeps == [1]
    assert result.analysis == "VERDICT: CORRECT"


@pytest.mark.asyncio
async def test_allows_only_one_potentially_billable_malformed_success_retry() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        payload = _success()
        payload["choices"][0]["message"]["content"] = None  # type: ignore[index]
        return httpx.Response(200, json=payload)

    async def sleep(_delay: float) -> None:
        return None

    settings = Settings.from_env({"OPENROUTER_API_KEY": "test-key"})
    client = OpenRouterClient(
        settings,
        transport=httpx.MockTransport(handler),
        sleep=sleep,
    )
    try:
        with pytest.raises(OpenRouterError, match="empty completion"):
            await client.complete(system="system", user="user")
    finally:
        await client.aclose()

    assert attempts == 2


@pytest.mark.asyncio
async def test_retries_malformed_usage_shape() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        payload = _success()
        payload["usage"] = ["not", "an", "object"]
        return httpx.Response(200, json=payload)

    async def sleep(_delay: float) -> None:
        return None

    settings = Settings.from_env({"OPENROUTER_API_KEY": "test-key"})
    client = OpenRouterClient(
        settings,
        transport=httpx.MockTransport(handler),
        sleep=sleep,
    )
    try:
        result = await client.complete(system="system", user="user")
    finally:
        await client.aclose()

    assert attempts == 1
    assert result.analysis == "VERDICT: CORRECT"
    assert result.total_tokens == 0


@pytest.mark.asyncio
async def test_does_not_retry_ambiguous_read_timeout() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("provider may still be generating", request=request)

    settings = Settings.from_env({"OPENROUTER_API_KEY": "test-key"})
    client = OpenRouterClient(settings, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(OpenRouterError, match="network error: ReadTimeout"):
            await client.complete(system="system", user="user")
    finally:
        await client.aclose()

    assert attempts == 1


@pytest.mark.asyncio
async def test_allows_only_one_ambiguous_network_retry() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadError("connection dropped", request=request)

    async def sleep(_delay: float) -> None:
        return None

    settings = Settings.from_env({"OPENROUTER_API_KEY": "test-key"})
    client = OpenRouterClient(
        settings,
        transport=httpx.MockTransport(handler),
        sleep=sleep,
    )
    try:
        with pytest.raises(OpenRouterError, match="network error: ReadError"):
            await client.complete(system="system", user="user")
    finally:
        await client.aclose()

    assert attempts == 2


@pytest.mark.asyncio
async def test_connection_failures_receive_full_retry_budget() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 8:
            raise httpx.ConnectError("cannot connect", request=request)
        return httpx.Response(200, json=_success())

    async def sleep(_delay: float) -> None:
        return None

    settings = Settings.from_env({"OPENROUTER_API_KEY": "test-key"})
    client = OpenRouterClient(
        settings,
        transport=httpx.MockTransport(handler),
        sleep=sleep,
    )
    try:
        result = await client.complete(system="system", user="user")
    finally:
        await client.aclose()

    assert attempts == 8
    assert result.attempts == 8


@pytest.mark.asyncio
async def test_total_deadline_bounds_all_attempts() -> None:
    attempts = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0.05)
        return httpx.Response(200, json=_success())

    settings = Settings.from_env(
        {
            "OPENROUTER_API_KEY": "test-key",
            "K3MCP_TOTAL_TIMEOUT_SECONDS": "0.01",
        }
    )
    client = OpenRouterClient(settings, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(OpenRouterError, match=r"0\.01-second total timeout"):
            await client.complete(system="system", user="user")
    finally:
        await client.aclose()

    assert attempts == 1


@pytest.mark.asyncio
async def test_retries_retryable_error_inside_http_200() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                200,
                headers={"Retry-After": "0.5"},
                json={"error": {"code": 503, "message": "provider unavailable"}},
            )
        return httpx.Response(200, json=_success())

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    settings = Settings.from_env({"OPENROUTER_API_KEY": "test-key"})
    client = OpenRouterClient(
        settings,
        transport=httpx.MockTransport(handler),
        sleep=sleep,
    )
    try:
        result = await client.complete(system="system", user="user")
    finally:
        await client.aclose()

    assert attempts == 2
    assert sleeps == [0.5]
    assert result.analysis == "VERDICT: CORRECT"


@pytest.mark.asyncio
async def test_does_not_retry_nonretryable_error_inside_http_200() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            json={"error": {"code": 402, "message": "insufficient credits"}},
        )

    settings = Settings.from_env({"OPENROUTER_API_KEY": "test-key"})
    client = OpenRouterClient(settings, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(OpenRouterError, match="in-body error 402: insufficient credits"):
            await client.complete(system="system", user="user")
    finally:
        await client.aclose()

    assert attempts == 1


@pytest.mark.asyncio
async def test_in_body_error_code_overflow_is_safely_reported() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            content=b'{"error":{"code":1e400,"message":"bad provider response"}}',
        )

    settings = Settings.from_env({"OPENROUTER_API_KEY": "test-key"})
    client = OpenRouterClient(settings, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(
            OpenRouterError,
            match="in-body error unknown: bad provider response",
        ):
            await client.complete(system="system", user="user")
    finally:
        await client.aclose()

    assert attempts == 1


@pytest.mark.asyncio
async def test_does_not_retry_empty_completion_caused_by_token_limit() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        payload = _success()
        payload["choices"][0]["finish_reason"] = "length"  # type: ignore[index]
        payload["choices"][0]["message"]["content"] = None  # type: ignore[index]
        return httpx.Response(200, json=payload)

    settings = Settings.from_env({"OPENROUTER_API_KEY": "test-key"})
    client = OpenRouterClient(settings, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(OpenRouterError, match="exhausted max_tokens"):
            await client.complete(system="system", user="user")
    finally:
        await client.aclose()

    assert attempts == 1


def test_completion_result_marks_length_truncation() -> None:
    payload = _success()
    payload["choices"][0]["finish_reason"] = "length"  # type: ignore[index]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async def run() -> bool:
        settings = Settings.from_env({"OPENROUTER_API_KEY": "test-key"})
        client = OpenRouterClient(settings, transport=httpx.MockTransport(handler))
        try:
            result = await client.complete(system="system", user="user")
            return result.truncated
        finally:
            await client.aclose()

    assert asyncio.run(run()) is True


@pytest.mark.asyncio
async def test_reports_server_error_after_retry_exhaustion() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, json={"error": {"message": "still unavailable"}})

    async def sleep(_delay: float) -> None:
        return None

    settings = Settings.from_env({"OPENROUTER_API_KEY": "test-key"})
    client = OpenRouterClient(
        settings,
        transport=httpx.MockTransport(handler),
        sleep=sleep,
    )
    try:
        with pytest.raises(OpenRouterError, match="HTTP 503: still unavailable"):
            await client.complete(system="system", user="user")
    finally:
        await client.aclose()

    assert attempts == 8


@pytest.mark.asyncio
async def test_input_limit_is_checked_before_request() -> None:
    settings = Settings.from_env({"OPENROUTER_API_KEY": "test-key", "K3MCP_MAX_INPUT_CHARS": "5"})
    client = OpenRouterClient(settings, transport=httpx.MockTransport(lambda _: None))
    try:
        with pytest.raises(OpenRouterError, match="configured limit"):
            await client.complete(system="123", user="456")
    finally:
        await client.aclose()
