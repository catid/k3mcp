# Kimi K3 MCP

A small, read-only [Model Context Protocol](https://modelcontextprotocol.io/) server that gives
Codex and other MCP clients a Kimi K3 second opinion through OpenRouter.

It exposes three tools:

- `review_algorithm`: challenge correctness, invariants, edge cases, counterexamples, and complexity.
- `review_code`: find concrete correctness, security, reliability, and performance defects.
- `plan_project`: produce an implementation-ready plan with dependencies, verification, risks, and rollback.

The server cannot read or modify local files. The MCP client chooses what code and context to send.
Submitted content is treated as untrusted data, and Kimi's private reasoning is excluded from tool
results. Responses include token usage, cost, latency, provider, and retry metadata.

The tools are strictly opt-in. Server instructions tell the MCP client not to call them for routine
reviews or planning; explicitly mention Kimi, K3, the Kimi MCP, or a tool name when you want a call.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- An `OPENROUTER_API_KEY` with access to `moonshotai/kimi-k3`

## Install

```bash
git clone git@github.com:catid/k3mcp.git
cd k3mcp
uv sync --frozen
```

Do not put the OpenRouter key in this repository or in `config.toml`. Export it in the environment
that starts Codex:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

## Configure Codex

Add the server to `~/.codex/config.toml`. Adjust the absolute path to the clone:

```toml
[mcp_servers.kimi_k3]
command = "/absolute/path/to/k3mcp/.venv/bin/k3mcp"
env_vars = ["OPENROUTER_API_KEY"]
enabled = true
required = false
startup_timeout_sec = 20
tool_timeout_sec = 600
default_tools_approval_mode = "approve"
```

Restart Codex after changing its MCP configuration. In the TUI, `/mcp` shows whether the server
and its tools loaded successfully.

## Other MCP clients

The server uses stdio transport:

```bash
OPENROUTER_API_KEY=... uv run k3mcp
```

The process writes MCP protocol messages to stdout. Application logs must go to stderr.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | required | OpenRouter credential |
| `OPENROUTER_MODEL` | `moonshotai/kimi-k3` | Model slug |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | API root |
| `K3MCP_MAX_TOKENS` | `16000` | Maximum completion tokens per call |
| `K3MCP_MAX_INPUT_CHARS` | `400000` | Combined system and user prompt limit |
| `K3MCP_TIMEOUT_SECONDS` | `600` | Per-attempt HTTP timeout |
| `K3MCP_MAX_ATTEMPTS` | `3` | Total attempts for retryable failures |
| `K3MCP_REASONING_EFFORT` | `max` | OpenRouter reasoning effort |
| `OPENROUTER_APP_NAME` | `k3mcp` | OpenRouter attribution title |
| `OPENROUTER_SITE_URL` | repository URL | OpenRouter attribution referrer |

Kimi K3 reasoning is mandatory on OpenRouter. The server requests maximum effort and excludes the
reasoning trace while retaining its token count in usage metadata.

## Development

```bash
uv sync --all-groups
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Tests use a mocked OpenRouter transport and a real MCP stdio handshake; they do not spend API
credits. Retry tests cover network failures, rate limits, every `5xx` status, malformed or empty
successful responses, and OpenRouter's occasional transient `invalid model ID` response for an
otherwise live model slug. OpenRouter can report provider failures inside HTTP `200` responses;
those are classified by their embedded status so billing, authentication, guardrail, and token-cap
errors are not retried. Retries use bounded exponential backoff and honor `Retry-After`.

GitHub Actions runs linting, formatting, tests, the stdio protocol handshake, and package builds on
Python 3.11, 3.12, and 3.13. Workflow actions are pinned to immutable commit SHAs.

## Security and cost

- Tool calls send the supplied code and project context to OpenRouter and its selected provider.
- The server is read-only but calls a metered external API (`openWorldHint=true`).
- Inputs and outputs are bounded by environment-configurable limits.
- Errors never include request headers or the API key.
- Model output is advisory. Verify findings and plans before applying changes.
