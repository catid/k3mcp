from __future__ import annotations

import os
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.asyncio
async def test_stdio_server_lists_expected_read_only_tools() -> None:
    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = "protocol-test-key"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "k3mcp"],
        env=env,
    )

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        initialized = await session.initialize()
        response = await session.list_tools()

    assert initialized.serverInfo.name == "Kimi K3 Review and Planning"
    tools = {tool.name: tool for tool in response.tools}
    assert set(tools) == {"review_algorithm", "review_code", "plan_project"}
    for tool in tools.values():
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
        assert tool.outputSchema is not None
