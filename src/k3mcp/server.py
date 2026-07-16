"""MCP tool definitions and stdio entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from mcp.types import ToolAnnotations
from pydantic import Field

from k3mcp.config import Settings
from k3mcp.openrouter import OpenRouterClient
from k3mcp.prompts import (
    ALGORITHM_SYSTEM,
    CODE_REVIEW_SYSTEM,
    PLANNING_SYSTEM,
    algorithm_prompt,
    code_review_prompt,
    planning_prompt,
)

SERVER_INSTRUCTIONS = """Use these read-only tools to get a Kimi K3 second opinion on algorithms,
code or diffs, and project plans. Supply complete requirements and the relevant evidence because
Kimi cannot read the workspace. Treat its response as advisory: verify findings before changing
code. Prefer review_algorithm for proofs, invariants, complexity, and counterexamples;
review_code for concrete implementation defects; and plan_project before substantial multi-stage
work. Calls use OpenRouter, incur usage cost, and may take several minutes."""

READ_ONLY_EXTERNAL = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=True,
)


@dataclass(slots=True)
class AppContext:
    client: OpenRouterClient


@asynccontextmanager
async def app_lifespan(_server: FastMCP) -> AsyncIterator[AppContext]:
    settings = Settings.from_env()
    client = OpenRouterClient(settings)
    try:
        yield AppContext(client=client)
    finally:
        await client.aclose()


mcp = FastMCP(
    name="Kimi K3 Review and Planning",
    instructions=SERVER_INSTRUCTIONS,
    lifespan=app_lifespan,
)


def _client(ctx: Context[ServerSession, AppContext]) -> OpenRouterClient:
    return ctx.request_context.lifespan_context.client


@mcp.tool(
    title="Review an algorithm with Kimi K3",
    annotations=READ_ONLY_EXTERNAL,
    structured_output=True,
)
async def review_algorithm(
    algorithm: Annotated[
        str,
        Field(
            description=(
                "Algorithm description, pseudocode, or implementation to review. "
                "Include all relevant details."
            )
        ),
    ],
    requirements: Annotated[
        str,
        Field(description="Required behavior, constraints, and correctness criteria."),
    ],
    ctx: Context[ServerSession, AppContext],
    context: Annotated[
        str,
        Field(
            description="Optional surrounding architecture, data assumptions, or known tradeoffs."
        ),
    ] = "",
    focus: Annotated[
        str,
        Field(
            description=(
                "Optional specific concern such as proof, complexity, numerical stability, "
                "or concurrency."
            )
        ),
    ] = "",
) -> dict[str, Any]:
    """Challenge an algorithm's correctness, invariants, edge cases, and complexity."""
    await ctx.info("Asking Kimi K3 to review the algorithm")
    result = await _client(ctx).complete(
        system=ALGORITHM_SYSTEM,
        user=algorithm_prompt(
            algorithm=algorithm,
            requirements=requirements,
            context=context,
            focus=focus,
        ),
    )
    return result.as_dict()


@mcp.tool(
    title="Review code with Kimi K3",
    annotations=READ_ONLY_EXTERNAL,
    structured_output=True,
)
async def review_code(
    code: Annotated[
        str,
        Field(
            description=(
                "Relevant code, patch, or diff. Include enough surrounding code to validate "
                "findings."
            )
        ),
    ],
    requirements: Annotated[
        str,
        Field(description="Intended behavior and review acceptance criteria."),
    ],
    ctx: Context[ServerSession, AppContext],
    context: Annotated[
        str,
        Field(
            description="Optional language, runtime, architecture, tests, and surrounding behavior."
        ),
    ] = "",
    focus: Annotated[
        str,
        Field(
            description=(
                "Optional review focus such as correctness, security, performance, or concurrency."
            )
        ),
    ] = "",
) -> dict[str, Any]:
    """Find concrete correctness, security, reliability, and performance defects."""
    await ctx.info("Asking Kimi K3 to review the code")
    result = await _client(ctx).complete(
        system=CODE_REVIEW_SYSTEM,
        user=code_review_prompt(
            code=code,
            requirements=requirements,
            context=context,
            focus=focus,
        ),
    )
    return result.as_dict()


@mcp.tool(
    title="Plan a project with Kimi K3",
    annotations=READ_ONLY_EXTERNAL,
    structured_output=True,
)
async def plan_project(
    objective: Annotated[
        str,
        Field(description="Concrete outcome the project must achieve."),
    ],
    project_context: Annotated[
        str,
        Field(
            description="Current system, repository evidence, relevant components, and known state."
        ),
    ],
    ctx: Context[ServerSession, AppContext],
    constraints: Annotated[
        str,
        Field(
            description=(
                "Optional technical, compatibility, operational, budget, or scope constraints."
            )
        ),
    ] = "",
    decisions: Annotated[
        str,
        Field(
            description=(
                "Optional decisions already made that the plan should preserve or explicitly "
                "challenge."
            )
        ),
    ] = "",
) -> dict[str, Any]:
    """Create a project plan with dependencies, verification, risks, and rollback."""
    await ctx.info("Asking Kimi K3 to produce a project plan")
    result = await _client(ctx).complete(
        system=PLANNING_SYSTEM,
        user=planning_prompt(
            objective=objective,
            project_context=project_context,
            constraints=constraints,
            decisions=decisions,
        ),
    )
    return result.as_dict()


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
