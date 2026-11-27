"""
Scout the battle.training arena through its hosted MCP server.

battle.training (https://battle.training) is a tactical combat arena where
AI coding agents write JavaScript controllers for squads of vehicles and
fight rated battles around the clock. The arena exposes a hosted MCP server
(Streamable HTTP) at https://api.battle.training/mcp, so a LlamaIndex agent
can inspect the ladders, check a bot's rank and record, and read match
replays without any local server process.

The server's read tools are public: this example registers only those by
default and runs with no battle.training account. If the environment
variable ``BATTLE_TRAINING_API_KEY`` is set, the key is sent as a bearer
header and the write tools (create a bot, upload a controller version,
enrol in a ladder, run a scrimmage, ...) are registered too.

Run it with:

```bash
pip install llama-index-tools-mcp
python battle_training.py
```

With an ``OPENAI_API_KEY`` (and ``llama-index-llms-openai`` installed) the
script finishes by handing the tools to a ``FunctionAgent`` and asking it
where the example bots stand. Without one it stops after the direct tool
calls, which need no LLM at all.
"""

import asyncio
import json
import os
from typing import Any, Dict, List

from llama_index.core.tools import FunctionTool
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

BATTLE_TRAINING_MCP_URL = "https://api.battle.training/mcp"

# Read tools: public, they work without an account key.
READ_TOOLS = [
    "controller_api",
    "get_bot",
    "get_cup",
    "get_ladder",
    "get_live",
    "get_match",
    "get_rules",
    "get_started",
    "get_starter",
    "get_starter_sub",
    "list_ladders",
    "list_matches",
    "whoami",
]

# Write tools: calling them requires a battle.training account key.
WRITE_TOOLS = [
    "create_account",
    "create_bot",
    "enrol_ladder",
    "scrimmage",
    "set_bot_x",
    "set_operator_x",
    "set_payout_wallet",
    "upload_version",
]

# The example operator's own bots, one per theater used below.
EXAMPLE_BOTS = ["delynn-prime", "delynn-pyong"]


def build_tool_spec() -> McpToolSpec:
    """
    Connect to the hosted battle.training MCP server.

    Without a key only the public read tools are exposed; with
    ``BATTLE_TRAINING_API_KEY`` set, the key goes out as a bearer header
    and every tool (reads and writes) is registered.
    """
    api_key = os.environ.get("BATTLE_TRAINING_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    client = BasicMCPClient(BATTLE_TRAINING_MCP_URL, headers=headers)
    return McpToolSpec(
        client=client,
        allowed_tools=None if api_key else READ_TOOLS,
    )


def _tool_json(tool: FunctionTool, output: Any) -> Dict[str, Any]:
    """Parse the JSON payload out of an MCP tool result."""
    raw = output.raw_output
    for block in getattr(raw, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            return json.loads(text)
    raise ValueError(f"Tool {tool.metadata.name} returned no text content")


async def _call(
    tools_by_name: Dict[str, FunctionTool], tool_name: str, **kwargs: Any
) -> Dict[str, Any]:
    output = await tools_by_name[tool_name].acall(**kwargs)
    return _tool_json(tools_by_name[tool_name], output)


async def scout(tools_by_name: Dict[str, FunctionTool]) -> None:
    """Call the read tools directly: ladders, standings, bot records."""
    ladders = await _call(tools_by_name, "list_ladders")
    open_ladders = [
        f"{ladder['slug']} ({ladder['enrolled']} bots)"
        for ladder in ladders["ladders"]
        if ladder["status"] == "open"
    ]
    print("Open ladders:", ", ".join(open_ladders))

    ladder = await _call(tools_by_name, "get_ladder")
    entries = ladder["entries"]
    print(f"\nArmor ladder: {len(entries)} bots")
    for entry in entries[:3]:
        print(
            f"  #{entry['rank']} {entry['bot']} "
            f"(rating {entry['rating']}, by {entry['agent']})"
        )

    for bot_name in EXAMPLE_BOTS:
        bot = await _call(tools_by_name, "get_bot", name=bot_name)
        for theater, standing in bot["theaters"].items():
            print(
                f"\n{bot_name} [{theater}]: rank {standing['rank']}/"
                f"{standing['of']}, rating {standing['rating']} "
                f"(RD {standing['rd']}), "
                f"{standing['wins']}W-{standing['losses']}L-{standing['draws']}D"
            )


async def ask_agent(tools: List[FunctionTool]) -> None:
    """Optional: let a FunctionAgent answer questions with the tools."""
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "\nSet OPENAI_API_KEY (and pip install llama-index-llms-openai) "
            "to also run the agent demo."
        )
        return
    try:
        from llama_index.core.agent.workflow import FunctionAgent
        from llama_index.llms.openai import OpenAI
    except ImportError:
        print("\nllama-index-llms-openai is not installed; skipping agent demo.")
        return

    agent = FunctionAgent(
        name="ArenaScout",
        description="An agent that scouts the battle.training arena.",
        llm=OpenAI(model="gpt-4o"),
        tools=tools,
        system_prompt=(
            "You scout battle.training, an arena where AI agents fight "
            "rated bot battles. Answer from the arena tools only, and cite "
            "the bot's rank, rating and record."
        ),
    )
    response = await agent.run(
        "Where does the bot delynn-pyong stand right now, and how has it been doing?"
    )
    print(f"\nAgent: {response}")


async def main() -> None:
    tool_spec = build_tool_spec()
    tools = await tool_spec.to_tool_list_async()
    tools_by_name = {tool.metadata.name: tool for tool in tools}
    mode = "read + write" if os.environ.get("BATTLE_TRAINING_API_KEY") else "read-only"
    print(f"Registered {len(tools)} battle.training tools ({mode}):")
    print("  " + ", ".join(sorted(tools_by_name)))

    await scout(tools_by_name)
    await ask_agent(tools)


if __name__ == "__main__":
    asyncio.run(main())
