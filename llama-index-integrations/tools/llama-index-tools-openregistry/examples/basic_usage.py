"""Walk a UK company's officers + persons with significant control via the
OpenRegistry MCP server.

Run with:

    pip install llama-index-tools-openregistry llama-index-llms-openai
    export OPENAI_API_KEY=sk-...
    python examples/basic_usage.py

The example uses the free anonymous tier — no OpenRegistry signup needed.
"""

import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

from llama_index.tools.openregistry import OpenRegistryToolSpec


async def main() -> None:
    tool_spec = OpenRegistryToolSpec(
        # For lower latency, allowlist just the tools the agent needs:
        allowed_tools=[
            "search_companies",
            "get_company_profile",
            "get_officers",
            "get_persons_with_significant_control",
        ],
    )

    agent = FunctionAgent(
        tools=tool_spec.to_tool_list(),
        llm=OpenAI(model="gpt-4.1"),
    )

    response = await agent.run(
        "Find Tesco PLC on Companies House (jurisdiction gb). Pull its current"
        " directors and persons with significant control, and quote the upstream"
        " `nature_of_control` strings exactly as the registry returns them."
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
