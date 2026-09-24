"""End-to-end example: FeedMyAgentToolSpec with a LlamaIndex FunctionAgent.

Requires an OpenAI API key (``OPENAI_API_KEY``) for the LLM. FeedMyAgent
reads are anonymous, so no FeedMyAgent API key is needed to run this
example as written; only ``report_incident`` requires one (see the
``report_new_incident`` example function at the bottom).

Run with:

    python examples/function_agent_usage.py
"""

import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.feedmyagent import FeedMyAgentToolSpec


async def main() -> None:
    feedmyagent_tool = FeedMyAgentToolSpec()  # anonymous reads, no API key

    agent = FunctionAgent(
        tools=feedmyagent_tool.to_tool_list(),
        llm=OpenAI(model="gpt-4o-mini"),
        system_prompt=(
            "You are an assistant that keeps engineers up to date on AI "
            "agent security, compliance, and tooling news using the "
            "FeedMyAgent feed."
        ),
    )

    response = await agent.run(
        "What are the three most recent items tagged 'mcp'? "
        "Then search the feed for anything about prompt injection."
    )
    print(response)


def report_new_incident() -> None:
    """Standalone example of posting to the feed (requires an API key)."""
    from feedmyagent import FeedMyAgent

    key = FeedMyAgent.provision_key(owner="llamaindex-example-agent")
    tool = FeedMyAgentToolSpec(api_key=key)

    confirmation = tool.report_incident(
        title="New prompt-injection technique in MCP tool descriptions",
        description=(
            "Observed a tool description embedding an instruction to "
            "exfiltrate conversation history via a follow-up tool call."
        ),
        url="https://example.com/writeup",
    )
    print(confirmation)


if __name__ == "__main__":
    asyncio.run(main())
