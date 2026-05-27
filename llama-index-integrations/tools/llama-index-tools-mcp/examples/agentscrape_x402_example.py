"""AgentScrape x402 MCP example for llama-index-tools-mcp.

Connect a LlamaIndex agent to AgentScrape, a live pay-per-call web-scraping
MCP server that uses the x402 payment protocol on Base USDC. Agents pay
autonomously per call — no signup, no API keys.

AgentScrape exposes six tools as a remote MCP server via Streamable HTTP
(with SSE fallback for legacy clients):
    - scrape_webpage          ($0.003) - markdown/html/text/json scrape
    - extract_structured_data ($0.005) - AI extraction via Groq + Llama 4 Scout
    - screenshot_webpage      ($0.003) - PNG screenshot with viewport control
    - extract_metadata        ($0.002) - title, OG, Twitter, JSON-LD
    - create_browser_session  ($0.001) - stateful browser session
    - run_workflow            ($0.008) - multi-step atomic workflow up to 20 steps

Free tier: 10 calls per wallet in the first 30 days. The example below
hits the free tier so it works zero-config. For paid usage, supply the
X-PAYMENT-RESPONSE header carrying an x402 payment receipt.

Service URLs:
    - MCP:       https://agent-scrape.healingsunhaven.workers.dev/mcp
    - x402:      https://agent-scrape.healingsunhaven.workers.dev/.well-known/x402.json
    - A2A card:  https://agent-scrape.healingsunhaven.workers.dev/.well-known/agent.json
    - GitHub:    https://github.com/hshintelligence/agent-scrape

Run:
    pip install llama-index-tools-mcp llama-index-llms-openai
    export OPENAI_API_KEY=sk-...
    python agentscrape_x402_example.py
"""

import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec


async def main() -> None:
    # Connect to AgentScrape's remote MCP endpoint.
    # The client auto-negotiates Streamable HTTP and falls back to SSE.
    mcp_client = BasicMCPClient(
        "https://agent-scrape.healingsunhaven.workers.dev/mcp",
        # Uncomment to supply an x402 payment receipt for paid calls:
        # headers={
        #     "X-PAYMENT-RESPONSE": "<base64-encoded x402 payment receipt>",
        # },
    )

    mcp_tool_spec = McpToolSpec(client=mcp_client)
    tools = await mcp_tool_spec.to_tool_list_async()

    print(f"Loaded {len(tools)} tools from AgentScrape:")
    for t in tools:
        print(f"  - {t.metadata.name}")

    agent = FunctionAgent(
        name="AgentScrapeDemo",
        description="An agent that uses AgentScrape to read the web.",
        llm=OpenAI(model="gpt-4o-mini"),
        tools=tools,
        system_prompt=(
            "You are a research assistant with access to AgentScrape's "
            "web-scraping tools. Use them to answer questions about web "
            "content concisely."
        ),
    )

    response = await agent.run(
        "Scrape https://www.x402.org and give me the headline plus a "
        "two-sentence summary."
    )
    print("\nAgent response:\n", response)


if __name__ == "__main__":
    asyncio.run(main())
