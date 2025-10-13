import asyncio
import os
from llama_index.tools.signnow import SignNowMCPToolSpec
from llama_index.core.agent.workflow import FunctionAgent


async def main():
    # Pass SignNow credentials directly via env_overrides (no .env required)
    spec = SignNowMCPToolSpec.from_env(
        env_overrides={
            # Option 1: token-based auth
            # "SIGNNOW_TOKEN": "your_signnow_token_here",

            # Option 2: credential-based auth
            "SIGNNOW_USER_EMAIL": "login",
            "SIGNNOW_PASSWORD": "password",
            "SIGNNOW_API_BASIC_TOKEN": "basic_token",
        }
    )
    # Fetch tools from MCP server
    tools = await spec.to_tool_list_async()
    print({"count": len(tools), "names": [t.metadata.name for t in tools]})

    # Create an agent and ask for templates list
    agent = FunctionAgent(
        name="SignNow Agent",
        description="Query SignNow via MCP tools",
        tools=tools,
        system_prompt="Be helpful.",
    )

    resp = await agent.run("Show me list of templates and their names")
    print(resp)


if __name__ == "__main__":
    asyncio.run(main())
