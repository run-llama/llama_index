"""
Example: Using the Code Execution MCP Server with Anthropic LLM in LlamaIndex.

This example shows how to connect the code-execution MCP server to a
LlamaIndex agent powered by Anthropic's Claude, giving the agent the
ability to execute terminal commands and Python code.

Prerequisites:
    pip install llama-index-tools-mcp llama-index-tools-mcp-code-execution
    pip install llama-index-llms-anthropic
    export ANTHROPIC_API_KEY="your-api-key"

Usage:
    python anthropic_code_agent.py
"""

import asyncio
import os
import sys

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.anthropic import Anthropic
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec


async def main() -> None:
    # 1. Connect to the code-execution MCP server via stdio.
    #    This starts the server as a subprocess.
    mcp_client = BasicMCPClient(
        # Use the installed entry-point
        "code-execution-mcp",
        timeout=30,
    )

    # 2. Convert MCP tools into LlamaIndex FunctionTool objects.
    tool_spec = McpToolSpec(client=mcp_client)
    tools = await tool_spec.to_tool_list_async()

    print(f"Available tools: {[t.metadata.name for t in tools]}")

    # 3. Create an Anthropic-powered agent with the code execution tools.
    llm = Anthropic(model="claude-sonnet-4-20250514", max_tokens=4096)

    agent = FunctionAgent(
        name="CodeExecutionAgent",
        description="An agent that can execute terminal commands and Python code.",
        llm=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful coding assistant. You have access to tools "
            "that let you execute terminal commands and Python code on the "
            "user's machine. Use these tools to help answer questions, run "
            "code, and solve problems. Always show the output to the user."
        ),
    )

    # 4. Run some example tasks.
    print("\n--- Task 1: System info ---")
    response = await agent.run("What operating system are we running on? Use uname -a.")
    print(response)

    print("\n--- Task 2: Python computation ---")
    response = await agent.run(
        "Calculate the first 20 Fibonacci numbers using Python."
    )
    print(response)

    print("\n--- Task 3: Multi-step ---")
    response = await agent.run(
        "Create a Python script that generates 10 random numbers, "
        "saves them to /tmp/random_numbers.txt, then reads the file "
        "back and prints the sum."
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
