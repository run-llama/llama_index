"""Minimal runnable example for the Ejentum Reasoning Harness tool spec.

Requires:
    pip install llama-index-tools-ejentum llama-index-llms-openai
    export EJENTUM_API_KEY=...
    export OPENAI_API_KEY=...
"""

import asyncio

from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.ejentum import EjentumToolSpec


async def main() -> None:
    # Reads EJENTUM_API_KEY from the environment. Pass api_key=... to override.
    spec = EjentumToolSpec()
    tools = spec.to_tool_list()
    print(f"Loaded tools: {[t.metadata.name for t in tools]}")

    agent = ReActAgent.from_tools(
        tools,
        llm=OpenAI(model="gpt-4o-mini"),
        verbose=True,
    )

    response = await agent.achat(
        "Why might our microservice return 503s only under specific load patterns?"
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
