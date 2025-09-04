# Get OS for environment variables
import os

# Set up OpenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.valyu import ValyuToolSpec


async def main():
    valyu_tool = ValyuToolSpec(
        api_key=os.environ["VALYU_API_KEY"],
        max_price=100,  # default is 100
    )

    agent = FunctionAgent(
        tools=valyu_tool.to_tool_list(),
        llm=OpenAI(model="gpt-4.1"),
    )

    print(
        await agent.run(
            "What are the key considerations and empirical evidence for implementing statistical arbitrage strategies using cointegrated pairs trading, specifically focusing on the optimal lookback period for calculating correlation coefficients and the impact of transaction costs on strategy profitability in high-frequency trading environments?"
        )
    )

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
