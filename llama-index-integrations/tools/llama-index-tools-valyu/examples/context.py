# Get OS for environment variables
import os

# Set up OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.tools.valyu import ValyuToolSpec


valyu_tool = ValyuToolSpec(
    api_key=os.environ["VALYU_API_KEY"],
    max_price=100,  # default is 100
)

agent = OpenAIAgent.from_tools(
    valyu_tool.to_tool_list(),
    verbose=True,
)

response = agent.chat(
    "What are the key considerations and empirical evidence for implementing statistical arbitrage strategies using cointegrated pairs trading, specifically focusing on the optimal lookback period for calculating correlation coefficients and the impact of transaction costs on strategy profitability in high-frequency trading environments?"
)
print(response)
