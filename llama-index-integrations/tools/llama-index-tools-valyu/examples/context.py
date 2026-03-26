# Get OS for environment variables
import os

# Set up OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.tools.valyu import ValyuToolSpec


valyu_tool = ValyuToolSpec(
    api_key=os.environ["VALYU_API_KEY"],
    max_price=100,  # default is 100
    fast_mode=True,  # Enable fast mode for faster but shorter results
    # Contents API configuration
    contents_summary=True,  # Enable AI summarization for content extraction
    contents_extract_effort="normal",  # Extraction thoroughness
    contents_response_length="medium",  # Content length per URL
)

agent = OpenAIAgent.from_tools(
    valyu_tool.to_tool_list(),
    verbose=True,
)

# Example 1: Search query
print("=== Search Example ===")
search_response = agent.chat(
    "What are the key considerations and empirical evidence for implementing statistical arbitrage strategies using cointegrated pairs trading, specifically focusing on the optimal lookback period for calculating correlation coefficients and the impact of transaction costs on strategy profitability in high-frequency trading environments?"
)
print(search_response)

# Example 2: URL content extraction
print("\n=== URL Content Extraction Example ===")
content_response = agent.chat(
    "Please extract and summarize the content from these URLs: https://arxiv.org/abs/1706.03762 and https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)"
)
print(content_response)
