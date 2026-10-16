# LlamaIndex Tools Integration for WarpPay402

This package provides a LlamaIndex tool integration for WarpPay402 x402 monetized tools and micro-oracles across Base, Solana, Arbitrum, and Arc Mainnet.

## Installation

```bash
pip install llama-index-tools-warppay402 llama-index-llms-openai
```

## Usage

```python
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.warppay402 import WarpPayLlamaIndexTools

tools = WarpPayLlamaIndexTools.get_tools()
agent = ReActAgent.from_tools(tools, llm=OpenAI(model="gpt-4o-mini"), verbose=True)

response = agent.chat("What is the current ETH/USDC spot price on Arc Mainnet?")
print(response)
```