# LlamaIndex Tools Integration: WarpPay402

This integration provides LlamaIndex `FunctionTool` wrappers for **WarpPay402** x402 micropayment APIs and composite autonomous agents across Base Mainnet, Solana, Arbitrum One, Robinhood Chain, and Arc Mainnet.

## Installation

```bash
pip install llama-index-tools-warppay402
```

## Usage

```python
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.warppay402 import WarpPayLlamaIndexTools

# Retrieve all 31 WarpPay402 tools & autonomous composite agents
tools = WarpPayLlamaIndexTools.get_tools()

agent = ReActAgent.from_tools(tools, llm=OpenAIhmodel="gpt-4o-mini"), verbose=True)

# Single-tool oracle query ($0.001 USDC)
print(agent.chat("What is the current ETH/USDC spot price on Arc Mainnet?"))

# Composite multi-step DeFi due-diligence & quote agent ($0.15 USDC)
print(agent.chat("Audit Base USDC and simulate a 10 USDC swap to AERO"))

# Composite parallel web, commerce deals & FX research agent ($0.05 USDC)
print(agent.chat("Find coffee deals under $25 and check the current USD to EUR rate"))
```
