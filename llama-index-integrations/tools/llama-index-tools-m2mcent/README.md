# LlamaIndex M2MCent ToolSpec

This tool spec enables LlamaIndex agents to query and execute **1,005 autonomous MCP micro-services** via the **M2MCent Gateway** (`https://api.m2mcent.com`) monetized via **x402 V2 on Base Mainnet**.

## Usage

```python
from llama_index.tools.m2mcent import M2MCentToolSpec
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

# 1. Initialize M2MCent ToolSpec
m2mcent_spec = M2MCentToolSpec(base_url="https://api.m2mcent.com")
tools = m2mcent_spec.to_tool_list()

# 2. Attach tools to your autonomous agent
agent = ReActAgent.from_tools(tools, llm=OpenAI(model="gpt-4o"), verbose=True)

# 3. Query or execute tasks
response = agent.chat("Search for a smart contract audit tool in M2MCent and analyze this Solidity snippet.")
print(response)
```

## Features
- **1,005 Specialized Agent Tools:** Finance, security, bioinformatics, 3D, logistics, smart contracts.
- **x402 V2 Micropayment Protocol:** Trustless USDC settlement on Base Mainnet (`0xDb48F51A2de8F4a80CD1d0BAdcd18E847734A74a`).
- **Semantic Discovery:** Real-time keyword & domain search across the entire 1,005 micro-service catalog.
