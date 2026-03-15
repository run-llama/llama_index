# MoltsPay Tool for LlamaIndex

Pay for AI services using USDC (gasless) via the x402 protocol.

## Installation

```bash
pip install llama-index-tools-moltspay
```

## Setup

1. Initialize a MoltsPay wallet:
```bash
npx moltspay init --chain base
```

2. Fund your wallet with USDC:
```bash
npx moltspay fund
```

## Usage

```python
from llama_index.tools.moltspay import MoltsPayToolSpec
from llama_index.core.agent import FunctionAgent
from llama_index.llms.openai import OpenAI

# Create tools
tools = MoltsPayToolSpec().to_tool_list()

# Create agent with payment capability
agent = FunctionAgent(
    tools=tools,
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can pay for AI services."
)

# Run agent
response = await agent.run(
    user_msg="Generate a video of a cat dancing using the Zen7 service at https://juai8.com/zen7"
)
print(response)
```

## Available Functions

### pay_service
Pay for and execute an AI service.

```python
result = tool.pay_service(
    provider_url="https://juai8.com/zen7",
    service_id="text-to-video",
    prompt="A dragon flying over mountains"
)
```

### get_services
List available services from a provider.

```python
services = tool.get_services("https://juai8.com/zen7")
```

### get_balance
Check wallet balance.

```python
balance = tool.get_balance()
```

### fund_wallet
Get instructions to fund the wallet with USDC.

```python
funding_info = tool.fund_wallet()
```

## Features

- **Gasless payments**: No ETH needed, pay only in USDC
- **Pay-for-success**: Payment settles only if service delivers
- **Agent-to-agent commerce**: AI agents can purchase from other AI services
- **Multi-chain**: Supports Base, Polygon, Ethereum

## Links

- [MoltsPay Documentation](https://docs.moltspay.com)
- [x402 Protocol](https://www.x402.org)
- [Available Services](https://moltspay.com/services)
