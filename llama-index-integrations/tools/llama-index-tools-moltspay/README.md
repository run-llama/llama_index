# MoltsPay Tool for LlamaIndex

Pay for AI services using USDC (gasless) via the x402 protocol.

## Installation

```bash
pip install llama-index-tools-moltspay moltspay
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
    service_url="https://juai8.com/zen7",
    service_id="text-to-video",
    prompt="A dragon flying over mountains"
)
```

### discover_services
Discover available services from a provider.

```python
services = tool.discover_services("https://juai8.com/zen7")
```

### get_balance
Check wallet balance.

```python
balance = tool.get_balance()
```

### fund_wallet
Get a link to fund the wallet with USDC.

```python
funding_info = tool.fund_wallet(amount=20.0)
```

### get_limits
Check current spending limits.

```python
limits = tool.get_limits()
```

### set_limits
Update spending limits.

```python
tool.set_limits(max_per_tx=5.0, max_per_day=50.0)
```

## Features

- **Gasless payments**: No ETH needed, pay only in USDC
- **Pay-for-success**: Payment settles only if service delivers
- **Agent-to-agent commerce**: AI agents can purchase from other AI services
- **Multi-chain**: Supports Base, Polygon, Ethereum
- **Spending limits**: Control how much your agent can spend

## Configuration

```python
# Custom wallet path and chain
tools = MoltsPayToolSpec(
    wallet_path="/path/to/wallet.json",
    chain="polygon"  # base, polygon, or ethereum
).to_tool_list()
```

## Links

- [MoltsPay Documentation](https://docs.moltspay.com)
- [x402 Protocol](https://www.x402.org)
- [Available Services](https://moltspay.com/services)
