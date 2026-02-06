# Nory x402 Payment Tool

This tool enables AI agents to make payments using the x402 HTTP payment protocol via Nory.

## Features

- **Multi-chain support**: Solana and 7 EVM chains (Base, Polygon, Arbitrum, Optimism, Avalanche, Sei, IoTeX)
- **Fast settlement**: Sub-400ms payment settlement
- **Simple integration**: Handle HTTP 402 Payment Required responses automatically

## Installation

```bash
pip install llama-index-tools-nory-x402
```

## Usage

```python
from llama_index.tools.nory_x402 import NoryX402ToolSpec
from llama_index.agent.openai import OpenAIAgent

# Initialize the tool
nory_tool = NoryX402ToolSpec(api_key="your-api-key")  # api_key is optional

# Create an agent with the tool
agent = OpenAIAgent.from_tools(
    nory_tool.to_tool_list(),
    verbose=True,
)

# Use the agent to handle payments
response = agent.chat(
    "Check the payment requirements for /api/premium/data with amount 0.10 USDC on Solana"
)
```

## Available Functions

### get_payment_requirements
Get x402 payment requirements for accessing a paid resource.

```python
result = nory_tool.get_payment_requirements(
    resource="/api/premium/data",
    amount="0.10",
    network="solana-mainnet"  # optional
)
```

### verify_payment
Verify a signed payment transaction before settlement.

```python
result = nory_tool.verify_payment(payload="base64-encoded-payload")
```

### settle_payment
Submit a verified payment to the blockchain (~400ms settlement).

```python
result = nory_tool.settle_payment(payload="base64-encoded-payload")
```

### lookup_transaction
Check the status of a previously submitted payment.

```python
result = nory_tool.lookup_transaction(
    transaction_id="tx-signature",
    network="solana-mainnet"
)
```

### health_check
Check Nory service health and supported networks.

```python
result = nory_tool.health_check()
```

## Supported Networks

| Network | ID |
|---------|-----|
| Solana Mainnet | `solana-mainnet` |
| Solana Devnet | `solana-devnet` |
| Base | `base-mainnet` |
| Polygon | `polygon-mainnet` |
| Arbitrum | `arbitrum-mainnet` |
| Optimism | `optimism-mainnet` |
| Avalanche | `avalanche-mainnet` |
| Sei | `sei-mainnet` |
| IoTeX | `iotex-mainnet` |

## Links

- [x402 Protocol Specification](https://x402.org)
- [Nory Documentation](https://noryx402.com/docs)
