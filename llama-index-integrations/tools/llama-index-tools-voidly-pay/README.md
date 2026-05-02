# Voidly Pay Tool

[Voidly Pay](https://voidly.ai/pay) is an x402 payment rail for AI agents — let your LlamaIndex
agent autonomously pay for HTTP 402 endpoints, no API keys, no Stripe customer object. Identity
is an Ed25519 keypair on disk.

Settlement happens off-chain in Voidly Pay credits (Stage 1) or on-chain USDC on Base mainnet
(Stage 2). The vault is Sourcify-verified at
[`0xb592...1c12`](https://basescan.org/address/0xb592512932a7b354969bb48039c2dc7ad6ad1c12) with
public reserves at [voidly.ai/pay/proof](https://voidly.ai/pay/proof).

## Installation

```bash
pip install llama-index-tools-voidly-pay
```

## Provision a wallet (10 free credits, no install)

Visit [voidly.ai/pay/claim](https://voidly.ai/pay/claim) — generates an Ed25519 keypair in your
browser and grants 10 credits via the public faucet. Save the keypair to
`~/.voidly-pay-keypair.json` and you're ready to call paid endpoints.

## Usage

```python
from llama_index.tools.voidly_pay import VoidlyPayToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

voidly_tool = VoidlyPayToolSpec()  # uses ~/.voidly-pay-keypair.json by default

agent = FunctionAgent(
    tools=voidly_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
)

await agent.run(
    "Browse the Voidly Pay marketplace, find a Wikipedia summary endpoint, "
    "and pay 1¢ to fetch the summary for 'Alan Turing'."
)
```

## Available Functions

- `pay_for_url(url, method, params, body)` — call any URL; auto-pays a `402 Payment Required`
  response by signing a transfer envelope, retries with `X-Payment`, and returns the body plus
  the signed receipt.
- `discover_paid_endpoints()` — list all paid endpoints currently advertised on the facilitator.
- `marketplace_browse(category)` — browse the open Voidly Pay marketplace
  ([api.voidly.ai/v1/pay/marketplace](https://api.voidly.ai/v1/pay/marketplace)).
- `health_check()` — run the 6-check trust report (facilitator reachability, vault verification,
  wallet balance, keypair validity, settlement health). Useful before a long-running paid task.
- `list_listing(name, endpoint_url, price_usdc, description, category)` — list your own paid
  endpoint on the marketplace.

## How x402 works

1. Agent calls a tool that hits a `402 Payment Required` endpoint
2. Endpoint returns a signed quote (price, recipient, expiry)
3. SDK signs a transfer envelope with the agent's Ed25519 keypair
4. Endpoint verifies and returns the actual response

One round-trip after the first 402. Sub-200ms typical settlement on the live facilitator.

## List your own paid endpoint

Visit [voidly.ai/pay/list-your-service](https://voidly.ai/pay/list-your-service) — browser-only
Ed25519 keypair, no install. Once listed, every Voidly-aware agent (LlamaIndex, LangChain,
Vercel AI, MCP) can find and call your URL. Voidly takes zero platform cut on Stage 1.

## Resources

- [Voidly Pay docs](https://voidly.ai/pay)
- [Live marketplace JSON](https://api.voidly.ai/v1/pay/marketplace)
- [Sourcify-verified vault](https://repo.sourcify.dev/contracts/full_match/8453/0xb592512932A7B354969BB48039C2dC7Ad6AD1c12/)
- [Public reserves dashboard](https://voidly.ai/pay/proof)
- [voidly-pay PyPI SDK](https://pypi.org/project/voidly-pay/) (this tool wraps it)

This loader is designed to be used as a way to load Voidly Pay primitives as Tools in an Agent.
