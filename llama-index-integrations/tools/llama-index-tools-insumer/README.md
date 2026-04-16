# LlamaIndex Tools Integration: InsumerAPI

Wallet auth and condition-based access for LlamaIndex agents. Across 33 chains — read → evaluate → sign, returning an ECDSA-signed boolean your agent can verify offline against the public JWKS. Boolean, not balance: the API never exposes wallet holdings, only a signed yes-or-no against the conditions you configure.

Part of [InsumerAPI](https://insumermodel.com/developers/). No secrets. No identity-first. No static credentials.

## Installation

```bash
pip install llama-index-tools-insumer
```

## Quickstart

Get a free API key at [insumermodel.com/developers/](https://insumermodel.com/developers/) (no credit card required for the free tier):

```bash
curl -X POST https://api.insumermodel.com/v1/keys/create \
    -H "Content-Type: application/json" \
    -d '{"email": "you@example.com", "appName": "my-agent", "tier": "free"}'
```

Then use the tool spec in any LlamaIndex agent:

```python
from llama_index.tools.insumer import InsumerToolSpec
from llama_index.agent.openai import OpenAIAgent

insumer = InsumerToolSpec(api_key="insr_live_...")

agent = OpenAIAgent.from_tools(
    insumer.to_tool_list(),
    verbose=True,
)

agent.chat(
    "Does wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 "
    "hold at least 100 USDC on Base?"
)
```

## The four tools

### `attest_wallet`

Run wallet attestation against 1–10 conditions. Returns an ECDSA-signed verdict per condition plus a condition hash for tamper detection.

Supported condition types:

- `token_balance` — ERC-20/SPL/XRPL trust line / native BTC ≥ threshold
- `nft_ownership` — ERC-721/ERC-1155/XRPL NFToken holding
- `eas_attestation` — EAS schema check (pass a `template` like `coinbase_verified_account` or a raw `schemaId`)
- `farcaster_id` — Farcaster ID registered on Optimism

```python
insumer.attest_wallet(
    wallet="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
    conditions=[
        {
            "type": "token_balance",
            "contractAddress": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "chainId": 8453,
            "threshold": 100,
            "decimals": 6,
            "label": "USDC on Base >= 100",
        },
    ],
)
```

Response shape:

```python
{
    "ok": True,
    "data": {
        "attestation": {
            "id": "ATST-...",
            "pass": True,
            "results": [...],      # per-condition booleans + conditionHash
            "passCount": 1,
            "failCount": 0,
            "attestedAt": "2026-04-16T...",
            "expiresAt": "2026-04-16T...",  # +30 min
        },
        "sig": "...",              # ECDSA P-256 signature, base64
        "kid": "insumer-attest-v1",
    },
    "meta": {"creditsRemaining": ..., "creditsCharged": 1, ...},
}
```

Costs 1 credit per call (2 with `proof="merkle"` for EIP-1186 storage proofs).

### `get_trust_profile`

Multi-dimensional wallet trust profile — stablecoins, governance, NFTs, staking (plus Solana, XRPL, Bitcoin when those wallet addresses are supplied). Returns a signed summary showing which dimensions have activity, without exposing raw balances.

```python
insumer.get_trust_profile(
    wallet="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
    solana_wallet="...",       # optional
    xrpl_wallet="r...",        # optional
    bitcoin_wallet="bc1q...",  # optional
)
```

Costs 3 credits per call (6 with `proof="merkle"`).

### `list_compliance_templates`

Discover pre-configured EAS compliance templates (Coinbase Verified Account, Coinbase Verified Country, Coinbase One, Gitcoin Passport, etc.). No API key required.

```python
templates = insumer.list_compliance_templates()
# Use a template name directly in attest_wallet:
insumer.attest_wallet(
    wallet="0x...",
    conditions=[{"type": "eas_attestation", "template": "coinbase_verified_account"}],
)
```

### `get_jwks`

Fetch the public JWKS used to sign attestation and trust responses. Enables offline verification of any result with a standard JWT/JOSE library. No API key required.

```python
jwks = insumer.get_jwks()
# {
#     "keys": [
#         {"kty": "EC", "crv": "P-256", "x": "...", "y": "...",
#          "use": "sig", "alg": "ES256", "kid": "insumer-attest-v1"}
#     ]
# }
```

## Supported chains

33 total:

- **30 EVM chains**: Ethereum, Base, Arbitrum, Optimism, Polygon, Avalanche, BNB, Unichain, Linea, zkSync, Scroll, Blast, Mantle, Celo, Gnosis, Fantom, Sonic, Cronos, Moonbeam, and more
- **Solana** (mainnet)
- **XRPL** (mainnet) — native XRP plus trust-line tokens
- **Bitcoin** (mainnet) — native BTC only

## Positioning

Wallet auth is the primitive. Condition-based access is the category. Token gating is one use case. The API turns a programmable predicate over on-chain state into a short-lived cryptographic artifact any service can verify.

- **No secrets**: conditions are public, the signature binds the condition hash.
- **No identity-first**: a wallet address and a condition are enough.
- **No static credentials**: every response has an expiry and is re-checkable.

## Learn more

- Docs: [insumermodel.com/developers/](https://insumermodel.com/developers/)
- OpenAPI spec: [insumermodel.com/openapi.yaml](https://insumermodel.com/openapi.yaml)
- Public JWKS: [api.insumermodel.com/.well-known/jwks.json](https://api.insumermodel.com/.well-known/jwks.json)
- Companion packages: `langchain-insumer` (LangChain), `mcp-server-insumer` (Model Context Protocol), `eliza-plugin-insumer` (ElizaOS)

## License

Apache-2.0
