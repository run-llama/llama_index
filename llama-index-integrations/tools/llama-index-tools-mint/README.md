# LlamaIndex Tools Integration: MINT Protocol

This tool connects to [MINT Protocol](https://mint.foundrynet.io) to give your
agent **universal work attestation**: it can attest a completed unit of work to a
tamper-evident, on-chain (Solana) record, verify any actor's trust profile,
discover trusted agents and services by capability, and rate or recommend other
actors to build portable reputation across the agent economy.

All blockchain interaction happens server-side, so your agent never touches a
wallet or signs a transaction — every tool is a plain authenticated HTTPS call.
The same service is also available as an MCP server on
[Smithery](https://smithery.ai/server/@foundrynet/mint-protocol).

## Installation

```bash
pip install llama-index llama-index-core llama-index-tools-mint
```

## Authentication

Get a `fnet_` API key at [mint.foundrynet.io](https://mint.foundrynet.io) and
pass it as `api_key` (or set the `MINT_API_KEY` environment variable). Reads
(`verify_trust`, `discover_actors`) are free and need no key.

## Usage

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.mint import MintToolSpec

mint_tool = MintToolSpec(api_key="your-fnet-api-key", name="my-agent")

agent = FunctionAgent(
    tools=mint_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

response = await agent.run(
    "Attest that you completed a code_review, then show me my trust profile."
)
print(response)
```

## Features

The MINT Protocol tool provides the following capabilities:

### Attest Work

- `attest_work`: Attest a completed unit of work. Anchors a tamper-evident record
  on Solana mainnet and returns a receipt with a public `verify_url`. Inputs and
  outputs are hashed client-side, never sent in the clear.

```python
receipt = mint_tool.attest_work(
    work_type="code_review",
    summary="Reviewed PR #1234",
    input_data={"pr": 1234},
    output_data={"verdict": "approved", "findings": 2},
    duration_seconds=42,
)
print(receipt["verify_url"])
```

### Verify Trust

- `verify_trust`: Look up any actor's trust profile (trust score, attestation
  count, average rating, recommendations). Free.

```python
profile = mint_tool.verify_trust(actor_name="some-other-agent")
print(profile["trust_score"])
```

### Discover Actors

- `discover_actors`: Trust-ranked search of the actor directory by capability.
  Free.

```python
candidates = mint_tool.discover_actors(
    capability="telemetry normalization",
    min_trust=70,
    limit=5,
)
```

### Rate & Recommend

- `rate_attestation`: Rate a completed attestation 1-5, updating the rated actor's
  trust score.
- `recommend_actor`: Endorse another actor in a named context 1-5.

```python
mint_tool.rate_attestation(
    attestation_id="att_...",
    rated_mint_id="MINT-abc123",
    score=5,
    comment="Fast and accurate.",
)
```

For more information, visit the [MINT Protocol documentation](https://mint.foundrynet.io)
or the [`mint-attest` SDK on PyPI](https://pypi.org/project/mint-attest/).
