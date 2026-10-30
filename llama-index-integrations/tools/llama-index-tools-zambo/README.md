# llama-index-tools-zambo

**Zambo: [zambo.dev](https://zambo.dev) | Add Zambo to your AI: [zambo.dev/add](https://zambo.dev/add) | AER-1: [zambo.dev/aer1](https://zambo.dev/aer1/)**

Zambo is the verifiable work layer for AI. This package is the LlamaHub native tool integration for LlamaIndex: it wraps the Zambo MCP endpoint and offline AER-1 receipt verification in the standard `BaseToolSpec` idiom, so any LlamaIndex agent can call Zambo tools and check verifiable receipts without leaving the framework.

## What Zambo gives a LlamaIndex agent

- 153 tools behind one MCP endpoint, https://zambo.dev/mcp, with 36 tools on the default surface.
- HANDOVER, the flagship: wrap up a session in one AI and pick it up in another, with the work, memory, and context carried across.
- A verifiable receipt for every job. A receipt proves the saved result was not changed after the fact. It does not prove the answer was right.
- Offline receipt verification against AER-1, the IETF draft at draft-zambo-aer1-26. The conformance corpus lives at https://zambo.dev/aer1/corpus.json (126 vectors, 129 checks).

Free tier: 20 calls + 1 wrap-up free every day; picking up in another AI is always free. Day Pass: $0.99 for unlimited use for 24 hours. No account and no API key are needed for the free tier.

For one-line AER-1 receipts collected from a running LlamaIndex agent, see the companion package [aer1-llamaindex](https://pypi.org/project/aer1-llamaindex/).

## Install

```bash
pip install llama-index-tools-zambo
```

## Use it

```python
from llama_index.tools.zambo import ZamboToolSpec

spec = ZamboToolSpec()
tools = spec.to_tool_list()

for tool in tools:
    print(tool.metadata.name)
```

The tool spec exposes three functions:

- `list_tools`: list the tools the Zambo MCP endpoint offers.
- `call_tool(tool_name, arguments_json)`: call any Zambo tool by name through the MCP endpoint. Every call returns a verifiable receipt for the job.
- `verify_receipt(receipt_json)`: verify an AER-1 workflow receipt offline. The Merkle root is recomputed from the receipt's own steps and the required fields are checked locally, with no network call and no trust in any server.

## Example: verify a receipt offline

```python
import json
from llama_index.tools.zambo import ZamboToolSpec, merkle_root

steps = [
    {"seq": 1, "receipt_id": "11111111-1111-4111-8111-111111111111", "tool": "search"},
    {"seq": 2, "receipt_id": "22222222-2222-4222-8222-222222222222", "tool": "summarize"},
]
receipt = {
    "type": "verifiable-workflow-receipt",
    "workflow_id": "33333333-3333-4333-8333-333333333333",
    "receipt_id": "44444444-4444-4444-8444-444444444444",
    "merkle_root": merkle_root([s["receipt_id"] for s in steps]),
    "steps": steps,
}

spec = ZamboToolSpec()
print(spec.verify_receipt(json.dumps(receipt)))
# verified: AER-1 workflow receipt 44444444-4444-4444-8444-444444444444 ...
```

## Links

- Zambo: https://zambo.dev
- Add Zambo to your AI: https://zambo.dev/add
- AER-1 standard: https://zambo.dev/aer1/
- IETF draft: https://datatracker.ietf.org/doc/draft-zambo-aer1/26/
- Verify a receipt: https://zambo.dev/verify/

License: Apache-2.0
