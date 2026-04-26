---
title: Tessera
---

# Tessera: Tool-Call Gating for LlamaIndex Agents

[Tessera](https://github.com/kenithphilip/Tessera) is an
Apache-2.0 library that gates LlamaIndex agent tool calls when
the active context contains untrusted segments. It composes with
LlamaIndex `AgentRunner` and the new `FunctionCallingAgent` via
a callback handler.

## Why this matters for LlamaIndex users

LlamaIndex agents are commonly fed RAG-retrieved context, which
is exactly the surface most affected by indirect prompt
injection (the canonical OWASP Agentic ASI01 vector). A poisoned
chunk in the index can cause a downstream tool call to be
invoked on the user's behalf without the user noticing.

Tessera's per-segment provenance gives the agent an automatic
floor: any UNTRUSTED segment in the context demotes `min_trust`,
and the next sensitive tool call (e.g. `book_hotel`,
`transfer_funds`) is denied with a structured event the
application can log and surface.

## Install

```bash
pip install llama-index-core tessera-mesh[llamaindex]
```

## Wire the handler

```python
import os, secrets
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.tools import FunctionTool
from tessera.adapters.llamaindex import MeshLlamaIndexHandler
from tessera.policy import Policy
from tessera.signing import HMACSigner

# Tool that the handler will gate.
def book_hotel(city: str, nights: int) -> str:
    return f"Booked {nights} nights in {city}"

# 1. Build a Tessera policy that requires USER trust on book_hotel.
policy = Policy()
policy.require("book_hotel", level="USER")

# 2. Wrap the agent with the Tessera handler.
agent = FunctionCallingAgent.from_tools(
    [FunctionTool.from_defaults(book_hotel)],
    llm=...,  # your LlamaIndex LLM
    callback_manager=MeshLlamaIndexHandler(
        policy=policy,
        signer=HMACSigner(os.environ.get("TESSERA_KEY", secrets.token_bytes(32))),
    ).as_callback_manager(),
)

# 3. Run normally. The handler labels retrieved RAG chunks as
#    UNTRUSTED segments; book_hotel is denied if the active
#    context drew from any of them.
response = agent.chat("Look up reviews for The Plaza and book if rated 4+")
print(response)
```

## What changes for the agent

- **Benign retrievals** behave identically. Tessera's policy
  decision is `Allow` when every context segment carries
  `trust_level >= USER`.
- **Injected RAG chunks** demote `min_trust` to `UNTRUSTED`.
  The next `book_hotel` call is denied; the agent receives a
  structured error and can either retry without the tainted
  context or surface the failure.

## Reference

- Tessera repo: <https://github.com/kenithphilip/Tessera>
- LlamaIndex adapter source:
  [`tessera/adapters/llamaindex.py`](https://github.com/kenithphilip/Tessera/blob/main/src/tessera/adapters/llamaindex.py)
- Threat model:
  <https://github.com/kenithphilip/Tessera/blob/main/SECURITY.md>
- Adapter test:
  [`tests/test_llamaindex_adapter.py`](https://github.com/kenithphilip/Tessera/blob/main/tests/test_llamaindex_adapter.py)
