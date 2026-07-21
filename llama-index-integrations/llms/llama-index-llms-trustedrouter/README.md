# LlamaIndex Llms Integration: TrustedRouter

[TrustedRouter](https://trustedrouter.com) is an OpenAI-compatible LLM router
that serves many models behind one endpoint through an open-source, verifiable
attested gateway. It does not log prompts or outputs by default.

## Installation

```bash
pip install llama-index-llms-trustedrouter
```

## Usage

Set your API key:

```bash
export TRUSTEDROUTER_API_KEY=***
```

```python
from llama_index.llms.trustedrouter import TrustedRouter

llm = TrustedRouter(model="trustedrouter/zdr")

resp = llm.complete("Who is Paul Graham?")
print(resp)
```

Streaming and chat work like any other OpenAI-like LLM:

```python
from llama_index.core.llms import ChatMessage

resp = llm.chat([ChatMessage(role="user", content="Tell me a joke")])
print(resp)

for chunk in llm.stream_complete("Count to five"):
    print(chunk.delta, end="")
```

Model routes include `trustedrouter/auto`, `trustedrouter/zdr` (zero data
retention), and `trustedrouter/confidential`, plus individually addressable
models — see the live catalog at `https://trustedrouter.com/v1/models`.
