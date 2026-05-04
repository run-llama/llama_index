# Synap

[Synap](https://maximem.ai) is a managed memory layer for AI agents and assistants.
It automatically extracts and structures knowledge from conversations — facts,
preferences, episodes, emotions, and temporal events — and retrieves only what is
relevant to the current query, so agents remember users across sessions without
any manual bookkeeping.

## Installation

```bash
pip install maximem-synap-llamaindex
```

Get an API key at [synap.maximem.ai](https://synap.maximem.ai).

## Retriever

`SynapRetriever` is a standard LlamaIndex `BaseRetriever`. It queries Synap for
memories relevant to the current query and returns them as `NodeWithScore` objects
with metadata including memory type and confidence score — ready to use in any
LlamaIndex pipeline or query engine.

```python
import os

from maximem_synap import MaximemSynapSDK
from synap_llamaindex import SynapRetriever

sdk = MaximemSynapSDK(api_key=os.environ["SYNAP_API_KEY"])

retriever = SynapRetriever(
    sdk=sdk,
    user_id="user_123",
    customer_id="acme_corp",
    mode="accurate",  # or "fast" (~50 ms) for real-time conversations
    max_results=20,
)

nodes = retriever.retrieve("What are the user's dietary restrictions?")
```

## Chat Memory

`SynapChatMemory` implements LlamaIndex's `BaseMemory`. Attach it to any LlamaIndex
chat engine to persist conversation context across sessions automatically.

```python
import os

from maximem_synap import MaximemSynapSDK
from synap_llamaindex import SynapChatMemory

sdk = MaximemSynapSDK(api_key=os.environ["SYNAP_API_KEY"])

memory = SynapChatMemory(
    sdk=sdk,
    conversation_id="conv_abc",
    user_id="user_123",
    customer_id="acme_corp",
)
```

## More Resources

- [Synap Documentation](https://docs.maximem.ai)
- [LlamaIndex Integration Guide](https://docs.maximem.ai/integrations/llamaindex)
- [Dashboard](https://synap.maximem.ai)
- [PyPI: maximem-synap-llamaindex](https://pypi.org/project/maximem-synap-llamaindex/)
- [Open source integration package](https://github.com/maximem-ai/maximem_synap_sdk/tree/main/packages/integrations)
