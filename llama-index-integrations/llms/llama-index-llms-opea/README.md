# LlamaIndex Llms Integration: OPEA LLM

OPEA (Open Platform for Enterprise AI) is a platform for building, deploying, and scaling AI applications. As part of this platform, many core gen-ai components are available for deployment as microservices, including LLMs.

Visit [https://opea.dev](https://opea.dev) for more information, and their [GitHub](https://github.com/opea-project/GenAIComps) for the source code of the OPEA components.

## Installation

1. Install the required Python packages:

```bash
%pip install llama-index-llms-opea
```

## Usage

```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.opea import OPEA

llm = OPEA(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    api_base="http://localhost:8080/v1",
    temperature=0.7,
    max_tokens=256,
    additional_kwargs={"top_p": 0.95},
)

# Complete a prompt
response = llm.complete("What is the capital of France?")
print(response)

# Stream a chat response
response = llm.stream_chat(
    [ChatMessage(role="user", content="What is the capital of France?")]
)
for chunk in response:
    print(chunk.delta, end="", flush=True)
```

All available methods include:

- `complete()`
- `stream_complete()`
- `chat()`
- `stream_chat()`

as well as async versions of the methods with the `a` prefix.
