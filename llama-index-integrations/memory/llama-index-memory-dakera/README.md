# LlamaIndex Memory Integration: Dakera

[Dakera](https://dakera.ai) provides persistent, decay-weighted cross-session memory for AI agents. Memories are stored as dense vectors and retrieved via semantic similarity search, with automatic importance decay so stale memories fade naturally while relevant context surfaces reliably.

## Installation

```bash
pip install llama-index-memory-dakera
```

## Setup

You need a running Dakera server (self-hosted or cloud) and an API key.

```python
from llama_index.memory.dakera import DakeraMemory

memory = DakeraMemory(
    base_url="https://api.dakera.ai",  # your Dakera server URL
    api_key="dak-...",                 # your API key
    session_id="user-42",             # unique ID per user/conversation
    top_k=10,                         # memories to retrieve per query
)
```

## Usage with SimpleChatEngine

```python
from llama_index.core import SimpleChatEngine
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o-mini")

engine = SimpleChatEngine.from_defaults(llm=llm, memory=memory)

response = await engine.achat("Hi, I prefer answers in French.")
response = await engine.achat("What language do I prefer?")
# → Dakera surfaces the stored preference: "User prefers answers in French."
```

## Usage with FunctionAgent

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool


def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


agent = FunctionAgent(
    tools=[FunctionTool.from_defaults(fn=search_web)],
    llm=llm,
)

response = await agent.run("Remember that my timezone is UTC+2.", memory=memory)
response = await agent.run("Schedule a meeting for 9am my time.", memory=memory)
# → Agent recalls timezone from memory when scheduling
```

## How It Works

- **`put(message)`** — stores a chat message in Dakera via `POST /v1/memories`
- **`get(input)`** — retrieves the top-k semantically relevant memories for the input via `POST /v1/memories/search`
- **`get_all()`** — fetches all memories for the session
- **`reset()`** — deletes all session memories via `DELETE /v1/memories`

Memories are namespaced by `session_id`, so different users or agent runs stay fully isolated.

## Self-Hosting

Dakera ships as a single Docker image:

```bash
docker run -p 8000:8000 -e DAKERA_API_KEY=my-key ghcr.io/dakera-ai/dakera:latest
```

See [dakera.ai](https://dakera.ai) for full deployment docs.

## References

- [Dakera website](https://dakera.ai)
- [Dakera API docs](https://dakera.ai/docs)
- [PyPI: llamaindex-dakera](https://pypi.org/project/llamaindex-dakera/)
