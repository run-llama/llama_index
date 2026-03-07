---
title: llamaindex-soul
---

# llamaindex-soul: Persistent Memory with RAG+RLM Retrieval

[llamaindex-soul](https://github.com/menonpg/llamaindex-soul) provides markdown-native chat storage for LlamaIndex with hybrid RAG+RLM retrieval.

## Features

- **Persistent Memory**: Chat history stored in human-readable markdown files
- **RAG + RLM Hybrid Retrieval**: Auto-routes queries to semantic search (RAG) or exhaustive reasoning (RLM)
- **Database Schema Intelligence**: Auto-document your database via LLM (soul-schema)
- **Managed Cloud Option**: SoulMate API for zero-infrastructure production deployments
- **Git-Versionable**: Memory files you can read, edit, and version control

## Installation

```bash
pip install llamaindex-soul
```

This installs:
- `soul-agent` — Core RAG+RLM memory library
- `soul-schema` — Database semantic layer generator
- `httpx` — For SoulMate API client

## Quick Start

### Basic Usage

```python
from llamaindex_soul import SoulChatStore
from llama_index.core.memory import ChatMemoryBuffer

# Create markdown-based chat storage
chat_store = SoulChatStore()
memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user1",
)

# Use with any LlamaIndex agent
from llama_index.core.agent import FunctionAgent
from llama_index.llms.openai import OpenAI

# Define your tools and LLM
tools = []  # Add your Tool objects here
llm = OpenAI(model="gpt-4")

agent = FunctionAgent(tools=tools, llm=llm)
await agent.run("Hello!", memory=memory)
```

### Semantic Search

```python
# Search past conversations
results = chat_store.recall("user1", "What did we discuss about databases?")
for result in results:
    print(f"[{result['score']:.2f}] {result['content']}")
```

### Database Schema Intelligence

```python
from llamaindex_soul import SchemaMemory

# Connect to any SQLAlchemy-compatible database
schema = SchemaMemory("postgresql://user:pass@host/db")
schema.generate()  # Auto-documents tables via LLM

# Get context for Text-to-SQL
context = schema.context_for("Show me revenue by region")
```

### Managed Cloud (SoulMate)

```python
from llamaindex_soul import SoulMateChatStore

# Zero-infrastructure option
chat_store = SoulMateChatStore(api_key="your-key")
memory = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store,
    chat_store_key="user1",
)
```

### Factory Function

```python
from llamaindex_soul import create_chat_store

# Choose your backend:
store = create_chat_store("local")      # File-based

# OR for managed cloud:
# store = create_chat_store("soulmate", api_key="your-key")
```

## Links

- **PyPI**: [llamaindex-soul](https://pypi.org/project/llamaindex-soul/)
- **GitHub**: [menonpg/llamaindex-soul](https://github.com/menonpg/llamaindex-soul)
- **Documentation**: [Blog Post](https://menonlab-blog-production.up.railway.app/blog/langchain-llamaindex-soul-integrations)

## Part of the Soul Ecosystem

- [soul-agent](https://github.com/menonpg/soul.py) — Core RAG+RLM library
- [soul-schema](https://github.com/menonpg/soul-schema) — Database semantic layer
- [crewai-soul](https://github.com/menonpg/crewai-soul) — CrewAI integration
- [langchain-soul](https://github.com/menonpg/langchain-soul) — LangChain integration
- [SoulMate](https://menonpg.github.io/soulmate) — Managed cloud service
