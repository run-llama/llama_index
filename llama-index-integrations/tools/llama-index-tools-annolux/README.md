# LlamaIndex Annolux Tool Integration

Official [LlamaIndex](https://llamaindex.ai) tool spec integration for **[Annolux](https://annolux.com)** — The curated English & Chinese web search API and MCP for AI Agents with explicit snapshot timestamps.

[![PyPI version](https://img.shields.io/pypi/v/llama-index-tools-annolux.svg?style=flat-square)](https://pypi.org/project/llama-index-tools-annolux/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

---

## ⚡ Installation

```bash
pip install llama-index-tools-annolux
```

---

## 🔑 Configuration

Get your free API key (1,000 requests/month) at [annolux.com](https://annolux.com) and set it in your environment:

```bash
export ANNOLUX_API_KEY="ann_live_your_key_here"
```

---

## 🚀 Quickstart

### 1. Standalone Tool Spec
```python
from llama_index.tools.annolux import AnnoluxToolSpec

tool_spec = AnnoluxToolSpec()
docs = tool_spec.annolux_search(query="Rust async runtime tokio best practices", limit=5)

for doc in docs:
    print(f"[{doc.extra_info['fetched_at']}] {doc.extra_info['title']}")
    print(doc.text[:200])
    print("-" * 40)
```

### 2. Using with LlamaIndex Agents
```python
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.llms.openai import OpenAI
from llama_index.tools.annolux import AnnoluxToolSpec

tool_spec = AnnoluxToolSpec()
tools = tool_spec.to_tool_list()

llm = OpenAI(model="gpt-4o")
agent = AgentRunner(FunctionCallingAgentWorker.from_tools(tools, llm=llm, verbose=True))

response = agent.chat("What are the latest updates to React Server Components in 2026?")
print(response)
```

---

## 📄 License

MIT © [Annolux](https://annolux.com)
