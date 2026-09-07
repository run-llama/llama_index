# llama-index-tools-opticparse

Official [LlamaIndex](https://www.llamaindex.ai/) `BaseToolSpec` integration for **OpticParse Multimodal Vision Web Scraper** and **PhishVision Zero-Day Cybersecurity & Drainer Detection**.

Empower your autonomous LlamaIndex agents, RAG workflows, and retrieval pipelines with stealth, anti-bot resilient web extraction and real-time security scanning.

---

## Installation

```bash
pip install llama-index-tools-opticparse
```

---

## Quickstart

### 1. Zero-CSS Autonomous Web Extraction

Equip your LlamaIndex Agent to scrape live websites without worrying about dynamic JavaScript, Cloudflare challenges, or brittle CSS selectors:

```python
import os
from llama_index.tools.opticparse import OpticParseToolSpec
from llama_index.core.agent import FunctionCallingAgentWorker

# Initialize tool spec
tool_spec = OpticParseToolSpec(
    api_key=os.getenv("OPTICPARSE_API_KEY", "op_live_demo_agent")
)

# Convert to LlamaIndex tool list
tools = tool_spec.to_tool_list()

# Attach tools directly to your LlamaIndex Agent
# agent = FunctionCallingAgentWorker.from_tools(tools, llm=llm).as_agent()
# response = agent.chat("Scrape the latest pricing and specs from https://news.ycombinator.com")
# print(response)
```

### 2. Standalone Tool Execution

You can also run tool operations directly as standard Python methods:

```python
from llama_index.tools.opticparse import OpticParseToolSpec

tool_spec = OpticParseToolSpec()

# Extract structured content
docs = tool_spec.extract(
    url="https://example.com",
    query="Extract product details and main headers"
)
print("Extracted content:", docs[0].text)

# Inspect unknown URLs for zero-day phishing or wallet drainers
threat_docs = tool_spec.inspect_threat(
    url="https://suspicious-crypto-portal.xyz"
)
print("Threat analysis:", threat_docs[0].text)
```

---

## Tool Functions Exposed

| Function | Description | Return Type |
| :--- | :--- | :--- |
| `extract(url, query)` | Visual web scraper with anti-bot evasion & structured parsing. | `List[Document]` |
| `inspect_threat(url)` | Real-time zero-day cybersecurity & Web3 wallet-drainer scanner. | `List[Document]` |

---

## Supported Ecosystems
- **LlamaIndex Agents**: `FunctionCallingAgent`, `OpenAIAgent`, `ReActAgent`
- **LangChain**: Use `langchain-opticparse`
- **Model Context Protocol (MCP)**: Native integration on Smithery & Glama
- **Coinbase AgentKit**: On-chain autonomous agent action provider

---

## License
MIT License.
