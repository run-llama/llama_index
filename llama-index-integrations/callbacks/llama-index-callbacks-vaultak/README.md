# llama-index-callbacks-vaultak

[![PyPI version](https://badge.fury.io/py/llama-index-callbacks-vaultak.svg)](https://pypi.org/project/llama-index-callbacks-vaultak)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Runtime security for LlamaIndex agents, powered by Vaultak.**

Intercept every agent action, tool call, and LLM query in real time — score risk, enforce policies, mask PII, and automatically block dangerous behavior before it reaches your production systems.

---

## Install

```bash
pip install llama-index-callbacks-vaultak
```

---

## Quick Start

```python
from llama_index.core.callbacks import CallbackManager
from llama_index.callbacks.vaultak import VaultakCallbackHandler

# Initialize the handler
handler = VaultakCallbackHandler(api_key="vtk_...")
callback_manager = CallbackManager([handler])

# Use with a query engine
query_engine = index.as_query_engine(
    callback_manager=callback_manager
)

# Query — every action is now monitored and secured
response = query_engine.query("Summarize our Q3 revenue data")
```

---

## Set Globally

```python
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.callbacks.vaultak import VaultakCallbackHandler

Settings.callback_manager = CallbackManager([
    VaultakCallbackHandler(api_key="vtk_...")
])
```

---

## Use with Agents

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.callbacks import CallbackManager
from llama_index.callbacks.vaultak import VaultakCallbackHandler

handler = VaultakCallbackHandler(
    api_key="vtk_...",
    agent_name="my-production-agent",
    risk_threshold=6.0,
)

agent = ReActAgent.from_tools(
    tools,
    callback_manager=CallbackManager([handler]),
)
```

---

## What Gets Monitored

| LlamaIndex Event | Vaultak Action |
|---|---|
| `FUNCTION_CALL` start | Risk-scores the action, blocks if above threshold |
| `FUNCTION_CALL` start | Checks tool call against policy rules |
| `FUNCTION_CALL` end | Scans output for PII and masks it |
| `LLM` start | Checks LLM inputs against policy |
| `EXCEPTION` | Sends alert + triggers rollback |
| `QUERY` end | Scans response for PII |

---

## Configuration

```python
handler = VaultakCallbackHandler(
    api_key="vtk_...",           # Required
    agent_name="my-agent",       # Label in the Vaultak dashboard
    block_on_high_risk=True,     # Block actions above threshold
    risk_threshold=7.0,          # 0-10 scale
    verbose=True,                # Log all scored actions
)
```

---

## Links

- [Vaultak docs](https://docs.vaultak.com)
- [LlamaIndex docs](https://docs.llamaindex.ai)
- [PyPI](https://pypi.org/project/llama-index-callbacks-vaultak)
- [GitHub](https://github.com/samueloladji-beep/llama-index-vaultak)
