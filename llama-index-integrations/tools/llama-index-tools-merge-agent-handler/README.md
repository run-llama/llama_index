# LlamaIndex Merge Agent Handler ToolSpec

LlamaIndex tool integration for connecting AI agents to [Merge Agent Handler](https://merge.dev/agent-handler) Tool Packs over MCP.

This package provides four agent-callable methods:

- `list_tool_packs`
- `list_registered_users`
- `list_tools`
- `call_tool`

## Prerequisites

- Python 3.9+
- Merge Agent Handler API key
- At least one Merge Tool Pack and Registered User

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

## Quick Start

```python
from llama_index.tools.merge_agent_handler import MergeAgentHandlerToolSpec
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

merge_tools = MergeAgentHandlerToolSpec(
    api_key="your-merge-api-key",
    tool_pack_id="your-tool-pack-id",
    registered_user_id="your-registered-user-id",
)

tools = merge_tools.to_tool_list()

llm = OpenAI(model="gpt-4o")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

response = agent.chat(
    "List the available tools in my Merge Tool Pack, then use one to fetch recent Jira tickets."
)
print(response)
```

Example file: [`examples/merge_agent_handler_example.py`](./examples/merge_agent_handler_example.py)

## Constructor Parameters

| Parameter | Type | Description |
|---|---|---|
| `api_key` | `str` | Merge Agent Handler API key |
| `tool_pack_id` | `str \| None` | Default Tool Pack ID fallback for `list_tools` and `call_tool` |
| `registered_user_id` | `str \| None` | Default Registered User ID fallback for `list_tools` and `call_tool` |
| `environment` | `str` | Default environment (`production` or `test`) for `list_registered_users` |

## Tool Reference

| Method | Description |
|---|---|
| `list_tool_packs()` | Lists Merge Tool Packs available to the API key |
| `list_registered_users(environment=None)` | Lists registered users filtered by `production` or `test` |
| `list_tools(tool_pack_id=None, registered_user_id=None)` | Lists MCP tools for a Tool Pack + Registered User |
| `call_tool(tool_name, arguments="{}", tool_pack_id=None, registered_user_id=None)` | Executes an MCP tool and returns text output or an error string |

## Local UI Tester

A Streamlit UI is included for manual testing.

```bash
source .venv/bin/activate
streamlit run ui/merge_agent_handler_ui.py
```

UI capabilities:

- initialize `MergeAgentHandlerToolSpec`
- list tool packs
- list registered users
- list MCP tools
- call MCP tools with raw JSON arguments

## Testing and Linting

```bash
source .venv/bin/activate
pytest tests/test_tools.py
ruff check .
mypy llama_index/tools/merge_agent_handler/base.py tests/test_tools.py examples/merge_agent_handler_example.py ui/merge_agent_handler_ui.py
```

Or using the project make targets:

```bash
make lint
make test
```

## Related Links

- Merge Agent Handler docs: <https://docs.ah.merge.dev/>
- Merge dashboard: <https://ah.merge.dev/>
- n8n implementation: `/Users/pritak/Documents/n8n-nodes-merge`
- Langflow implementation: `/Users/pritak/Documents/langflow-merge-agent-handler`
- Agno implementation: `/Users/pritak/Documents/agno-merge-agent-handler`
- Letta implementation: `/Users/pritak/Documents/letta-merge-agent-handler`

## License

[MIT](./LICENSE)
