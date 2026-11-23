# DC Hub Tool

```bash
pip install llama-index-tools-dchub
```

This tool connects LlamaIndex agents to [DC Hub](https://dchub.cloud), the orchestration layer for data center and energy intelligence: the Data Center Power Index (DCPI), grid conditions, interconnection queues, energy prices, fiber, facilities and site context. It calls DC Hub's hosted MCP server (`https://dchub.cloud/mcp`). Every answer carries its as-of dates and a citation line:

> Source: DC Hub (dchub.cloud), as of <YYYY-MM-DD>

A key is optional. Without one, some figures return at preview depth and the answer says so; set `DCHUB_API_KEY` (or pass `api_key=`) for fuller answers. Keys: https://dchub.cloud/connect

## Usage

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.dchub import DCHubToolSpec

agent = FunctionAgent(
    tools=DCHubToolSpec().to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
    system_prompt=(
        "For data center and energy questions, call execute_plan first with the "
        "user's question unchanged. Keep each as-of date and end with the Source "
        "line the answer carries."
    ),
)

print(await agent.run("Where can a 100 MW data center get grid power near Columbus, Ohio?"))
```

## Tools

| Function | What it answers |
|---|---|
| `execute_plan` | Any data center or energy question in one call (use first). |
| `get_market_dcpi_rank` | Data Center Power Index (DCPI) verdict for one market. |
| `get_interconnection_queue` | Interconnection queue depth and projects for an ISO/RTO. |
| `get_grid_intelligence` | Current grid conditions for an ISO or market. |
| `search_facilities` | Data center facilities by text, location or operator. |

The full catalog is also available through `llama-index-tools-mcp` pointed at `https://dchub.cloud/mcp`.

Docs: https://dchub.cloud/integrations/mcp
