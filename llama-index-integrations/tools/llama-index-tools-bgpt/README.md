# BGPT Search Tool

This tool connects to [BGPT](https://bgpt.pro/mcp/) and lets an Agent search scientific papers with structured full-text experimental evidence: methods, sample sizes, limitations, conflicts of interest, falsifiability prompts, and more.

## REST API

This integration uses the plain HTTP REST endpoint (not MCP transport):

- Search: `POST https://bgpt.pro/api/mcp-search`
- DOI lookup: `POST https://bgpt.pro/api/mcp-doi-lookup`

The free tier works without an API key (50 results per network).

## Usage

```python
from llama_index.tools.bgpt import BGPTToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = BGPTToolSpec()

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

await agent.run(
    "Search for papers on semaglutide cardiovascular outcomes and summarize limitations"
)
await agent.run("Look up DOI 10.1038/s41586-024-07386-0")
```

`search_papers`: Keyword search returning structured evidence fields per paper.

`lookup_paper`: Fetch a single paper by DOI.

This loader is designed to be used as a Tool in an Agent.
