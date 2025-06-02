# Linkup Research Tool

[Linkup](https://app.linkup.so/) is a robust research API tailored specifically for LLM Agents. It seamlessly integrates with diverse data sources to ensure a superior, relevant research experience.

- you need to obtain an API key on the [Linkup dashboard](https://app.linkup.com/)

### Quick Start:

```bash
pip install llama-index-tools-linkup-research
```

```python
import os
from llama_index.core.agent import FunctionCallingAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.linkup_research.base import LinkupToolSpec


# structured_schema=json.dumps(your schema here) # Only if output type is structured
# Initialisation of the tool
linkup_tool = LinkupToolSpec(
    api_key="your Linkup API Key",
    depth="",  # Choose (standard) for a faster result (deep) for a slower but more complete result.
    output_type="",  # Choose (searchResults) for a list of results relative to your query, (sourcedAnswer) for an answer and a list of sources, or (structured) if you want a specific schema.
    # structured_output_schema=structured_schema # Only if output type is structured
)

# Creation of the agent
agent = FunctionCallingAgent.from_tools(
    linkup_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
)

# Query for the agent
agent.chat("Can you tell me which women were awarded the Physics Nobel Prize")
```

This loader is designed to be used as a way to load data as a Tool in an Agent.
