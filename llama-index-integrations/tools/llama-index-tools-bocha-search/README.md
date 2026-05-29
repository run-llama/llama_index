# LlamaIndex Tools Integration: Bocha AI Web Search

This tool integrates the Bocha AI Web Search API with LlamaIndex. It allows agentic workflows to search the web for real-time information, page summaries, and images.

## Prerequisite

You will need a Bocha AI API key. Register and obtain a key from the [Bocha AI Open Platform](https://open.bocha.cn).

## Usage

```python
from llama_index.tools.bocha_search import BochaSearchToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

# Initialize the tool specification with your API key
tool_spec = BochaSearchToolSpec(api_key="your-bocha-api-key")

# Create an agent equipped with Bocha search tools
agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4"),
)

# Run agent queries
response = await agent.run("What are the latest developments in AI search?")
print(response)
```
