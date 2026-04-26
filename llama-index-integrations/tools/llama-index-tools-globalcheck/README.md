# GlobalCheck Tool

This tool allows LlamaIndex agents to deterministically verify international trade compliance before executing purchases or shipments.

## Usage

This tool requires an API Key for GlobalCheck (available on RapidAPI).

```python
from llama_index.tools.globalcheck import GlobalCheckToolSpec
from llama_index.agent.openai import OpenAIAgent

tool_spec = GlobalCheckToolSpec(api_key="your-rapidapi-key")

agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

agent.chat("Can I ship 50 lithium batteries from CN to the US?")
```
