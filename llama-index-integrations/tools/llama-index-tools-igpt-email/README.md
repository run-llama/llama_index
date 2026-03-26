# LlamaIndex Tools Integration: iGPT Email Intelligence

This tool connects to [iGPT](https://igpt.ai/) to give your agent
structured, reasoning-ready context from connected email threads.

iGPT handles thread reconstruction, participant role detection, temporal
reasoning, and intent extraction before returning results — so agents
receive clean, structured JSON instead of raw message data.

To begin, you need to obtain an API key at [docs.igpt.ai](https://docs.igpt.ai).

## Usage

```python
# %pip install llama-index llama-index-core llama-index-tools-igpt-email

from llama_index.tools.igpt_email import IGPTEmailToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = IGPTEmailToolSpec(api_key="your-key", user="user-id")

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(await agent.run("What tasks were assigned to me this week?"))
```

`ask`: Ask a question about email context using iGPT's reasoning engine. Returns structured context including tasks, decisions, owners, sentiment, deadlines, and citations.

`search`: Search email context for relevant messages and threads. Returns matching email context as Documents with thread metadata (subject, participants, date, thread ID).

This tool is designed to be used as a way to load data as a Tool in an Agent.
