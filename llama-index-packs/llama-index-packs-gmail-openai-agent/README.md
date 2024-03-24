# Gmail OpenAI Agent Pack

Create an OpenAI agent pre-loaded with a tool to interact with Gmail. The tool used is the [Gmail LlamaHub tool](https://llamahub.ai/l/tools-gmail).

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack GmailOpenAIAgentPack --download-dir ./gmail_pack
```

You can then inspect the files at `./gmail_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a the `./gmail_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
GmailOpenAIAgentPack = download_llama_pack(
    "GmailOpenAIAgentPack", "./gmail_pack"
)

gmail_agent_pack = GmailOpenAIAgentPack()
```

From here, you can use the pack, or inspect and modify the pack in `./gmail_pack`.

The `run()` function is a light wrapper around `agent.chat()`.

```python
response = gmail_agent_pack.run("What is my most recent email?")
```

You can also use modules individually.

```python
# Use the agent
agent = gmail_agent_pack.agent
response = agent.chat("What is my most recent email?")

# Use the tool spec in another agent
from llama_index.core.agent import ReActAgent

tool_spec = gmail_agent_pack.tool_spec
agent = ReActAgent.from_tools(tool_spec.to_tool_lost())
```
