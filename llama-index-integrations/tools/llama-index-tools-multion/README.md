# MultiOn Tool

```bash
pip install llama-index-tools-multion
```

This tool connects to [MultiOn](https://www.multion.ai/) to enable your agent to easily
connect to the internet through your Chrome Web browser and act on your behalf

You will need to have the MultiOn chrome extension installed and a MultiOn account
to use this integration

## Usage

This tool has more a extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-multion/examples/multion.ipynb).

Here's an example usage of the MultionToolSpec.

```python
from llama_index.tools.multion import MultionToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

multion_tool = MultionToolSpec(api_key="your-multion-key")

agent = FunctionAgent(
    tools=multion_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(await agent.run("Can you read the latest tweets from my followers"))
print(await agent.run("What's the next thing on my google calendar?"))
```

`browse`: The core function that takes natural language instructions to pass to the web browser to execute

This loader is designed to be used as a way to load data as a Tool in a Agent.
