# Zapier Tool

This tool connects to a Zapier account and allows access to the natural language actions API. You can learn more about and enable the NLA API here: https://nla.zapier.com/start/

## Usage

This tool has more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-zapier/examples/zapier.ipynb)

Here's an example usage of the ZapierToolSpec.

```python
from llama_index.tools.zapier import ZapierToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI


zapier_spec = ZapierToolSpec(api_key="sk-ak-your-key")
## Or
zapier_spec = ZapierToolSpec(api_key="oauth-token")

agent = FunctionAgent(
    tools=zapier_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(await agent.run("what actions are available"))
print(await agent.run("Can you find the taco night file in google drive"))
```

`list_actions`: Get the actions that you have enabled through zapier
`natural_language_query`: Make a natural language query to zapier

This loader is designed to be used as a way to load data as a Tool in a Agent.
