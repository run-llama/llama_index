# Zapier Tool

This tool connects to a Zapier account and allows access to the natural language actions API. You can learn more about and enable the NLA API here: https://nla.zapier.com/start/

## Usage

This tool has more extensive example usage documented in a Jupyter notebook [here](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/tools/notebooks/zapier.ipynb)

Here's an example usage of the ZapierToolSpec.

```python
from llama_index.tools.zapier import ZapierToolSpec
from llama_index.agent.openai import OpenAIAgent


zapier_spec = ZapierToolSpec(api_key="sk-ak-your-key")
## Or
zapier_spec = ZapierToolSpec(api_key="oauth-token")

agent = OpenAIAgent.from_tools(zapier_spec.to_tool_list(), verbose=True)

agent.chat("what actions are available")
agent.chat("Can you find the taco night file in google drive")
```

`list_actions`: Get the actions that you have enabled through zapier
`natural_language_query`: Make a natural language query to zapier

This loader is designed to be used as a way to load data as a Tool in a Agent.
