# Wolfram Alpha Tool

This tool connects to a Wolfram alpha account and allows an Agent to perform searches

You will need to provide an API key: https://products.wolframalpha.com/api

## Usage

This tool has more extensive example usage documented in a Jupyter notebook [here](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/tools/notebooks/wolfram_alpha.ipynb)

Here's an example usage of the WolframAlphaToolSpec.

```python
from llama_index.tools.wolfram_alpha import WolframAlphaToolSpec
from llama_index.agent.openai import OpenAIAgent


wolfram_spec = WolframAlphaToolSpec(app_id="API-key")

agent = OpenAIAgent.from_tools(wolfram_spec.to_tool_list(), verbose=True)

agent.chat("how many calories are in 100g of milk chocolate")
agent.chat("what is the mass of the helium in the sun")
```

`wolfram_alpha_query`: Get the result of a query from Wolfram Alpha

This loader is designed to be used as a way to load data as a Tool in a Agent.
