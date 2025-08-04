# Wolfram Alpha Tool

This tool connects to a Wolfram alpha account and allows an Agent to perform searches

You will need to provide an API key: https://products.wolframalpha.com/api

## Usage

This tool has more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-wolfram-alpha/examples/wolfram_alpha.ipynb)

Here's an example usage of the WolframAlphaToolSpec.

```python
from llama_index.tools.wolfram_alpha import WolframAlphaToolSpec
from llama_index.agent.core.agent import FunctionAgent
from llama_index.llms.openai import OpenAI


wolfram_spec = WolframAlphaToolSpec(app_id="API-key")

agent = FunctionAgent(
    tools=wolfram_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(await agent.run("how many calories are in 100g of milk chocolate"))
print(await agent.run("what is the mass of the helium in the sun"))
```

`wolfram_alpha_query`: Get the result of a query from Wolfram Alpha

This loader is designed to be used as a way to load data as a Tool in a Agent.
