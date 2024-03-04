# Passio Nutrition AI Tool

This tool connects to a Passio Nutrition AI account and allows an Agent to perform searches for 4000 different unique foods, 1.5M packaged food, images and videos.

You will need to set up a search key using Azure,learn more here: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview

## Usage

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/tools/notebooks/bing_search.ipynb)

Here's an example usage of the NutritionAIToolSpec.

```python
from llama_index.tools.passio_nutrition_ai import NutritionAIToolSpec
from llama_index.agent import OpenAIAgent

tool_spec = NutritionAIToolSpec(api_key="your-key")

agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

agent.chat("What is the nutritional value of an apple?")
agent.chat("I had a salad for lunch, how many calories did I eat?")
```

`passio_nutrition_ai`: Search for foods and their micro nutrition results related to a query

This loader is designed to be used as a way to load data as a Tool in a Agent. See [here](https://github.com/emptycrown/llama-hub/tree/main) for examples.