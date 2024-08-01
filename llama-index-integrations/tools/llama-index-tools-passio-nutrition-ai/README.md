# Passio Nutrition AI Tool

This tool connects to a Passio Nutrition AI account and allows an Agent to perform searches against a database of over 2.2M foods.

You will need to set up a search key using Passio Nutrition API,learn more here: https://www.passio.ai/nutrition-ai#nutrition-api-pricing

## Usage

Here's an example usage of the NutritionAIToolSpec.

```python
from llama_index.tools.passio_nutrition_ai import NutritionAIToolSpec
from llama_index.agent import OpenAIAgent

tool_spec = NutritionAIToolSpec(api_key="your-key")

agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

agent.chat("What is the nutritional value of an apple?")
agent.chat("I had a cobb salad for lunch, how many calories did I eat?")
```

`passio_nutrition_ai`: Search for foods and their micro nutrition results related to a query

This loader is designed to be used as a way to load data as a Tool in a Agent.
