# Yelp Tool

This tool connects to Yelp and allows the Agent to search for business and fetch the reviews.

## Usage

This tool has more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-yelp/examples/yelp.ipynb)

Here's an example usage of the YelpToolSpec.

```python
from llama_index.tools.yelp import YelpToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI


tool_spec = YelpToolSpec(api_key="your-key", client_id="your-id")

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(await agent.run("what good restaurants are in toronto"))
print(await agent.run("what are the details of lao lao bar"))
```

`business_search`: Use a natural language query to search for businesses
`business_reviews`: Use a business id to fetch reviews

This loader is designed to be used as a way to load data as a Tool in a Agent.
