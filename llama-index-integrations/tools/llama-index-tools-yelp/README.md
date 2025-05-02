# Yelp Tool

This tool connects to Yelp and allows the Agent to search for business and fetch the reviews.

## Usage

This tool has more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-yelp/examples/yelp.ipynb)

Here's an example usage of the YelpToolSpec.

```python
from llama_index.tools.yelp import YelpToolSpec


tool_spec = YelpToolSpec(api_key="your-key", client_id="your-id")

agent = OpenAIAgent.from_tools(zapier_spec.to_tool_list(), verbose=True)

agent.chat("what good restaurants are in toronto")
agent.chat("what are the details of lao lao bar")
```

`business_search`: Use a natural language query to search for businesses
`business_reviews`: Use a business id to fetch reviews

This loader is designed to be used as a way to load data as a Tool in a Agent.
