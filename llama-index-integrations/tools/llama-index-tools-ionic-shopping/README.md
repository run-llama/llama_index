# LlamaIndex Tools Integration: Ionic Shopping

```bash
pip install llama-index-tools-ionic-shopping
```

[Ionic](https://ioniccommerce.com) is a plug and play ecommerce marketplace for AI Assistants.
By including the Ionic Tool in your agent, you are effortlessly providing your users with the ability
to shop and transact directly within your agent, and you’ll get a cut of the transaction.

## Attribution

Llearn more about how [Ionic attributes sales](https://docs.ioniccommerce.com/guides/attribution)
to your agent. Provide your Ionic API Key when instantiating the tool:

```python
from llama_index.tools.ionic_shopping import IonicShoppingToolSpec

ionic_tool = IonicShoppingToolSpec(api_key="<my Ionic API Key>").to_tool_list()
```

## Usage

Try it out using the [Jupyter notebook](https://github.com/run-llama/llama-hub/blob/main/llama_hub/tools/notebooks/ionic_shopping.ipynb).

```python
import openai
from llama_index.core.agent import (
    OpenAIAgent,
)  # requires llama-index-agent-openai
from llama_index.tools.ionic_shopping import IonicShoppingToolSpec

openai.api_key = "sk-api-key"

ionic_tool = IonicShoppingToolSpec(api_key="<my Ionic API Key>").to_tool_list()

agent = OpenAIAgent.from_tools(ionic_tool)
print(
    agent.chat(
        "I'm looking for a 5k monitor can you find me 3 options between $600 and $1000"
    )
)
```

`query`: used to search for products and to get product recommendations

Your users can use natural language to specify how many results they would like to see
and what their budget is.

For more information on setting up your Agent with Ionic, see the [Ionic documentation](https://docs.ioniccommerce.com).
