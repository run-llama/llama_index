# Adding other tools

Now that you've built a capable agent, we hope you're excited about all it can do. The core of expanding agent capabilities is the tools available, and we have good news: [LlamaHub](https://llamahub.ai) from LlamaIndex has hundreds of integrations, including [dozens of existing agent tools](https://llamahub.ai/?tab=tools) that you can use right away. We'll show you how to use one of the existing tools, and also how to build and contribute your own.

## Using an existing tool from LlamaHub

For our example, we're going to use the [Yahoo Finance tool](https://llamahub.ai/l/tools/llama-index-tools-yahoo-finance?from=tools) from LlamaHub. It provides a set of six agent tools that look up a variety of information about stock ticker symbols.

First we need to install the tool:

```bash
pip install llama-index-tools-yahoo-finance
```

Then we can set up our dependencies. This is exactly the same as our previous examples, except for the final import:

```python
from dotenv import load_dotenv

load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
```

To show how custom tools and LlamaHub tools can work together, we'll include the code from our previous examples the defines a "multiple" tool. We'll also take this opportunity to set up the LLM:

```python
# settings
Settings.llm = OpenAI(model="gpt-4o", temperature=0)


# function tools
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)
```

Now we'll do the new step, which is to fetch the array of tools:

```python
finance_tools = YahooFinanceToolSpec().to_tool_list()
```

This is just a regular array, so we can use Python's `extend` method to add our own tools to the mix:

```python
finance_tools.extend([multiply_tool, add_tool])
```

Then we set up the agent as usual, and ask a question:

```python
agent = ReActAgent.from_tools(finance_tools, verbose=True)

response = agent.chat("What is the current price of NVDA?")

print(response)
```

The response is very wordy, so we've truncated it:

```
Thought: The current language of the user is English. I need to use a tool to help me answer the question.
Action: stock_basic_info
Action Input: {'ticker': 'NVDA'}
Observation: Info:
{'address1': '2788 San Tomas Expressway'
...
'currentPrice': 135.58
...}
Thought: I have obtained the current price of NVDA from the stock basic info.
Answer: The current price of NVDA (NVIDIA Corporation) is $135.58.
The current price of NVDA (NVIDIA Corporation) is $135.58.
```

Perfect! As you can see, using existing tools is a snap.

As always, you can check [the repo](https://github.com/run-llama/python-agents-tutorial/blob/main/6_tools.py) to see this code all in one place.

## Building and contributing your own tools

We love open source contributions of new tools! You can see an example of [what the code of the Yahoo finance tool looks like](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-yahoo-finance/llama_index/tools/yahoo_finance/base.py):
* A class that extends `BaseToolSpec`
* A set of arbitrary Python functions
* A `spec_functions` list that maps the functions to the tool's API

Once you've got a tool working, follow our [contributing guide](https://github.com/run-llama/llama_index/blob/main/CONTRIBUTING.md#2--contribute-a-pack-reader-tool-or-dataset-formerly-from-llama-hub) for instructions on correctly setting metadata and submitting a pull request.

Congratulations! You've completed our guide to building agents with LlamaIndex. We can't wait to see what use-cases you build!
