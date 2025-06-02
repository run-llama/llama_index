# Python File Tool

This tool loads a python file and extracts function names, arguments and descriptions automatically. This tool is particular useful for automatically creating custom Tool Specs when you already have well documented python functions.

## Usage

This tool has more extensive example usage documented in a Jupyter notebook [here](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/tools/notebooks/create_a_tool.ipynb)

Here's an example usage of the PythonFileToolSpec.

```python
from llama_index.tools.python_file import PythonFileToolSpec
from llama_index.agent.openai import OpenAIAgent

pyfile = PythonFileToolSpec("./numpy_linalg.py")

agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

agent.chat(
    """Load the eig, transpose and solve functions from the python file,
and then write a function definition using only builtin python types (List, float, Tuple)
with a short 5-10 line doc string tool prompts for the functions that only has a small description and arguments
"""
)
```

`function_definitions`: Get all of the function definitions from the Python file
`get_function`: Get a specific function definition from the Python file
`get_functions`: Get a list of functions from the python file

This loader is designed to be used as a way to load data as a Tool in a Agent.
