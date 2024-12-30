# Code Interpreter Tool

This tool can be used to run python scripts and capture the results of stdout and stderr

WARNING: This tool provides the Agent access to the `subprocess.run` command.
Arbitrary code execution is possible on the machine running this tool.
This tool is not recommended to be used in a production setting, and would require heavy sandboxing or virtual machines

## Usage

Here's an example usage of the CodeInterpreterToolSpec.

```python
from llama_index.tools.code_interpreter import CodeInterpreterToolSpec
from llama_index.agent.openai import OpenAIAgent

code_spec = CodeInterpreterToolSpec()

agent = OpenAIAgent.from_tools(code_spec.to_tool_list())

# Prime the agent to use the tool
agent.chat(
    "Can you help me write some python code to pass to the code_interpreter tool"
)
agent.chat(
    "write a python function to calculate volume of a sphere with radius 4.3cm"
)
```

The tools available are:

`code_interpreter`: A tool to evaluate a python script

This loader is designed to be used as a way to load data as a Tool in a Agent.
