# Code Interpreter Tool

This tool can be used to run python scripts and capture the results of stdout and stderr

WARNING: By default, this tool provides the Agent access to the `subprocess.run` command.
Arbitrary code execution is possible on the machine running this tool without sandboxing.

### Secure Sandboxed Execution (Vetto)

To run code in production securely without containers or root daemons, enable [Vetto](https://github.com/shleder/vetto) kernel sandboxing (<4ms cold start, Linux Landlock LSM ABI 1-6, macOS Seatbelt, Windows LPAC):

```python
# Option 1: Pass sandbox="vetto" to CodeInterpreterToolSpec
from llama_index.tools.code_interpreter import CodeInterpreterToolSpec

code_spec = CodeInterpreterToolSpec(sandbox="vetto", timeout=30)

# Option 2: Use VettoCodeInterpreterToolSpec directly
from llama_index.tools.code_interpreter import VettoCodeInterpreterToolSpec

code_spec = VettoCodeInterpreterToolSpec(timeout=30)
```

Sandboxed execution isolates the workspace directory, blocks network access by default, and masks sensitive host environment variables (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AWS_*`).

## Usage

Here's an example usage of the CodeInterpreterToolSpec.

```python
from llama_index.tools.code_interpreter import CodeInterpreterToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

code_spec = CodeInterpreterToolSpec()

agent = FunctionAgent(
    tools=code_spec.to_tool_list(), llm=OpenAI(model="gpt-4.1")
)

# Prime the agent to use the tool
resp = await agent.run(
    "Can you help me write some python code to pass to the code_interpreter tool"
)
resp = await agent.run(
    "write a python function to calculate volume of a sphere with radius 4.3cm"
)
```

The tools available are:

`code_interpreter`: A tool to evaluate a python script

This loader is designed to be used as a way to load data as a Tool in a Agent.
