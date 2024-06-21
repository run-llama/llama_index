# LlamaIndex Agent Integration: LLM Compiler

This Agent integration implements the [LLMCompiler agent paper](https://github.com/SqueezeAILab/LLMCompiler).

A lot of code came from the source repo, we repurposed with LlamaIndex abstractions. All credits
to the original authors for a great work!

A full notebook guide can be found [here](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/agent/llm_compiler.ipynb).

## Usage

First install the package:

```bash
pip install llama-index-agent-llm-compiler
```

```python
# setup pack arguments

from llama_index.core.agent import AgentRunner
from llama_index.agent.llm_compiler.step import LLMCompilerAgentWorker

agent_worker = LLMCompilerAgentWorker.from_tools(
    tools, llm=llm, verbose=True, callback_manager=callback_manager
)
agent = AgentRunner(agent_worker, callback_manager=callback_manager)

# start using the agent
response = agent.chat("What is (121 * 3) + 42?")
```
