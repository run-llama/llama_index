# LLMCompiler Agent Pack

This LlamaPack implements the [LLMCompiler agent paper](https://github.com/SqueezeAILab/LLMCompiler).

A lot of code came from the source repo, we repurposed with LlamaIndex abstractions. All credits
to the original authors for a great work!

A full notebook guide can be found [here](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/agents/llm_compiler/llm_compiler.ipynb).

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack LLMCompilerAgentPack --download-dir ./llm_compiler_agent_pack
```

You can then inspect the files at `./llm_compiler_agent_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a directory. **NOTE**: You must specify `skip_load=True` - the pack contains multiple files,
which makes it hard to load directly.

We will show you how to import the agent from these files!

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
download_llama_pack("LLMCompilerAgentPack", "./llm_compiler_agent_pack")
```

From here, you can use the pack. You can import the relevant modules from the download folder (in the example below we assume it's a relative import or the directory has been added to your system path).

```python
# setup pack arguments

from llama_index.core.agent import AgentRunner
from llm_compiler_agent_pack.step import LLMCompilerAgentWorker

agent_worker = LLMCompilerAgentWorker.from_tools(
    tools, llm=llm, verbose=True, callback_manager=callback_manager
)
agent = AgentRunner(agent_worker, callback_manager=callback_manager)

# start using the agent
response = agent.chat("What is (121 * 3) + 42?")
```

You can also use/initialize the pack directly.

```python
from llm_compiler_agent_pack.base import LLMCompilerAgentPack

agent_pack = LLMCompilerAgentPack(tools, llm=llm)
```

The `run()` function is a light wrapper around `agent.chat()`.

```python
response = pack.run("Tell me about the population of Boston")
```

You can also directly get modules from the pack.

```python
# use the agent
agent = pack.agent
response = agent.chat("task")
```
