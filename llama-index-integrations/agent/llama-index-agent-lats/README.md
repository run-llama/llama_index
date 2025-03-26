# Language Agent Tree Search (LATS) Agent

This agent integration implements the Language Agent Tree Search method introduced
in the paper titled "Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models" by Zhou et al. 2023.

Check out the source paper: https://arxiv.org/pdf/2310.04406.pdf

## Usage

LATS is implemented as a `BaseAgentWorker` and as such is used with an `AgentRunner`.

```python
from llama_index.agent.lats import LATSAgentWorker
from llama_index.core.agent import AgentRunner

tools = ...
llm = ...
agent_worker = LATSAgentWorker(tools=tools, llm=llm)
agent = AgentRunner(agent_worker)

agent.chat(...)
```
