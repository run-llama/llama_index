# LlamaIndex Agent Integration: Lats

This integration implements the Language Agent Tree search method introduced
in the paper entitled "Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models" by Zhou et al. 2023.

Source: https://arxiv.org/pdf/2310.04406.pdf

## Usage

LATs is implemented as a `BaseAgentWorker` and as such is used with an `AgentRunner`.

```python
from llama_index.agent.lats import LATSAgentWorker
from llama_index.core.agent import AgentRunner

tools = ...
llm = ...
agent_worker = LATSAgentWorker(tools=tools, llm=llm)
agent = AgentRunner(agent_worker)

agent.chat(...)
```
