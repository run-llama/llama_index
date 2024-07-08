# LlamaIndex Instrumentation Integration: Agentops

To integrate AgentOps into your agent workflow,
simply import and initialize an `AgentOpsHandler`,
as demonstrated below. Note that all keyword arguments
anticipated by AgentOps' `AOClient` can be provided
to this client using the same keyword arguments in
`init()`.

```
from llama_index.callbacks.agentops import AgentOpsHandler

AgentOpsHandler.init()
```
