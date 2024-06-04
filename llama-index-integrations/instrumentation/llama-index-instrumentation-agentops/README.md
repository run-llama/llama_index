# LlamaIndex Instrumentation Integration: Agentops

To integrate AgentOps into your agent workflow,
simply import `AgentOpsHandler` and add a handler,
as demonstrated below. Note that all keyword arguments
anticipated by AgentOps' `AOClient` can be provided
to this client using the same keyword arguments in
`add_handler()`.

```
from llama_index.instrumentation.agentops import AgentOpsHandler

AgentOpsHandler.add_handler()
```
