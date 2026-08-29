# LlamaIndex Callback Integration: Agentops

```shell
pip install llama-index-callbacks-agentops
```

To integrate AgentOps into your agent workflow,
simply import and initialize an `AgentOpsHandler`,
as demonstrated below. Note that all keyword arguments
anticipated by AgentOps' `AOClient` can be provided
to this client using the same keyword arguments in
`init()`.

You can initialize globally using

```python
from llama_index.core import set_global_handler

set_global_handler("agentops", api_key="...")
```

or:

```python
from llama_index.callbacks.agentops import AgentOpsHandler

AgentOpsHandler.init(api_key="...")
```
