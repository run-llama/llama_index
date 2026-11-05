# ProveCart ToolSpec

```python
from llama_index.tools.provecart import ProveCartToolSpec

spec = ProveCartToolSpec(api_key="...")   # integrator key
tools = spec.to_tool_list()
```

Tools:

- `verify_cart` — check a cart an agent built against the
  customer's signed mandate before payment capture. Verdicts:
  MATCH (capture may proceed), BLOCK (the response names exactly
  what to fix), REVIEW (ask the buyer). ProveCart never holds or
  moves money.
- `certification_info` — read the public Verified Agent record:
  currently certified agents, scores, permanent result links. No
  key needed.

Docs: https://provecart.com/portal/docs ·
OpenAPI: https://api.provecart.com/openapi.json

Published by ProveCart, a verification service for agent commerce
based in the Bay Area. Patent pending.
