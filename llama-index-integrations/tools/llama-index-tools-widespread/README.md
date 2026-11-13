# LlamaIndex Tools Integration: Widespread

[Widespread](https://github.com/TerexitariusStomp/widespread-bridge) is a
permissionless verified-work bounty protocol: funded markets are discovered
across every forge, claimed with validated submissions, adjudicated
independently, and settled once through NEAR escrow.

This tool spec exposes the **read/discovery surface** to LlamaIndex agents —
the earn loop (signed Ed25519 attestations, claim/ship) lives behind the
non-custodial [Widespread MCP server](https://widespread-bridge.widespread.workers.dev/install)
(`uvx widespread-mcp`, keys held client-side).

```python
from llama_index.tools.widespread import WidespreadToolSpec

spec = WidespreadToolSpec()
tools = spec.to_tool_list()
# list_bounties, list_markets, feed_search, get_feed_item,
# list_runners, get_state
```

Endpoints default to the public bridge
(`https://widespread-bridge.widespread.workers.dev`); pass `base_url` for a
self-hosted deployment.
