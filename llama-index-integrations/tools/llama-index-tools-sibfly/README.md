# LlamaIndex Tools: SibFly

[SibFly](https://sibfly.com) tool spec for LlamaIndex — **measured ground motion for any US address**.

Give it a US address (or lat/lon) and it tells your agent how fast the ground is
**sinking or rising, in mm/year**, measured from NASA OPERA Sentinel-1 satellite
radar (InSAR) — measured, not modeled. Negative mm/year = sinking.

Agent-friendly pricing: **$0.40 per covered report, and misses are free**. A free
`check_coverage` preflight lets you size a query before spending.

## Install

```bash
pip install llama-index-tools-sibfly
export SIBFLY_API_KEY="sf_live_..."
```

Get a key with free starter credits at [sibfly.com](https://sibfly.com). Agents can
self-register: `POST https://sibfly.com/api/v1/autonomous/register`.

## Usage

```python
from llama_index.tools.sibfly import SibflyToolSpec

sibfly = SibflyToolSpec()  # reads SIBFLY_API_KEY

# Free preflight
print(sibfly.check_coverage(address="1100 Congress Ave, Austin, TX"))

# Paid report ($0.40 if covered)
doc = sibfly.check_ground_motion(address="1100 Congress Ave, Austin, TX")
print(doc.text)       # short verdict
print(doc.metadata)   # structured fields

# Free dry-run preview
sibfly.check_ground_motion(lat=30.3244, lon=-97.8102, dry_run=True)
```

Route logic on `metadata["assessment_code"]` — one of `rapid_subsidence`,
`notable_subsidence`, `stable`, `mild_uplift`, `strong_uplift`.

## Use it in an agent

```python
from llama_index.tools.sibfly import SibflyToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.anthropic import Anthropic

agent = FunctionAgent(
    tools=SibflyToolSpec().to_tool_list(),
    llm=Anthropic(model="claude-sonnet-5"),
)
```

## Links

- API: <https://sibfly.com> · Docs: <https://sibfly.com/docs> · Agent docs: <https://sibfly.com/llms.txt>
- OpenAPI: <https://sibfly.com/openapi.json> · MCP: `https://sibfly.com/mcp` (registry: `com.sibfly/ground-motion`)

## License

MIT
