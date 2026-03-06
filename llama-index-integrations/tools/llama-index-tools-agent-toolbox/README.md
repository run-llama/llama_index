# LlamaIndex Tools Integration: Agent Toolbox

13 production-ready tools for AI agents — web search, content extraction, screenshots, weather, finance, email validation, translation, GeoIP, news, WHOIS, DNS, PDF extraction, and QR code generation.

Get a free API key (1,000 calls/month): https://api.sendtoclaw.com/v1/auth/register

## Usage

```python
from llama_index.tools.agent_toolbox import AgentToolboxToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = AgentToolboxToolSpec(api_key="atb_your_key")

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

# Search the web
answer = await agent.run("Search for the latest AI agent frameworks")

# Get news
answer = await agent.run("What are today's top technology news?")

# Look up a domain
answer = await agent.run("Who owns the domain openai.com?")
```

## Available Tools (13)

| Tool | Description |
|------|-------------|
| `search` | Web search via DuckDuckGo |
| `extract` | Extract content from any URL |
| `screenshot` | Capture webpage screenshots |
| `weather` | Weather & forecast |
| `finance` | Stock quotes & FX rates |
| `validate_email` | Email validation (MX + SMTP) |
| `translate` | 100+ language translation |
| `geoip` | IP geolocation lookup |
| `news` | News article search |
| `whois` | Domain WHOIS lookup |
| `dns` | DNS record queries |
| `pdf_extract` | PDF text extraction |
| `qr_generate` | QR code generation |

## Links

- [Agent Toolbox API](https://github.com/Vincentwei1021/agent-toolbox)
- [API Documentation](https://api.sendtoclaw.com/docs)
- [Interactive Playground](https://api.sendtoclaw.com/playground)
