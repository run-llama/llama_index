# OpenRegistry Tool

This tool gives a LlamaIndex agent live, real-time access to **27 national company registries** through the hosted [OpenRegistry](https://openregistry.sophymarine.com) MCP server — UK Companies House, France RNE, Germany Handelsregister, Italy InfoCamere via EU BRIS, Spain BORME, Korea OpenDART, and 21 more.

Every tool call is a real-time query against the upstream government API at the moment the agent asks. No data is cached or aggregated, so the agent always sees the registry's own response with all field names preserved.

## Installation

```bash
pip install llama-index-tools-openregistry
```

## Usage

Anonymous use — no signup, no API key:

```python
from llama_index.tools.openregistry import OpenRegistryToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = OpenRegistryToolSpec()

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

await agent.run(
    "Find Tesco PLC on Companies House and walk its corporate ownership chain"
    " across jurisdictions until you reach the ultimate beneficial owner."
)
```

## Authenticated tier

The free anonymous tier allows 20 calls/min per IP and a cross-border fan-out of 3 countries / 60s. For higher limits, complete the OAuth 2.1 flow at [openregistry.sophymarine.com/account](https://openregistry.sophymarine.com/account) and pass the resulting bearer token:

```python
tool_spec = OpenRegistryToolSpec(oauth_token=OPENREGISTRY_TOKEN)
```

## Restricting tools

OpenRegistry exposes ~30 tools (search, profile, officers, PSCs, charges, filings, document download, jurisdiction-specific specialised records, etc.). To keep the agent's tool list lean, allowlist just the ones you need:

```python
tool_spec = OpenRegistryToolSpec(
    allowed_tools=[
        "search_companies",
        "get_company_profile",
        "get_officers",
        "get_persons_with_significant_control",
    ],
)
```

## Tools available

The full per-jurisdiction capability matrix is published at [`list_jurisdictions`](https://openregistry.sophymarine.com/jurisdictions) — each registry advertises which subset of the tool surface it supports. Highlights:

- `search_companies` — name / number / address search across any single jurisdiction
- `get_company_profile` — statutory profile (status, address, incorporation date, accounting reference date, etc.)
- `get_officers` — directors and secretaries with full appointment history
- `get_persons_with_significant_control` — UK PSC Register and equivalents
- `get_shareholders` — shareholder lists where the registry publishes them
- `get_charges` — registered charges / mortgages
- `list_filings` / `fetch_document` — raw XHTML iXBRL, PDF, or XBRL filing bytes
- `get_financials` — latest statutory accounts as machine-readable XBRL where available
- Jurisdiction-specific tools for KR (OpenDART), DE (Handelsregister), ES (BORME / actos inscritos), IT (InfoCamere via BRIS), and others

## License

MIT — see [LICENSE](LICENSE).

OpenRegistry is a platform by [Sophymarine](https://sophymarine.com). The hosted service is not affiliated with any of the national company registries it proxies.
