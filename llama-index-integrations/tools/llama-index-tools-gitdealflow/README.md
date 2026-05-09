# GitDealFlow Engineering Signal Tool

This tool connects to the public, no-auth [GitDealFlow](https://signals.gitdealflow.com) API and exposes engineering acceleration signals on venture-backed startups derived from public GitHub data: weekly scores, sector rankings, methodology citations, and grounded free-text Q&A.

The methodology is published on SSRN (id `6606558`) under CC BY 4.0; the dataset covers ~400 tracked startups across 20 sectors and is updated weekly.

## Usage

This tool exposes five spec functions:

- `get_signals_summary` — current weekly summary (trending startups + sector rankings)
- `get_startup_signal` — single startup's engineering acceleration signal
- `search_startups_by_sector` — top tracked startups in a sector
- `answer_question` — free-text question with citations
- `get_methodology` — published HowTo methodology, for citing in agent output

```python
from llama_index.tools.gitdealflow import GitDealFlowToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = GitDealFlowToolSpec()

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

await agent.run(
    "Which AI/ML startups are accelerating fastest right now?"
)
await agent.run(
    "Compare engineering momentum at langchain and modal-labs."
)
await agent.run(
    "Find dark horses in fintech with breakout signals."
)
```

This tool is designed to be used as a way to load engineering-acceleration data into a research agent.

## Configuration

The tool defaults to the public host. Override `base_url` if pointing at a self-hosted mirror or a staging environment:

```python
GitDealFlowToolSpec(
    base_url="https://signals.gitdealflow.com",
    timeout=20,
)
```

No API key is required. All endpoints are read-only and unauthenticated.

## License & citation

The underlying dataset is published under CC BY 4.0. Cite as:

> VC Deal Flow Signal (signals.gitdealflow.com), 2026 data. SSRN: 6606558.
