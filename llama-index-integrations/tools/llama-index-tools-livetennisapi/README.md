# Live Tennis API Tool

This tool connects to the [Live Tennis API](https://livetennisapi.com) through the official
[`livetennisapi`](https://pypi.org/project/livetennisapi/) Python client. It gives an agent
real-time tennis scores, players, rankings, fixtures and results for ATP, WTA, Challenger and ITF.

You initialize the tool with your API key. Get a free key (1000 requests/day) at
<https://livetennisapi.com/subscribe/free>. The key can also be supplied via the
`LIVETENNISAPI_KEY` environment variable.

The tool has access to the following functions:

Free tier:

- `get_live_matches`: tennis matches currently in progress, with live scores
- `get_upcoming_matches`: matches scheduled to start soon
- `get_match`: full detail for one match by id
- `search_players`: search players by name
- `get_player`: one player's profile
- `get_fixtures`: the forward schedule of scheduled fixtures
- `check_status`: whether the API is reachable and which plan the key is on

Higher tiers (these return a plain-English "requires the … plan" message on a free key rather
than raising):

- `get_recent_results`: recently completed matches (BASIC)
- `get_match_events`: a match's event timeline (PRO)
- `get_match_odds`: match-winner market prices (PRO)
- `get_match_analysis`: model win probability and thesis (ULTRA)

## Usage

```python
from llama_index.tools.livetennisapi import LiveTennisAPIToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = LiveTennisAPIToolSpec(api_key="twjp_...")

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(await agent.run("What tennis matches are live right now?"))
print(await agent.run("What is Carlos Alcaraz's current ranking?"))
```

You can also call the tools directly:

```python
tool_spec = LiveTennisAPIToolSpec(api_key="twjp_...")
print(tool_spec.get_live_matches(limit=5))
print(tool_spec.search_players("alcaraz"))
```

This loader is designed to be used as a way to load data as a Tool in an Agent.
