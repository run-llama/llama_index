# LlamaIndex Tools Integration: Famulor

![Famulor](https://www.famulor.io/images/famulor-fav.png)

[Famulor](https://www.famulor.io) is a platform for AI voice and phone agents.
This tool calls the public Platform v1 REST API at `https://app.famulor.io/api/v1`.

Sign up and create an API key at [https://app.famulor.io](https://app.famulor.io)
under **Settings → API & MCP**. Keys look like `fam_...`. The workspace is
encoded in the key, so requests only send `Authorization: Bearer`.

API reference: [https://docs.famulor.io/api-reference/introduction](https://docs.famulor.io/api-reference/introduction).

## Installation

```bash
pip install llama-index-tools-famulor
```

## API key

```bash
export FAMULOR_API_KEY="fam_your_key"
```

A white-label host uses the same paths:

```python
from llama_index.tools.famulor import FamulorToolSpec

tool = FamulorToolSpec(
    api_key="fam_your_key",
    base_url="https://voice.example.com/api/v1",
)
```

Dashboard keys include every scope. A scoped key needs `assistants:read`,
`phone_numbers:read`, `calls:read`, `calls:write` (for `make_call`),
`leads:read`, `leads:write`, `campaigns:read`, and `campaigns:write` for the
methods you enable.

## Usage

```python
import os

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.famulor import FamulorToolSpec

famulor_tool = FamulorToolSpec(api_key=os.environ["FAMULOR_API_KEY"])

agent = FunctionAgent(
    tools=famulor_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
)

# Read assistants. This does not place a call.
print(famulor_tool.list_assistants())
```

`make_call` places a **real outbound phone call** through `POST /calls`.
The assistant dials the E.164 `to_number`. It is not a simulation. The API
returns the call with status `queued`; use `get_call` for the transcript and
summary.

```python
famulor_tool.make_call(
    assistant_id="a1b2c3d4-0000-4000-8000-000000000001",
    to_number="+4930123456",
)
```

## Tools

| Method               | API                                  | Docs                                                                                                 |
| -------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `list_assistants`    | `GET /assistants`                    | [List assistants](https://docs.famulor.io/api-reference/assistants/list-assistants)                  |
| `get_assistant`      | `GET /assistants/{id}`               | [Get an assistant](https://docs.famulor.io/api-reference/assistants/get-an-assistant)                |
| `list_phone_numbers` | `GET /phone-numbers`                 | [List phone numbers](https://docs.famulor.io/api-reference/phone-numbers/list-phone-numbers)         |
| `list_calls`         | `GET /calls`                         | [List calls](https://docs.famulor.io/api-reference/calls/list-calls)                                 |
| `get_call`           | `GET /calls/{id}`                    | [Get a call](https://docs.famulor.io/api-reference/calls/get-a-call)                                 |
| `list_leads`         | `GET /leads`                         | [List Audience contacts](https://docs.famulor.io/api-reference/leads/list-audience-contacts)         |
| `create_lead`        | `POST /leads`                        | [Create an Audience contact](https://docs.famulor.io/api-reference/leads/create-an-audience-contact) |
| `get_lead`           | `GET /campaigns/{id}/leads/{leadId}` | [Get a campaign lead](https://docs.famulor.io/api-reference/leads/get-a-campaign-lead)               |
| `list_campaigns`     | `GET /campaigns`                     | [List campaigns](https://docs.famulor.io/api-reference/campaigns/list-campaigns)                     |
| `get_campaign`       | `GET /campaigns/{id}`                | [Get a campaign](https://docs.famulor.io/api-reference/campaigns/get-a-campaign)                     |
| `create_campaign`    | `POST /campaigns`                    | [Create a campaign](https://docs.famulor.io/api-reference/campaigns/create-a-campaign)               |
| `make_call`          | `POST /calls`                        | [Start an outbound call](https://docs.famulor.io/api-reference/calls/start-an-outbound-call)         |

`create_campaign` creates a draft and does not start dialing. `create_lead`
writes a contact and does not place a call. Campaign start and stop are not
wrapped.

A longer example is in
[`examples/famulor_agent.py`](examples/famulor_agent.py).
