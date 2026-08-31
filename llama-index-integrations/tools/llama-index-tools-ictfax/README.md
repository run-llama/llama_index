# LlamaIndex Tools Integration: ICTFax

Send and track faxes from a LlamaIndex agent through an [ICTFax](https://ictfax.org)
server. ICTFax is an open source fax server built on the ICTCore framework; the same
REST API also backs ICTPBX and ICTDialer.

## Tools

- `upload_fax_document(file_path, name)` — upload a PDF/TIFF/image, returns a document id
- `send_fax(to_number, document_id, title, recipient_name)` — send a fax and return its transmission id
- `get_fax_status(transmission_id)` — poll delivery status
- `list_faxes()` — list transmissions with their status

## Usage

```python
from llama_index.tools.ictfax import ICTFaxToolSpec
from llama_index.agent.openai import OpenAIAgent

tool_spec = ICTFaxToolSpec(
    base_url="https://fax.example.com/api",
    username="admin",
    password="mysecret",
)

agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())
agent.chat("Fax document 12 to +12025550123 and tell me when it goes through")
```

This tool is a client for an ICTFax server, so it needs a reachable ICTFax/ICTCore
install and an API account. It does not deliver faxes on its own.

---

Built and maintained by [ICT Innovations](https://www.ictinnovations.com/) — the team behind [ICTFax](https://ictfax.org), ICTPBX, ICTDialer and ICTContact.
