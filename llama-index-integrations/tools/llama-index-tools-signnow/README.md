# llama-index-tools-signnow

> **A plug‑and‑play ToolSpec that lets LlamaIndex agents use SignNow e‑signature workflows through the [SignNow MCP server](https://github.com/signnow/sn-mcp-server).**
> It discovers the server’s tools, exposes them to your agent, and keeps setup simple for users who already work with LlamaIndex.

---

## ✨ What you get

- **One‑liner tool discovery** for the SignNow MCP server (STDIO spawn by default).
- **Works with any LlamaIndex agent** that supports function/tools.
- **Covers common e‑signature flows** end‑to‑end (templates, invites, embedded signing/sending/editor, status, downloads).
- **Environment‑first configuration** (no code changes required for typical deployments).

---

## Installation

```bash
# Core LlamaIndex + MCP client + SignNow ToolSpec
pip install -U llama-index llama-index-tools-mcp llama-index-tools-signnow
```

## Quick start

```python
import asyncio
from llama_index.tools.signnow import SignNowMCPToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import (
    OpenAI,
)  # or any LLM that supports tool/function calling


async def main():
    # Option A: pass credentials directly (no .env needed)
    spec = SignNowMCPToolSpec.from_env(
        env_overrides={
            # Option 1: token-based auth
            # "SIGNNOW_TOKEN": "your_signnow_token_here",
            # Option 2: credential-based auth
            "SIGNNOW_USER_EMAIL": "login@example.com",
            "SIGNNOW_PASSWORD": "password",
            "SIGNNOW_API_BASIC_TOKEN": "basic_token_base64",
        }
    )

    # Fetch tools from the MCP server
    tools = await spec.to_tool_list_async()
    print({"count": len(tools), "names": [t.metadata.name for t in tools]})

    # Wire them into a LlamaIndex agent
    agent = FunctionAgent(
        name="SignNow Agent",
        description="Query SignNow via MCP tools",
        tools=tools,
        llm=OpenAI(model="gpt-4o"),  # make sure your LLM supports tools
        system_prompt="Be helpful.",
    )

    # Ask for something useful
    resp = await agent.run("Show me the list of templates and their names.")
    print(resp)


if __name__ == "__main__":
    asyncio.run(main())
```

### Prompt ideas

- “Create a document from the template ‘NDA v3’ and generate an embedded signing link for alice@example.com”
- “Send an invite to bob@example.com for document XYZ with signer order Alice first, Bob second.”
- “What’s the current invite status for document ABC?”
- “Give me a direct download link for the completed document (merged if it’s a group).”

---

## Available tools (provided by the SignNow MCP server)

- list_all_templates — List templates & template groups with simplified metadata.
- list_document_groups — Browse your document groups and statuses.
- create_from_template — Make a document or a group from a template/group.
- send_invite — Email invites (documents or groups), ordered recipients supported.
- create_embedded_invite — Embedded signing session without email delivery.
- create_embedded_sending — Embedded “sending/management” experience.
- create_embedded_editor — Embedded editor link to place/adjust fields.
- send_invite_from_template — One‑shot: create from template and invite.
- create_embedded_sending_from_template — One‑shot: template → embedded sending.
- create_embedded_editor_from_template — One‑shot: template → embedded editor.
- create_embedded_invite_from_template — One‑shot: template → embedded signing.
- get_invite_status — Current invite status/steps for document or group.
- get_document_download_link — Direct download link (merged output for groups).
- get_document — Normalized document/group structure with field values.
- update_document_fields — Prefill text fields in individual documents.

> Tip: A common flow is list*all_templates → create_from_template → one‑shot or send_invite / create_embedded*\* → get_invite_status → get_document_download_link.

---

## Configuration

The ToolSpec reads standard SignNow environment variables and forwards them to the server when spawning `sn-mcp`

### Auth options

```env
# Username / Password (recommended for desktop dev flows)
SIGNNOW_USER_EMAIL=you@example.com
SIGNNOW_PASSWORD=********
SIGNNOW_API_BASIC_TOKEN=base64(app_id:app_secret)  # SignNow Basic token

# or a direct API token
SIGNNOW_TOKEN=eyJhbGciOi...
```

---

## How it works (under the hood)

`SignNowMCPToolSpec` wraps the generic MCP client from `llama-index-tools-mcp`.
On `from_env(...)` it spawns the `sn-mcp` server (STDIO) with your environment and converts the advertised MCP tools into LlamaIndex `FunctionTools` for your agent to call.
