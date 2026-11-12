# Changelog

## [Unreleased] - 2026-11-12

### Added
- **Outbound client**: `BasicA2AClient` for connecting to remote A2A agents
  - `get_agent_card()` — discover agent metadata and capabilities
  - `list_skills()` — list available skills on a remote agent
  - `call_skill(name, arguments)` — invoke a skill via tasks/send
  - `send_message(message)` — send a natural language message via messages/send
- **Tool spec**: `A2AToolSpec` converts A2A skills to LlamaIndex `FunctionTool`s
  - Supports `allowed_tools` filtering
  - Supports `global_partial_params` and `partial_params_by_tool`
- **Inbound server**: `workflow_as_a2a(workflow)` exposes any LlamaIndex Workflow as an A2A agent
  - Implements JSON-RPC methods: `message/send`, `tasks/get`
  - Serves `/.well-known/agent-card.json` for discovery
  - Built on `A2AFastAPIApplication` from `a2a-sdk`
- **Workflow executor**: `WorkflowAgentExecutor(AgentExecutor)` wraps a LlamaIndex `Workflow`
- **Pydantic models**: `AgentCard`, `Skill`, `Message`, `Task`, `TaskResult`, etc.
- **Utility functions**: `aget_tools_from_a2a_card()`, `get_tools_from_a2a_card()`, `create_agent_card()`
- **Cookbook**: `examples/a2a.ipynb` demonstrating end-to-end A2A flow
- **Server example**: `examples/a2a_server.py` standalone A2A server script
- **Tests**: 13 integration tests covering server endpoints, client methods, and tool spec conversion

### Dependencies
- `a2a-sdk>=0.1.0,<1` (tested against 0.3.26)
- `llama-index-core>=0.13.0,<0.15`
- `pydantic>=2,<3`
- `httpx>=0.27,<1`