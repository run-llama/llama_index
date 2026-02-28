# CHANGELOG

## [0.2.0] - 2026-02-26

- Add `upload_file`, `upload_files`, `install_packages`, `download_file`, `download_files` tools to Code Interpreter
- Add `generate_live_view_url` tool to Browser
- Add browser lifecycle management tools: `list_browsers`, `create_browser`, `delete_browser`, `get_browser`
- Add browser collaboration tools: `take_control`, `release_control`
- Add code interpreter lifecycle management tools: `list_code_interpreters`, `create_code_interpreter`, `delete_code_interpreter`, `get_code_interpreter`
- Add standalone `clear_context` tool for code interpreter
- Add `identifier` parameter for custom/VPC-enabled resources
- Add VPC configuration support (`subnet_ids`, `security_group_ids`) for `create_browser` and `create_code_interpreter`
- Add `integration_source="llamaindex"` telemetry to all SDK clients
- Bump `bedrock-agentcore` minimum version to `>=1.2.0`
- Browser tool count: 8 → 14, Code Interpreter tool count: 9 → 19

## [0.1.2] - 2026-02-13

- Add `delete_files` tool to Code Interpreter
- Add maintainers and keywords from library.json (llamahub)
