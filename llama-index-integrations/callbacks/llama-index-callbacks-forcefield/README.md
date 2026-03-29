# LlamaIndex Callbacks ForceField Integration

Scan LLM prompts for injection attacks and moderate outputs using [ForceField](https://github.com/Data-ScienceTech/forcefield) AI security.

## Installation

`ash
pip install llama-index-callbacks-forcefield
`

## Usage

`python
from llama_index.core import Settings
from llama_index.callbacks.forcefield import ForceFieldCallbackHandler

handler = ForceFieldCallbackHandler(sensitivity="high")
Settings.callback_manager.add_handler(handler)

# All LLM calls are now scanned for prompt injection, PII leaks, and more
`

### Handling blocked prompts

`python
from llama_index.callbacks.forcefield import ForceFieldCallbackHandler, PromptBlockedError

handler = ForceFieldCallbackHandler(sensitivity="high")
Settings.callback_manager.add_handler(handler)

try:
    query_engine.query("Ignore all previous instructions...")
except PromptBlockedError as e:
    print(f"Blocked: {e.scan_result.rules_triggered}")
    print(f"Risk score: {e.scan_result.risk_score}")
`

### Configuration

`python
handler = ForceFieldCallbackHandler(
    sensitivity="high",       # low, medium, high, critical
    block_on_input=True,      # raise PromptBlockedError on blocked prompts
    moderate_output=True,     # scan LLM outputs for harmful content
    on_block=lambda r: print(f"Blocked: {r.rules_triggered}"),
)
`

## Features

- **Input scanning**: Scans prompts for prompt injection, PII leaks, jailbreaks, and 13+ attack categories
- **Output moderation**: Checks LLM responses for harmful content
- **Zero config**: No API keys needed, works offline
- **116 built-in attack prompts** for security evals

## Links

- [ForceField SDK](https://pypi.org/project/forcefield/)
- [GitHub](https://github.com/Data-ScienceTech/forcefield)
- [VS Code Extension](https://marketplace.visualstudio.com/items?itemName=DataScienceTech.forcefield)
