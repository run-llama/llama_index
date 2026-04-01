# LlamaIndex Llms Integration: HPC-AI

Integration for [HPC-AI](https://api.hpc-ai.com/) inference using an OpenAI-compatible HTTP API.

## Supported models

Example model IDs (pass via the `model` argument):

- `minimax/minimax-m2.5`
- `moonshotai/kimi-k2.5`

The default model is `minimax/minimax-m2.5`. Set an appropriate `context_window` for your model if the default is not correct for your deployment.

## Install

```bash
pip install llama-index-llms-hpc-ai
```

## Usage

```python
from llama_index.llms.hpc_ai import HpcAiLLM

llm = HpcAiLLM(
    model="moonshotai/kimi-k2.5",
    api_key="your-api-key",
)

response = llm.complete("Hello, world!")
print(response)
```

## Environment variables

| Variable | Description |
|----------|-------------|
| `HPC_AI_API_KEY` | API key (used when `api_key` is not passed to the constructor). |
| `HPC_AI_BASE_URL` | Override API base URL (default: `https://api.hpc-ai.com/inference/v1`). |

Constructor arguments take precedence over environment variables when provided.

## Develop

From this directory, install in editable mode (with dev deps via `uv` if you use the monorepo workflow):

```bash
pip install -e .
```

## Testing

```bash
make test
```

Integration tests run only when `HPC_AI_API_KEY` is set in the environment.

## Linting and formatting

```bash
make format
make lint
```
