# LlamaIndex Llms Integration: Sarvam

This is the Sarvam integration for LlamaIndex. It uses Sarvam Token Factory's
OpenAI-compatible API at `https://api.sarvam.ai/v2`. Visit
[Sarvam Token Factory](https://docs.sarvam.ai/api/token-factory) for information
on how to get an API key and which models are supported.

## Installation

```bash
pip install llama-index-llms-sarvam
```

## Usage

```python
from llama_index.llms.sarvam import Sarvam

llm = Sarvam(model="sarvam-105b", api_key="your-api-key")
response = llm.complete("Hello World!")
print(response)
```

Set the `SARVAM_API_KEY` environment variable instead of passing `api_key` if
you prefer. Override the endpoint with `SARVAM_API_BASE` or `api_base`.

## Models

| Model                       |   Context | Reasoning                      | Vision |
| --------------------------- | --------: | ------------------------------ | ------ |
| `sarvam-105b`               |   128,000 | on by default, `none` disables | no     |
| `sarvam-105b-conversations` |    32,000 | no                             | no     |
| `glm5.3`                    | 1,048,576 | low / high                     | no     |
| `glm5.3-flash`              | 1,048,576 | low / high                     | yes    |
| `gemma4`                    |   262,144 | off by default, can enable     | yes    |
| `deepseekv4.1-flash-1017`   | 1,048,576 | none / low / high / max        | yes    |

## Reasoning

`sarvam-105b` reasons by default. Reasoning tokens count toward `max_tokens`, so
a small cap can return empty content even when the request succeeds. On the v2
endpoint, disable reasoning when you need a short, deterministic answer:

```python
llm = Sarvam(
    model="sarvam-105b",
    api_key="your-api-key",
    reasoning_effort="none",
)
```

`sarvam-105b-conversations` does not reason. `gemma4` keeps reasoning off unless
you enable it. The GLM and DeepSeek models accept their own effort levels; see
the Sarvam Token Factory documentation for the current values.

## Token limits

The integration sends `max_tokens`, which is the field Sarvam's 105B models
require. Leave `max_tokens` unset to let the API use its default. The model
metadata reports the context window and output cap for each supported model.
