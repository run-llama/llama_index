# LlamaIndex Llms Integration: OrcaRouter

[OrcaRouter](https://www.orcarouter.ai) is an OpenAI-compatible meta-router
that exposes 150+ upstream models behind a single API key, with an adaptive
router (`orcarouter/auto`) that picks the cheapest / fastest / highest-quality
upstream per request via a learned contextual bandit policy.

## Installation

```bash
%pip install llama-index-llms-orcarouter
!pip install llama-index
```

## Setup

Set the environment variable `ORCAROUTER_API_KEY`, or pass `api_key` to the
constructor. Get a key at [orcarouter.ai](https://www.orcarouter.ai).

```python
from llama_index.llms.orcarouter import OrcaRouter
from llama_index.core.llms import ChatMessage

# Default: use the adaptive router. OrcaRouter picks the best upstream.
llm = OrcaRouter(
    api_key="<your-api-key>",
    max_tokens=256,
    context_window=128000,
)
```

## Generate Chat Responses

```python
message = ChatMessage(role="user", content="Tell me a joke")
resp = llm.chat([message])
print(resp)
```

### Streaming Responses

```python
message = ChatMessage(role="user", content="Tell me a story in 250 words")
for r in llm.stream_chat([message]):
    print(r.delta, end="")
```

### Completion

```python
resp = llm.complete("Tell me a joke")
print(resp)
```

## Model Configuration

Pin a specific upstream model with the `<namespace>/<model>` syntax:

```python
llm = OrcaRouter(model="anthropic/claude-opus-4.7")
```

See the full catalog at [orcarouter.ai/models](https://www.orcarouter.ai/models).

## Adaptive Routing (`orcarouter/auto`)

The default model `orcarouter/auto` is not a model but a *router*. It applies
one of five strategies (`cheapest`, `balanced`, `quality`, `adaptive` LinUCB,
`gated_adaptive`) configured per-workspace at
[orcarouter.ai/console/routing](https://www.orcarouter.ai/console/routing).
Same endpoint, different upstream chosen per request based on prompt features
(length, code/math/JSON density, declared `max_tokens` budget, similarity to
recent traffic).

## Fallback List

Provide a fallback chain via `fallback_models` — if the primary model fails,
OrcaRouter walks the list in order:

```python
llm = OrcaRouter(
    model="openai/gpt-4o-mini",
    fallback_models=["openai/gpt-4o", "anthropic/claude-sonnet-4.6"],
)
```

This is sent as the OrcaRouter-specific
`extra_body = {"models": [...], "route": "fallback"}` request extension.

## Reasoning Models

OrcaRouter exposes reasoning models like `anthropic/claude-opus-4.7`,
`openai/gpt-5`, and `deepseek/deepseek-reasoner`. These models reject the
`temperature` parameter — pass `temperature=None` or omit it.

```python
llm = OrcaRouter(
    model="anthropic/claude-opus-4.7",
    temperature=None,
)
```

Per-vendor reasoning controls (e.g. `reasoning_effort` for OpenAI o-series,
`thinking` block for Anthropic) can be passed via `additional_kwargs`. See
[docs.orcarouter.ai](https://docs.orcarouter.ai).
