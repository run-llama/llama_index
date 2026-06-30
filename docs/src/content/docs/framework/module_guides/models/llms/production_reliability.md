---
title: Production Reliability Patterns for Custom LLMs
---

When deploying custom LLM wrappers in production, you'll encounter failure modes that don't appear during development. This guide covers defensive patterns that prevent the most common production issues.

## The Problem: Silent Failures

A minimal custom LLM wrapper works perfectly in development. In production, you'll encounter:

- **Empty responses**: The API returns HTTP 200 but the response body is empty or truncated
- **Schema violations**: The model returns valid JSON but with missing or extra fields
- **Latency spikes**: Network congestion causes timeouts that cascade through your agent pipeline
- **Model deprecation**: A model name becomes invalid after a provider update

None of these necessarily raise exceptions. Your `try/except` block won't catch them.

## Pattern 1: Response Validation

Always validate the response structure before returning it:

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata

class ReliableLLM(CustomLLM):
    context_window: int = 8192
    num_output: int = 4096
    model_name: str = "custom-reliable"

    def __init__(self, api_key: str, model: str, **kwargs):
        super().__init__(**kwargs)
        self.client = self._build_client(api_key)
        self.model = model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(lambda e: isinstance(e, (TimeoutError, ConnectionError)))
    )
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            timeout=30,  # Always set explicit timeouts
        )

        # Validate response content — catches silent failures
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("Empty response from model — API returned HTTP 200 with no content")

        return CompletionResponse(text=content)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )
```

**Why this matters:** Without the empty-content check, an HTTP 200 with an empty body passes through as a "successful" response. Your agent then processes empty data, and the failure silently propagates downstream.

## Pattern 2: Graceful Degradation with Fallback

When your primary provider fails, switch to a backup — but validate the backup's response too:

```python
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata

class ResilientLLM(CustomLLM):
    context_window: int = 8192
    num_output: int = 4096
    model_name: str = "resilient-multi-provider"

    def __init__(self, providers: list[dict]):
        super().__init__()
        self.providers = providers  # [{"client": ..., "model": ...}, ...]

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        last_error = None

        for provider in self.providers:
            try:
                response = provider["client"].chat.completions.create(
                    model=provider["model"],
                    messages=[{"role": "user", "content": prompt}],
                    timeout=30,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return CompletionResponse(text=content)
            except Exception as e:
                last_error = e
                continue  # Try next provider

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )
```

**Key insight:** Simply switching providers isn't enough. If the original request had an invalid model name or malformed body, the backup provider will fail the same way. You need to **diagnose the failure** before retrying — not just retry blindly.

## Pattern 3: Contract Validation for Agent Pipelines

For agentic workflows, validate that the LLM output conforms to the contract your pipeline expects:

```python
import json

def validate_agent_output(text: str, expected_schema: dict) -> bool:
    """Validate that LLM output matches the expected contract."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False

    # Check required fields
    for field in expected_schema.get("required", []):
        if field not in parsed:
            return False

    # Check field types
    for field, ftype in expected_schema.get("properties", {}).items():
        if field in parsed and not isinstance(parsed[field], ftype):
            return False

    return True

# Usage in your agent loop
expected = {
    "required": ["action", "parameters"],
    "properties": {"action": str, "parameters": dict}
}

response = llm.complete("Decide the next action...")
if not validate_agent_output(response.text, expected):
    # Don't silently pass invalid output to the next step
    raise ValueError(f"Agent output failed validation: {response.text[:200]}")
```

## Diagnosing Failures Before Retrying

When a fallback fails the same way as the primary, the root cause is usually in the **request**, not the provider:

| Failure Mode | Symptom | Root Cause | Fix Before Retry |
|---|---|---|---|
| Invalid model name | HTTP 400 on all providers | Typo, deprecated model, or meta-agent hallucination | Validate model name against provider's model list |
| Empty body | HTTP 200 with no content | Serialization bug, dropped payload | Reconstruct request body before retry |
| Aggressive timeout | Timeout on all providers | Network congestion, overloaded endpoint | Adjust timeout based on provider's P95 latency |
| Token budget overrun | HTTP 400 with token limit error | Prompt too long for model's context window | Truncate or compress prompt before retry |

Standard failover sends the same broken request to a different provider and gets the same broken result. Reliable systems **diagnose** before they **retry**.

## Key Takeaways

1. **Always set explicit timeouts** — never rely on default (which may be infinite)
2. **Validate response content**, not just HTTP status codes
3. **Use exponential backoff** for retries, with a maximum attempt count
4. **Fail loudly** — raise exceptions for invalid responses rather than passing them downstream
5. **Diagnose before retrying** — understand WHY a call failed before trying again
6. **Test with fault injection** — your staging environment should simulate timeouts, empty responses, and invalid model names

For complex production environments with multiple providers, dynamic model routing, and automated failure recovery, consider evaluating dedicated reliability frameworks such as [Correctover](https://github.com/Correctover/correctover-sdk) that implement autonomic healing loops on top of these patterns.
