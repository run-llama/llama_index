# [DEPRECATED] LlamaIndex Callbacks Integration: Langfuse

> **⚠️ DEPRECATED** — This package is deprecated and will be removed in a future release.
> Please migrate to the new Langfuse integration using the instrumentation module.

## Migration Guide

Replace the deprecated callback handler with the Langfuse instrumentation module:

### Old (deprecated)

```python
from llama_index.core import set_global_handler

set_global_handler("langfuse")
```

### New (recommended)

```python
# pip install langfuse>=4.7 openinference-instrumentation-llama-index
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

LlamaIndexInstrumentor().instrument()
```

Make sure to set your environment variables `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_HOST` as shown in your [Langfuse project settings](https://cloud.langfuse.com).

For the full guide, see the [Langfuse LlamaIndex Integration](https://langfuse.com/docs/integrations/llama-index/get-started) documentation.

---

## Why was this deprecated?

Langfuse now uses LlamaIndex's [OpenInference/OpenTelemetry-based instrumentation module](https://docs.llamaindex.ai/en/stable/module_guides/observability/instrumentation/), which is the recommended tracing path for all LlamaIndex observability integrations.

Key changes:
- **Before**: `set_global_handler("langfuse")` with `llama-index-callbacks-langfuse`
- **After**: `from openinference.instrumentation.llama_index import LlamaIndexInstrumentor; LlamaIndexInstrumentor().instrument()` with `langfuse>=4.7`
- Requires Langfuse [Fast Preview](https://langfuse.com/docs/changelog#fast-preview) (available on Langfuse Cloud and self-hosted instances)

## Migration Risks

### 1. Different data shape (events vs spans)
The legacy callback emits **callback events**; `LlamaIndexInstrumentor` emits **OpenTelemetry spans** carrying [OpenInference semantic attributes](https://github.com/Arize-AI/openinference) (`llm.model_name`, `retrieval.documents[].id`, `input.value`/`output.value`, token usage). These are structurally different. Dashboards, alerts, or audit queries built against the callback event shape will not break loudly — they receive plausible span-shaped data and silently stop matching expected fields. "Traces still flowing" is a false-positive for migration success if downstream tooling depended on the callback event contract.

### 2. Double-instrumentation during transition
Do **not** run both the callback handler and the instrumentor simultaneously. With both active, one request emits duplicate traces — the same logical operation counted twice, or two conflicting spans where an audit query expects one. Enable the instrumentor and remove the callback handler in the same change; do not overlap them.
