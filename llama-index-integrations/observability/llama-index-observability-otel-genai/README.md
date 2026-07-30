# LlamaIndex OpenTelemetry GenAI Observability Integration

This optional integration maps LlamaIndex's native instrumentation events to
OpenTelemetry GenAI semantic-convention spans and metrics.

```python
from llama_index.observability.otel_genai import LlamaIndexOtelGenAIInstrumentor

instrumentor = LlamaIndexOtelGenAIInstrumentor()
instrumentor.start_registering()
```

The initial release instruments LLM chat/completion and embedding operations.
Message content is controlled by the standard
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` setting and is not
captured by default.
