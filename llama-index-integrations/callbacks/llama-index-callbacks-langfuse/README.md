# LlamaIndex Callbacks Integration: Langfuse

[Langfuse](https://langfuse.com/docs) is an open source LLM engineering platform to help teams collaboratively debug, analyze and iterate on their LLM Applications. With the Langfuse integration, you can seamlessly track and monitor performance, traces, and metrics of your LlamaIndex application. Detailed traces of the LlamaIndex context augmentation and the LLM querying processes are captured and can be inspected directly in the Langfuse UI.

#### Usage Pattern

```python
import llama_index.core.instrumentation as inst
from llama_index.callbacks.langfuse import LangfuseSpanHandler

langfuse_span_handler = LangfuseSpanHandler(
    public_key="<Your public key>",
    secret_key="<Your secret key>",
    host="<Host URL>",
)
dispatcher = inst.get_dispatcher()
dispatcher.add_span_handler(langfuse_span_handler)
```

![langfuse-tracing](https://static.langfuse.com/llamaindex-langfuse-docs.gif)
