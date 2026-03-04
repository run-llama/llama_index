from typing import Any

from llama_index.core.callbacks.base_handler import BaseCallbackHandler


def arize_phoenix_callback_handler(**kwargs: Any) -> BaseCallbackHandler:
    # newer versions of arize, v2.x
    try:
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk import trace as trace_sdk
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        endpoint = kwargs.get("endpoint", "http://127.0.0.1:6006/v1/traces")
        tracer_provider = trace_sdk.TracerProvider()
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(OTLPSpanExporter(endpoint))
        )

        return LlamaIndexInstrumentor().instrument(
            tracer_provider=kwargs.get("tracer_provider", tracer_provider),
            separate_trace_from_runtime_context=kwargs.get(
                "separate_trace_from_runtime_context"
            ),
        )
    except ImportError:
        # using an older version of arize
        pass

    # older versions of arize, v1.x
    try:
        from phoenix.trace.llama_index import OpenInferenceTraceCallbackHandler
    except ImportError:
        raise ImportError(
            "Please install Arize Phoenix with `pip install -q arize-phoenix`"
        )
    return OpenInferenceTraceCallbackHandler(**kwargs)
