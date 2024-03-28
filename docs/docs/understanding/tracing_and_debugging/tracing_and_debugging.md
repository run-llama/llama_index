# Tracing and Debugging

Debugging and tracing the operation of your application is key to understanding and optimizing it. LlamaIndex provides a variety of ways to do this.

## Basic logging

The simplest possible way to look into what your application is doing is to turn on debug logging. That can be done anywhere in your application like this:

```python
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
```

## Callback handler

LlamaIndex provides callbacks to help debug, track, and trace the inner workings of the library. Using the callback manager, as many callbacks as needed can be added.

In addition to logging data related to events, you can also track the duration and number of occurrences
of each event.

Furthermore, a trace map of events is also recorded, and callbacks can use this data however they want. For example, the `LlamaDebugHandler` will, by default, print the trace of events after most operations.

You can get a simple callback handler like this:

```python
import llama_index.core

llama_index.core.set_global_handler("simple")
```

You can also learn how to [build you own custom callback handler](../../module_guides/observability/callbacks/index.md).

## Observability

LlamaIndex provides **one-click observability** to allow you to build principled LLM applications in a production setting.

This feature allows you to seamlessly integrate the LlamaIndex library with powerful observability/evaluation tools offered by our partners. Configure a variable once, and you'll be able to do things like the following:

- View LLM/prompt inputs/outputs
- Ensure that the outputs of any component (LLMs, embeddings) are performing as expected
- View call traces for both indexing and querying

To learn more, check out our [observability docs](../../module_guides/observability/index.md)
