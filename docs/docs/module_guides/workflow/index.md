# Workflows

A `Workflow` in LlamaIndex is an event-driven abstraction used to chain together several events. Workflows are made up
of `steps`, with each step responsible for handling certain event types and emitting new events.

You can create a `Workflow` to do anything! Build an agent, a RAG flow, an extraction flow, or anything else you want.

Workflows are also automatically instrumented, so you get observability into each step using tools like [Arize Pheonix](../observability/index.md#arize-phoenix-local). (**NOTE:** Observability works for integrations that take advantage of the newer instrumentation system. Usage may vary.)


!!! important
    The Workflows library can be installed standalone, via `pip install llama-index-workflows`. However,
    `llama-index-core` comes with a stable version of Workflows included.

    When installing `llama-index-core` or the `llama-index` umbrella package, Workflows can be accessed with the import
    path `llama_index.core.workflow`. In order to maintain the `llama_index` API stable and avoid breaking changes,
    the Workflows library version included is usually older than the latest version of `llama-index-workflows`.

    At the moment, the latest version of `llama-index-workflows` is 2.0 while the one shipped with `llama-index` is
    1.3

- [v1.x Documentation](./v1/index.md)
- [v2.x Documentation](./v2/index.md)
