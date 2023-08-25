# Tracing with Graphsignal

[Graphsignal](https://graphsignal.com/) provides observability for AI agents and LLM-powered applications. It helps developers ensure AI applications run as expected and users have the best experience.

Graphsignal **automatically** traces and monitors LlamaIndex. Traces and metrics provide execution details for query, retrieval, and index operations. These insights include **prompts**, **completions**, **embedding statistics**, **retrieved nodes**, **parameters**, **latency**, and **exceptions**.

When OpenAI APIs are used, Graphsignal provides additional insights such as **token counts** and **costs** per deployment, model or any context.


### Installation and Setup

Adding [Graphsignal tracer](https://github.com/graphsignal/graphsignal-python) is simple, just install and configure it:

```sh
pip install graphsignal
```

```python
import graphsignal

# Provide an API key directly or via GRAPHSIGNAL_API_KEY environment variable
graphsignal.configure(api_key='my-api-key', deployment='my-llama-index-app-prod')
```

You can get an API key [here](https://app.graphsignal.com/).

See the [Quick Start guide](https://graphsignal.com/docs/guides/quick-start/), [Integration guide](https://graphsignal.com/docs/integrations/llama-index/), and an [example app](https://github.com/graphsignal/examples/blob/main/llama-index-app/main.py) for more information.


### Tracing Other Functions

To additionally trace any function or code, you can use a decorator or a context manager:

```python
with graphsignal.start_trace('load-external-data'):
    reader.load_data()
```

See [Python API Reference](https://graphsignal.com/docs/reference/python-api/) for complete instructions.


### Useful Links

* [Tracing and Monitoring LlamaIndex Applications](https://graphsignal.com/blog/tracing-and-monitoring-llama-index-applications/)
* [Monitor OpenAI API Latency, Tokens, Rate Limits, and More](https://graphsignal.com/blog/monitor-open-ai-api-latency-tokens-rate-limits-and-more/)
* [OpenAI API Cost Tracking: Analyzing Expenses by Model, Deployment, and Context](https://graphsignal.com/blog/open-ai-api-cost-tracking-analyzing-expenses-by-model-deployment-and-context/)
