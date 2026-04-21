---
title: Semantic Validation with semantix-ai
---

[semantix-ai](https://github.com/labrat-akhona/semantix-ai) is a semantic type system for AI outputs. It validates that LLM outputs satisfy natural language intents — checking meaning, not just shape — using local NLI (Natural Language Inference) models with no API cost and ~15ms latency.

Where schema validation checks structure ("is this a string?"), semantix validates semantics ("is this polite and professional?"). It runs entirely locally, so there are no external API calls and no data leaves your environment.

### Installation and Setup

```sh
pip install semantix-ai
```

No additional configuration is required. semantix downloads a small NLI model on first use and caches it locally.

### Inline Validation with a LlamaIndex Query Engine

Use `assert_semantic` to validate the string output of any LlamaIndex query engine against a natural language constraint:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from semantix.testing import assert_semantic

# Build a simple RAG query engine
documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Query the engine
response = query_engine.query("Summarize the key findings.")
output = str(response)

# Validate — raises AssertionError on failure
assert_semantic(output, "must be a clear, factual summary with no speculation")
```

### Using Intent Classes with `@validate_intent`

For reusable validation, define an `Intent` subclass and use the `@validate_intent` decorator. The function's return type annotation tells semantix which intent to enforce:

```python
from semantix import Intent, validate_intent

class FactualSummary(Intent):
    """The text must be a clear, factual summary with no speculation."""

@validate_intent
def summarize_docs(query_engine, question: str) -> FactualSummary:
    response = query_engine.query(question)
    return str(response)  # returns a plain string; decorator validates it

result = summarize_docs(query_engine, "What are the key findings?")
```

### Composite Intents

Combine multiple constraints with the `&` operator to build richer validation:

```python
from semantix import Intent

class Polite(Intent):
    """The text must be polite and professional."""

class Factual(Intent):
    """The text must be factual with no speculation."""

# AllOf — output must satisfy both intents
PoliteFactual = Polite & Factual

@validate_intent
def polite_summary(query_engine, question: str) -> PoliteFactual:
    return str(query_engine.query(question))
```

### Pluggable Judges

By default semantix uses its built-in NLI judge. You can swap in other judges for different tradeoffs:

```python
from semantix import NLIJudge, EmbeddingJudge, LLMJudge
from semantix.testing import assert_semantic

output = str(query_engine.query("List the risks in the document."))

# Use the NLI judge explicitly
assert_semantic(output, "must identify at least one specific risk", judge=NLIJudge())

# Or use the embedding-based judge
assert_semantic(output, "must use neutral, factual language", judge=EmbeddingJudge())
```

### Why Use semantix with LlamaIndex?

- **No API cost** — inference runs on a local NLI model (~15ms per check)
- **Intent-based validation** — catches hallucinations, tone drift, and off-topic responses that schema checks miss
- **Zero configuration** — drop it in after any `query_engine.query()` call
- **MIT licensed**

### Useful Links

- [semantix-ai on PyPI](https://pypi.org/project/semantix-ai/)
- [semantix-ai on GitHub](https://github.com/labrat-akhona/semantix-ai)
- [LlamaIndex Query Engine docs](https://docs.llamaindex.ai/en/stable/understanding/querying/querying/)
