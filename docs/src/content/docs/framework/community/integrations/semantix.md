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

### Basic Usage with a LlamaIndex Query Engine

You can validate the string output of any LlamaIndex query engine against a natural language intent:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from semantix import Semantic, validate_output

# Build a simple RAG query engine
documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Query the engine
response = query_engine.query("Summarize the key findings.")
output = str(response)

# Validate the output against a natural language intent
result = validate_output(
    output=output,
    intent="must be a clear, factual summary with no speculation",
)

if result.is_valid:
    print("Output passed semantic validation.")
else:
    print(f"Validation failed: {result.reason}")
```

### Using Semantic Types

semantix lets you declare intent directly in type annotations using `Semantic`. This is useful for wrapping output pipelines with reusable intent constraints:

```python
from semantix import Semantic

# Declare a semantic type: a string that must be polite and professional
PoliteSummary = Semantic[str, "must be polite and professional"]

# Use validate_output with the semantic type
from semantix import validate_output

response = query_engine.query("What are the action items from the report?")
output = str(response)

result = validate_output(output=output, intent=PoliteSummary)
print(result.is_valid, result.reason)
```

### Composite Intents

You can combine multiple constraints to build richer validation logic:

```python
from semantix import Semantic, validate_output

# Validate against multiple intents at once
response = query_engine.query("List the risks mentioned in the document.")
output = str(response)

result = validate_output(
    output=output,
    intent=[
        "must identify at least one specific risk",
        "must not include recommendations or solutions",
        "must use neutral, factual language",
    ],
)

print(result.is_valid)   # True / False
print(result.reason)     # Explanation from the NLI model
```

### Why Use semantix with LlamaIndex?

- **No API cost** — inference runs on a local NLI model (~15ms per check)
- **Intent-based validation** — catches hallucinations, tone drift, and off-topic responses that schema checks miss
- **Zero configuration** — drop it in after any `query_engine.query()` call
- **166 tests, MIT licensed**

### Useful Links

- [semantix-ai on PyPI](https://pypi.org/project/semantix-ai/)
- [semantix-ai on GitHub](https://github.com/labrat-akhona/semantix-ai)
- [LlamaIndex Query Engine docs](https://docs.llamaindex.ai/en/stable/understanding/querying/querying/)
