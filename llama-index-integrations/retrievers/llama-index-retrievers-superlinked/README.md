# LlamaIndex Superlinked Retriever

A LlamaIndex retriever integration for [Superlinked](https://github.com/superlinked/superlinked), mirroring the structure of official LlamaIndex retriever packages.

## Installation

Option A (standalone dev):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
pip install pytest
```

Option B (monorepo): Add this directory under `llama-index-integrations/retrievers` and install with the monorepo tooling.

Note: Examples require Python 3.10–3.12 (Superlinked does not support Python 3.9).

## Usage

```python
from llama_index.retrievers.superlinked import SuperlinkedRetriever
from llama_index.core import QueryBundle

retriever = SuperlinkedRetriever(
    sl_client=app,                 # Superlinked App
    sl_query=query_descriptor,     # Superlinked QueryDescriptor
    page_content_field="text",
    query_text_param="query_text",
    metadata_fields=None,
    k=4,
)

nodes = retriever.retrieve("What is a landmark in Paris?")
```

## Development

- Follows LlamaIndex contribution guidelines.
- Run tests: `pytest -q`.

## Testing without Superlinked

Tests use mocks for the `superlinked` imports so they can run without the dependency installed.

## Example

An end-to-end example is provided in `examples/steam_games_example.py`.
