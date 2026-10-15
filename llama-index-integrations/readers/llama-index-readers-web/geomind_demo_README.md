# GeoMind retrieval example

This example loads three GeoMind web pages, extracts article text,
splits it into chunks, and retrieves relevant chunks using local
multilingual E5 embeddings.

Retrieved chunks retain their source URLs. The example evaluates
whether the expected source appears in the first one or two results.

## Setup

Prerequisites: uv and Python 3.10 or newer. This example was developed
using Python 3.10.

From the repository root:

```bash
cd llama-index-integrations/readers/llama-index-readers-web
uv sync
uv pip install --python .venv/bin/python llama-index-embeddings-huggingface
```

The package environment supplies the HTML parsing dependencies.
The additional Hugging Face integration supplies the local embedding
model support.

## Run the example

From the same directory:

```bash
uv run --no-sync python geomind_demo.py
```

The first run downloads the embedding model. No API key is required.

Use `--no-sync` to preserve the additional dependency installed above,
which is not declared in this package's project configuration.

## Run the extraction tests

```bash
uv run --no-sync python -m pytest tests/test_geomind_demo.py -q
```

These two tests use small HTML strings and do not fetch web pages
or download model weights.

## Model and chunking

- Embedding model: intfloat/multilingual-e5-small
- Device: CPU
- Chunk size: 400 tokens, measured with the model tokenizer
- Chunk overlap: 40 tokens

## Evaluation

On the six example questions, my latest run produced:

- Hit@1: 5/6
- Hit@2: 6/6

These results are a small demonstration, not a general quality benchmark.
Website content changes can affect the results.

## Limitations

- Requires internet access to fetch the pages and initially download the model.
- Extracts content inside HTML article elements.
- Retrieves text; it does not generate answers.
- Does not yet evaluate the site's cross-reference links.
