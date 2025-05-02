# Contributing to LlamaIndex

Interested in contributing to LlamaIndex? Here's how to get started!

## Contribution Guideline

The best part of LlamaIndex is our community of users and contributors.

### What should I work on?

1. 🆕 Extend core modules
2. 🐛 Fix bugs
3. 🎉 Add usage examples
4. 🧪 Add experimental features
5. 📄 Improve code quality & documentation

Also, join our Discord for ideas and discussions: <https://discord.gg/dGcwcsnxhU>.

### 1. 🆕 Extend Core Modules

The most impactful way to contribute to LlamaIndex is by extending our core modules:
![LlamaIndex modules](https://github.com/jerryjliu/llama_index/raw/main/docs/_static/contribution/contrib.png)

We welcome contributions in _all_ modules shown above.
So far, we have implemented a core set of functionalities for each.
As a contributor, you can help each module unlock its full potential.

**NOTE**: We are making rapid improvements to the project, and as a result,
some interfaces are still volatile. Specifically, we are actively working on making the following components more modular and extensible (uncolored boxes above): core indexes, document stores, index queries, query runner

#### Module Details

Below, we will describe what each module does, give a high-level idea of the interface, show existing implementations, and give some ideas for contribution.

---

#### Data Loaders

A data loader ingests data of any format from anywhere into `Document` objects, which can then be parsed and indexed.

**Interface**:

- `load_data` takes arbitrary arguments as input (e.g. path to data), and outputs a sequence of `Document` objects.
- `lazy_load_data` takes arbitrary arguments as input (e.g. path to data), and outputs an iterable object of `Document` objects. This is a lazy version of `load_data`, which is useful for large datasets.

> **Note**: If only `lazy_load_data` is implemented, `load_data` will be delegated to it.

**Examples**:

- [Google Sheets Loader](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/google_sheets)
- [Gmail Loader](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/gmail)
- [Github Repository Loader](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/github_repo)

Contributing a data loader is easy and super impactful for the community.
The preferred way to contribute is by making a PR at [LlamaHub Github](https://github.com/emptycrown/llama-hub).

**Ideas**

- Want to load something but there's no LlamaHub data loader for it yet? Make a PR!

---

#### Node Parser

A node parser parses `Document` objects into `Node` objects (atomic units of data that LlamaIndex operates over, e.g., chunk of text, image, or table).
It is responsible for splitting text (via text splitters) and explicitly modeling the relationship between units of data (e.g. A is the source of B, C is a chunk after D).

**Interface**: `get_nodes_from_documents` takes a sequence of `Document` objects as input, and outputs a sequence of `Node` objects.

**Examples**:

- [Simple Node Parser](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/file/simple_file.py)

See [the API reference](https://docs.llamaindex.ai/en/latest/api_reference/index.html) for full details.

**Ideas**:

- Add new `Node` relationships to model hierarchical documents (e.g. play-act-scene, chapter-section-heading).

---

#### Text Splitters

Text splitter splits a long text `str` into smaller text `str` chunks with desired size and splitting "strategy" since LLMs have a limited context window size, and the quality of text chunk used as context impacts the quality of query results.

**Interface**: `split_text` takes a `str` as input, and outputs a sequence of `str`

**Examples**:

- [Token Text Splitter](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/text/token.py#L22)
- [Sentence Splitter](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/text/sentence.py#L34)
- [Code Splitter](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/text/code.py#L17)

---

#### Document/Index/KV Stores

Under the hood, LlamaIndex also supports a swappable **storage layer** that allows you to customize Document Stores (where ingested documents (i.e., `Node` objects) are stored), and Index Stores (where index metadata are stored)

We have an underlying key-value abstraction backing the document/index stores.
Currently we support in-memory and MongoDB storage for these stores. Open to contributions!

See [Storage guide](https://docs.llamaindex.ai/en/stable/module_guides/storing/kv_stores.html) for details.

---

#### Managed Index

A managed index is used to represent an index that's managed via an API, exposing API calls to index documents and query documents.

Currently we support the [VectaraIndex](https://github.com/run-llama/llama_index/tree/ca09272af000307762d301c99da46ddc70d3bfd2/llama_index/indices/managed/vectara).
Open to contributions!

See [Managed Index docs](https://docs.llamaindex.ai/en/stable/community/integrations/managed_indices.html) for details.

---

#### Vector Stores

Our vector store classes store embeddings and support lookup via similarity search.
These serve as the main data store and retrieval engine for our vector index.

**Interface**:

- `add` takes in a sequence of `NodeWithEmbeddings` and inserts the embeddings (and possibly the node contents & metadata) into the vector store.
- `delete` removes entries given document IDs.
- `query` retrieves top-k most similar entries given a query embedding.

**Examples**:

- [Pinecone](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/vector_stores/llama-index-vector-stores-pinecone/llama_index/vector_stores/pinecone/base.py)
- [Faiss](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/vector_stores/llama-index-vector-stores-faiss/llama_index/vector_stores/faiss/base.py)
- [Chroma](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/vector_stores/llama-index-vector-stores-chroma/llama_index/vector_stores/chroma/base.py)
- [DashVector](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/vector_stores/llama-index-vector-stores-dashvector/llama_index/vector_stores/dashvector/base.py)

**Ideas**:

- See a vector database out there that we don't support yet? Make a PR!

See [reference](https://docs.llamaindex.ai/en/stable/api_reference/indices/vector_store.html) for full details.

---

#### Retrievers

Our retriever classes are lightweight classes that implement a `retrieve` method.
They may take in an index class as input - by default, each of our indices
(list, vector, keyword) has an associated retriever. The output is a set of
`NodeWithScore` objects (a `Node` object with an extra `score` field).

You may also choose to implement your own retriever classes on top of your own
data if you wish.

**Interface**:

- `retrieve` takes in a `str` or `QueryBundle` as input, and outputs a list of `NodeWithScore` objects

**Examples**:

- [Vector Index Retriever](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/vector_store/retrievers/retriever.py#L24)
- [List Index Retriever](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/list/retrievers.py)
- [Transform Retriever](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/retrievers/transform_retriever.py)

**Ideas**:

- Besides the "default" retrievers built on top of each index, what about fancier retrievers? E.g. retrievers that take in other retrievers as input? Or other
  types of data?

---

#### Query Engines

Our query engine classes are lightweight classes that implement a `query` method; the query returns a response type.
For instance, they may take in a retriever class as input; our `RetrieverQueryEngine`
takes in a `retriever` as input as well as a `BaseSynthesizer` class for response synthesis, and
the `query` method performs retrieval and synthesis before returning the final result.
They may take in other query engine classes as input too.

**Interface**:

- `query` takes in a `str` or `QueryBundle` as input, and outputs a `Response` object.

**Examples**:

- [Retriever Query Engine](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/query_engine/retriever_query_engine.py#L25)
- [Transform Query Engine](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/query_engine/transform_query_engine.py#L11)

---

#### Query Transforms

A query transform augments a raw query string with associated transformations to improve index querying.
This can interpreted as a pre-processing stage, before the core index query logic is executed.

**Interface**: `run` takes in a `str` or `Querybundle` as input, and outputs a transformed `QueryBundle`.

**Examples**:

- [Hypothetical Document Embeddings](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/query/query_transform/base.py#L108)
- [Query Decompose](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/query/query_transform/base.py#L164)

See [guide](https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/query_transformations.html#hyde-hypothetical-document-embeddings) for more information.

---

#### Token Usage Optimizers

A token usage optimizer refines the retrieved `Nodes` to reduce token usage during response synthesis.

**Interface**: `optimize` takes in the `QueryBundle` and a text chunk `str`, and outputs a refined text chunk `str` that yields a more optimized response

**Examples**:

- [Sentence Embedding Optimizer](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/postprocessor/optimizer.py#L16)

---

#### Node Postprocessors

A node postprocessor refines a list of retrieved nodes given configuration and context.

**Interface**: `postprocess_nodes` takes a list of `Nodes` and extra metadata (e.g. similarity and query), and outputs a refined list of `Nodes`.

**Examples**:

- [Keyword Postprocessor](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/postprocessor/node.py#L26): filters nodes based on keyword match
- [Similarity Postprocessor](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/postprocessor/node.py#L70): filers nodes based on similarity threshold
- [Prev Next Postprocessor](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/postprocessor/node.py#L149): fetches additional nodes to augment context based on node relationships.

---

#### Output Parsers

An output parser enables us to extract structured output from the plain text output generated by the LLM.

**Interface**:

- `format`: formats a query `str` with structured output formatting instructions, and outputs the formatted `str`
- `parse`: takes a `str` (from LLM response) as input, and gives a parsed structured output (optionally also validated, error-corrected).

**Examples**:

- [Guardrails Output Parser](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/output_parsers/llama-index-output-parsers-guardrails/llama_index/output_parsers/guardrails/base.py#L17)
- [Langchain Output Parser](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/output_parsers/llama-index-output-parsers-langchain/llama_index/output_parsers/langchain/base.py#L12)

See [guide](https://docs.llamaindex.ai/en/stable/module_guides/querying/structured_outputs/output_parser.html) for more information.

---

### 2. 🐛 Fix Bugs

Most bugs are reported and tracked in the [Github Issues Page](https://github.com/jerryjliu/llama_index/issues).
We try our best in triaging and tagging these issues:

- Issues tagged as `bug` are confirmed bugs.
- New contributors may want to start with issues tagged with `good first issue`.

Please feel free to open an issue and/or assign an issue to yourself.

### 3. 🎉 Add Usage Examples

If you have applied LlamaIndex to a unique use-case (e.g. interesting dataset, customized index structure, complex query), we would love your contribution in the form of:

1. a guide: e.g. [guide to LlamIndex + Structured Data](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/structured_data.html)
2. an example notebook: e.g. [Email Info Extraction](https://docs.llamaindex.ai/en/stable/examples/usecases/email_data_extraction.html)

### 4. 🧪 Add Experimental Features

If you have a crazy idea, make a PR for it!
Whether if it's the latest research, or what you thought of in the shower, we'd love to see creative ways to improve LlamaIndex.

### 5. 📄 Improve Code Quality & Documentation

We would love your help in making the project cleaner, more robust, and more understandable. If you find something confusing, it most likely is for other people as well. Help us be better!

## Development Guideline

### Environment Setup

LlamaIndex is a Python package. We've tested primarily with Python versions >= 3.8. Here's a quick
and dirty guide to getting your environment setup.

First, create a fork of LlamaIndex, by clicking the "Fork" button on the [LlamaIndex Github page](https://github.com/jerryjliu/llama_index).
Following [these steps](https://docs.github.com/en/get-started/quickstart/fork-a-repo) for more details
on how to fork the repo and clone the forked repo.

Then, create a new Python virtual environment using poetry.

- [Install poetry](https://python-poetry.org/docs/#installation) - this will help you manage package dependencies
- `poetry shell` - this command creates a virtual environment, which keeps installed packages contained to this project
- `poetry install --with dev,docs` - this will install all dependencies needed for most local development

Now you should be set!

### Validating your Change

Let's make sure to `format/lint` our change. For bigger changes,
let's also make sure to `test` it and perhaps create an `example notebook`.

#### Formatting/Linting

You can format and lint your changes with the following commands in the root directory:

```bash
make format; make lint
```

You can also make use of our pre-commit hooks by setting up git hook scripts:

```bash
pre-commit install
```

We run an assortment of linters: `black`, `ruff`, `mypy`.

#### Testing

For bigger changes, you'll want to create a unit test. Our tests are in the `tests` folder.
We use `pytest` for unit testing. To run all unit tests, run the following in the root dir:

```bash
pytest tests
```

or

```bash
make test
```

### Creating an Example Notebook

For changes that involve entirely new features, it may be worth adding an example Jupyter notebook to showcase
this feature.

Example notebooks can be found in this folder: <https://github.com/run-llama/llama_index/tree/main/docs/docs/examples>.

### Creating a pull request

See [these instructions](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
to open a pull request against the main LlamaIndex repo.
