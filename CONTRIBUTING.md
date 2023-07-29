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

Also, join our Discord for ideas and discussions: https://discord.gg/dGcwcsnxhU.


### 1. 🆕 Extend Core Modules
The most impactful way to contribute to LlamaIndex is extending our core modules:
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

**Interface**: `load_data` takes arbitrary arguments as input (e.g. path to data), and outputs a sequence of `Document` objects.


**Examples**:
* [Google Sheets Loader](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/google_sheets)
* [Gmail Loader](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/gmail)
* [Github Repository Loader](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/github_repo)

Contributing a data loader is easy and super impactful for the community.
The preferred way to contribute is making a PR at [LlamaHub Github](https://github.com/emptycrown/llama-hub).

**Ideas**
* Want to load something but there's no LlamaHub data loader for it yet? Make a PR!

---
#### Node Parser
A node parser parses `Document` objects into `Node` objects (atomic unit of data that LlamaIndex operates over, e.g., chunk of text, image, or table).
It is responsible for splitting text (via text splitters) and explicitly modeling the relationship between units of data (e.g. A is the source of B, C is a chunk after D).

**Interface**: `get_nodes_from_documents` takes a sequence of `Document` objects as input, and outputs a sequence of `Node` objects.

**Examples**:
* [Simple Node Parser](https://github.com/jerryjliu/llama_index/blob/main/llama_index/node_parser/simple.py)

See [the API reference](https://gpt-index.readthedocs.io/en/latest/api_reference/index.html) for full details.

**Ideas**:
* Add new `Node` relationships to model to model hierarchical documents (e.g. play-act-scene, chapter-section-heading).

---
#### Text Splitters
Text splitter splits a long text `str` into smaller text `str` chunks with desired size and splitting "strategy" since LLMs have a limited context window size, and the quality of text chunk used as context impacts the quality of query results.

**Interface**: `split_text` takes a `str` as input, and outputs a sequence of `str`

**Examples**:
* [Token Text Splitter](https://github.com/jerryjliu/llama_index/blob/main/llama_index/langchain_helpers/text_splitter.py#L23)
* [Sentence Splitter](https://github.com/jerryjliu/llama_index/blob/main/llama_index/langchain_helpers/text_splitter.py#L239)

---

#### Document/Index/KV Stores

Under the hood, LlamaIndex also supports a swappable **storage layer** that allows you to customize Document Stores (where ingested documents (i.e., `Node` objects) are stored), and Index Stores (where index metadata are stored)

We have an underlying key-value abstraction backing the document/index stores.
Currently we support in-memory and MongoDB storage for these stores. Open to contributions!

See [Storage guide](https://gpt-index.readthedocs.io/en/latest/how_to/storage.html) for details.

----

#### Vector Stores
Our vector store classes store embeddings and support lookup via similiarity search.
These serve as the main data store and retrieval engine for our vector index.

**Interface**:
* `add` takes in a sequence of `NodeWithEmbeddings` and insert the embeddings (and possibly the node contents & metadata) into the vector store.
* `delete` removes entries given document IDs.
* `query` retrieves top-k most similar entries given a query embedding.

**Examples**:
* [Pinecone](https://github.com/jerryjliu/llama_index/blob/main/llama_index/vector_stores/pinecone.py)
* [Faiss](https://github.com/jerryjliu/llama_index/blob/main/llama_index/vector_stores/faiss.py)
* [Chroma](https://github.com/jerryjliu/llama_index/blob/main/llama_index/vector_stores/chroma.py)

**Ideas**:
* See a vector database out there that we don't support yet? Make a PR!

See [reference](https://gpt-index.readthedocs.io/en/latest/reference/indices/vector_stores/stores.html) for full details.

---
#### Retrievers

Our retriever classes are lightweight classes that implement a `retrieve` method.
They may take in an index class as input - by default, each of our indices
(list, vector, keyword) have an associated retriever. The output is a set of 
`NodeWithScore` objects (a `Node` object with an extra `score` field).

You may also choose to implement your own retriever classes on top of your own
data if you wish.

**Interface**:
- `retrieve` takes in a `str` or `QueryBundle` as input, and outputs a list of `NodeWithScore` objects

**Examples**:
* [Vector Index Retriever](https://github.com/jerryjliu/llama_index/blob/main/llama_index/indices/vector_store/retrievers.py)
* [List Index Retriever](https://github.com/jerryjliu/llama_index/blob/main/llama_index/indices/list/retrievers.py)
* [Transform Retriever](https://github.com/jerryjliu/llama_index/blob/main/llama_index/retrievers/transform_retriever.py)

**Ideas**:
* Besides the "default" retrievers built on top of each index, what about fancier retrievers? E.g. retrievers that take in other retrivers as input? Or other
types of data?

---

#### Query Engines

Our query engine classes are lightweight classes that implement a `query` method; the query returns a response type.
For instance, they may take in a retriever class as input; our `RetrieverQueryEngine` 
takes in a `retriever` as input as well as a `BaseSynthesizer` class for response synthesis, and
the `query` method performs retrieval and synthesis before returning the final result.
They may take in other query engine classes in as input too.

**Interface**:
- `query` takes in a `str` or `QueryBundle` as input, and outputs a `Response` object.

**Examples**:
- [Retriever Query Engine](https://github.com/jerryjliu/llama_index/blob/main/llama_index/query_engine/retriever_query_engine.py)
- [Transform Query Engine](https://github.com/jerryjliu/llama_index/blob/main/llama_index/query_engine/transform_query_engine.py)

---

#### Query Transforms
A query transform augments a raw query string with associated transformations to improve index querying.
This can interpreted as a pre-processing stage, before the core index query logic is executed.

**Interface**: `run` takes in a `str` or `Querybundle` as input, and outputs a transformed `QueryBundle`.

**Examples**:
* [Hypothetical Document Embeddings](https://github.com/jerryjliu/llama_index/blob/main/llama_index/indices/query/query_transform/base.py#L77)
* [Query Decompose](https://github.com/jerryjliu/llama_index/blob/main/llama_index/indices/query/query_transform/base.py#L124)

See [guide](https://gpt-index.readthedocs.io/en/latest/how_to/query/query_transformations.html#hyde-hypothetical-document-embeddings) for more information.

---
#### Token Usage Optimizers
A token usage optimizer refines the retrieved `Nodes` to reduce token usage during response synthesis.

**Interface**: `optimize` takes in the `QueryBundle` and a text chunk `str`, and outputs a refined text chunk `str` that yeilds a more optimized response

**Examples**:
* [Sentence Embedding Optimizer](https://github.com/jerryjliu/llama_index/blob/main/llama_index/optimization/optimizer.py)

---
#### Node Postprocessors
A node postprocessor refines a list of retrieve nodes given configuration and context.

**Interface**: `postprocess_nodes` takes a list of `Nodes` and extra metadata (e.g. similarity and query), and outputs a refined list of `Nodes`.


**Examples**:
* [Keyword Postprocessor](https://github.com/jerryjliu/llama_index/blob/main/llama_index/indices/postprocessor/node.py#L32): filters nodes based on keyword match
* [Similarity Postprocessor](https://github.com/jerryjliu/llama_index/blob/main/llama_index/indices/postprocessor/node.py#L62): filers nodes based on similarity threshold
* [Prev Next Postprocessor](https://github.com/jerryjliu/llama_index/blob/main/llama_index/indices/postprocessor/node.py#L135): fetchs additional nodes to augment context based on node relationships.

---
#### Output Parsers
A output parser enables us to extract structured output from the plain text output generated by the LLM.

**Interface**:
* `format`: formats a query `str` with structured output formatting instructions, and outputs the formatted `str` 
* `parse`: takes a `str` (from LLM response) as input, and gives a parsed tructured output (optionally also validated, error-corrected).

**Examples**:
* [Guardrails Output Parser](https://github.com/jerryjliu/llama_index/blob/main/llama_index/output_parsers/guardrails.py)
* [Langchain Output Parser](https://github.com/jerryjliu/llama_index/blob/main/llama_index/output_parsers/langchain.py)

See [guide](https://gpt-index.readthedocs.io/en/latest/how_to/output_parsing.html) for more information.

---

### 2. 🐛 Fix Bugs
Most bugs are reported and tracked in the [Github Issues Page](https://github.com/jerryjliu/llama_index/issues).
We try our best in triaging and tagging these issues:
* Issues tagged as `bug` are confirmed bugs. 
* New contributors may want to start with issues tagged with `good first issue`. 

Please feel free to open an issue and/or assign an issue to yourself.

### 3. 🎉 Add Usage Examples
If you have applied LlamaIndex to a unique use-case (e.g. interesting dataset, customized index structure, complex query), we would love your contribution in the form of:
1. a guide: e.g. [guide to LlamIndex + Structured Data](https://gpt-index.readthedocs.io/en/latest/guides/tutorials/sql_guide.html)
Todo.
2. an example notebook: e.g. [Composable Indices Demo](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/composable_indices/ComposableIndices-Prior.ipynb)

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

Then, create a new Python virtual environment. The command below creates an environment in `.venv`,
and activates it:
```bash
python -m venv .venv
source .venv/bin/activate
```

if you are in windows, use the following to activate your virtual environment:

```bash
.venv\scripts\activate
```

Install the required dependencies (this will also install LlamaIndex through `pip install -e .` 
so that you can start developing on it):

```bash
pip install -r requirements.txt
```

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

### Creating an Example Notebook

For changes that involve entirely new features, it may be worth adding an example Jupyter notebook to showcase
this feature. 

Example notebooks can be found in this folder: https://github.com/jerryjliu/llama_index/tree/main/examples.


### Creating a pull request

See [these instructions](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
to open a pull request against the main LlamaIndex repo.













