# Contributing to LlamaIndex

Interested in contributing to LlamaIndex? Here's how to get started!

## Contribution Guideline

The best part of LlamaIndex is our community of users and contributors.

### What should I work on?

1. 🆕 Extend core modules by contributing an integration
2. 📦 Contribute a Tool, Reader, Pack, or Dataset (formerly from llama-hub)
3. 🧠 Add new capabilities to core
4. 🐛 Fix bugs
5. 🎉 Add usage examples
6. 🧪 Add experimental features
7. 📄 Improve code quality & documentation

Also, join our Discord for ideas and discussions: <https://discord.gg/dGcwcsnxhU>.

### 1. 🆕 Extend Core Modules

The most impactful way to contribute to LlamaIndex is by extending our core modules:
![LlamaIndex modules](https://github.com/jerryjliu/llama_index/raw/main/docs/_static/contribution/contrib.png)

We welcome contributions in _all_ modules shown above.
So far, we have implemented a core set of functionalities for each, all of
which are encapsulated in the LlamaIndex core package. As a contributor,
you can help each module unlock its full potential. Provided below are
brief description of these modules. You can also refer to their respective
folders within this Github repository for some example integrations.

Contributing an integration involves submitting the source code for a new Python
package. For now, these integrations will live in the LlamaIndex Github repository
and the team will be responsible for publishing the package to PyPi. (Having
these packages live outside of this repository and maintained by our community
members is in consideration.)

#### Creating A New Integration Package

Both `llama-index` and `llama-index-core` come equipped
with a command-line tool that can be used to initialize a new integration package.

```shell
cd ./llama-index-integrations/llms
llamaindex-cli new-package --kind "llms" --name "gemini"
```

Executing the above commands will create a new folder called `llama-index-llms-gemini`
within the `llama-index-integrations/llms` directory.

Please ensure to add a detailed README for your new package as it will appear in
both [llamahub.ai](https://llamahub.ai) as well as the PyPi.org website.
In addition to preparing your source code and supplying a detailed README, we
also ask that you fill in some
metadata for your package to appear in [llamahub.ai](https://llamahub.ai) with the
correct information. You do so by adding the required metadata under the `[tool.llamahub]`
section with your new package's `pyproject.toml`.

Below is the example of the metadata required for all of our integration packages. Please
replace the default author "llama-index" with your own Github user name.

```toml
[tool.llamahub]
contains_example = false
import_path = "llama_index.llms.anthropic"

[tool.llamahub.class_authors]
Anthropic = "llama-index"
```

([source](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/llms/llama-index-llms-anthropic/pyproject.toml))

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

- [Simple Node Parser](https://github.com/jerryjliu/llama_index/blob/main/llama_index/node_parser/simple.py)

See [the API reference](https://docs.llamaindex.ai/en/latest/api_reference/index.html) for full details.

**Ideas**:

- Add new `Node` relationships to model hierarchical documents (e.g. play-act-scene, chapter-section-heading).

---

#### Text Splitters

Text splitter splits a long text `str` into smaller text `str` chunks with desired size and splitting "strategy" since LLMs have a limited context window size, and the quality of text chunk used as context impacts the quality of query results.

**Interface**: `split_text` takes a `str` as input, and outputs a sequence of `str`

**Examples**:

- [Token Text Splitter](https://github.com/jerryjliu/llama_index/blob/main/llama_index/langchain_helpers/text_splitter.py#L26)
- [Sentence Splitter](https://github.com/jerryjliu/llama_index/blob/main/llama_index/langchain_helpers/text_splitter.py#L276)
- [Code Splitter](https://github.com/jerryjliu/llama_index/blob/main/llama_index/langchain_helpers/text_splitter.py#L476)

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

- [Pinecone](https://github.com/jerryjliu/llama_index/blob/main/llama_index/vector_stores/pinecone.py)
- [Faiss](https://github.com/jerryjliu/llama_index/blob/main/llama_index/vector_stores/faiss.py)
- [Chroma](https://github.com/jerryjliu/llama_index/blob/main/llama_index/vector_stores/chroma.py)
- [DashVector](https://github.com/jerryjliu/llama_index/blob/main/llama_index/vector_stores/dashvector.py)

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

- [Vector Index Retriever](https://github.com/jerryjliu/llama_index/blob/main/llama_index/indices/vector_store/retrievers.py)
- [List Index Retriever](https://github.com/jerryjliu/llama_index/blob/main/llama_index/indices/list/retrievers.py)
- [Transform Retriever](https://github.com/jerryjliu/llama_index/blob/main/llama_index/retrievers/transform_retriever.py)

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

- [Retriever Query Engine](https://github.com/jerryjliu/llama_index/blob/main/llama_index/query_engine/retriever_query_engine.py)
- [Transform Query Engine](https://github.com/jerryjliu/llama_index/blob/main/llama_index/query_engine/transform_query_engine.py)

---

#### Query Transforms

A query transform augments a raw query string with associated transformations to improve index querying.
This can interpreted as a pre-processing stage, before the core index query logic is executed.

**Interface**: `run` takes in a `str` or `Querybundle` as input, and outputs a transformed `QueryBundle`.

**Examples**:

- [Hypothetical Document Embeddings](https://github.com/jerryjliu/llama_index/blob/main/llama_index/indices/query/query_transform/base.py#L77)
- [Query Decompose](https://github.com/jerryjliu/llama_index/blob/main/llama_index/indices/query/query_transform/base.py#L124)

See [guide](https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/query_transformations.html#hyde-hypothetical-document-embeddings) for more information.

---

#### Token Usage Optimizers

A token usage optimizer refines the retrieved `Nodes` to reduce token usage during response synthesis.

**Interface**: `optimize` takes in the `QueryBundle` and a text chunk `str`, and outputs a refined text chunk `str` that yields a more optimized response

**Examples**:

- [Sentence Embedding Optimizer](https://github.com/jerryjliu/llama_index/blob/main/llama_index/optimization/optimizer.py)

---

#### Node Postprocessors

A node postprocessor refines a list of retrieved nodes given configuration and context.

**Interface**: `postprocess_nodes` takes a list of `Nodes` and extra metadata (e.g. similarity and query), and outputs a refined list of `Nodes`.

**Examples**:

- [Keyword Postprocessor](https://github.com/run-llama/llama_index/blob/main/llama_index/postprocessor/node.py#L32): filters nodes based on keyword match
- [Similarity Postprocessor](https://github.com/run-llama/llama_index/blob/main/llama_index/postprocessor/node.py#L74): filers nodes based on similarity threshold
- [Prev Next Postprocessor](https://github.com/run-llama/llama_index/blob/main/llama_index/postprocessor/node.py#L175): fetches additional nodes to augment context based on node relationships.

---

#### Output Parsers

An output parser enables us to extract structured output from the plain text output generated by the LLM.

**Interface**:

- `format`: formats a query `str` with structured output formatting instructions, and outputs the formatted `str`
- `parse`: takes a `str` (from LLM response) as input, and gives a parsed structured output (optionally also validated, error-corrected).

**Examples**:

- [Guardrails Output Parser](https://github.com/jerryjliu/llama_index/blob/main/llama_index/output_parsers/guardrails.py)
- [Langchain Output Parser](https://github.com/jerryjliu/llama_index/blob/main/llama_index/output_parsers/langchain.py)

See [guide](https://docs.llamaindex.ai/en/stable/module_guides/querying/structured_outputs/output_parser.html) for more information.

---

### 2. 📦 Contribute a Pack, Reader, Tool, or Dataset (formerly from llama-hub)

Tools, Readers, and Packs have all been migrated from [llama-hub](https://github.com/run-llama/llama-hub) to the main
llama-index repo (i.e., this one). Datasets still reside in llama-hub, but will be
migrated here in the near future.

Contributing a new Reader or Tool involves submitting a new package within
the [llama-index-integrations/readers](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/readers) and [llama-index-integrations/tools](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/tools),
folders respectively.

The LlamaIndex command-line tool can be used to initialize new Packs and Integrations. (NOTE: `llama-index-cli` comes installed with `llama-index`.)

```shell
cd ./llama-index-packs
llamaindex-cli new-package --kind "packs" --name "my new pack"

cd ./llama-index-integrations/readers
llamaindex-cli new-package --kind "readers" --name "new reader"
```

Executing the first set of shell commands will create a new folder called `llama-index-packs-my-new-pack`
within the `llama-index-packs` directory. While the second set will create a new
package directory called `llama-index-readers-new-reader` within the `llama-index-integrations/readers` directory.

Please ensure to add a detailed README for your new package as it will appear in
both [llamahub.ai](https://llamahub.ai) as well as the PyPi.org website.
In addition to preparing your source code and supplying a detailed README, we
also ask that you fill in some
metadata for your package to appear in [llamahub.ai](https://llamahub.ai) with the
correct information. You do so by adding the required metadata under the `[tool.llamahub]`
section with your new package's `pyproject.toml`.

Below is the example of the metadata required for packs, readers and tools:

```toml
[tool.llamahub]
contains_example = true
import_path = "llama_index.packs.agent_search_retriever"

[tool.llamahub.class_authors]
AgentSearchRetrieverPack = "logan-markewich"
```

([source](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-agent-search-retriever/pyproject.toml))

### 3. 🧠 Add new capabilities to core

We would greatly appreciate any and all contributions to our core abstractions
that represent enhancements from the current set of capabilities.
General improvements that make these core abstractions more robust and thus
easier to build on are also welcome!

A [Requests For Contribution Project Board](https://github.com/orgs/run-llama/projects/2)
has been curated and can provide some ideas for what contributions can be made
into core.

### 4. 🐛 Fix Bugs

Most bugs are reported and tracked in the [Github Issues Page](https://github.com/jerryjliu/llama_index/issues).
We try our best in triaging and tagging these issues:

- Issues tagged as `bug` are confirmed bugs.
- New contributors may want to start with issues tagged with `good first issue`.

Please feel free to open an issue and/or assign an issue to yourself.

### 5. 🎉 Add Usage Examples

If you have applied LlamaIndex to a unique use-case (e.g. interesting dataset, customized index structure, complex query), we would love your contribution in the form of:

1. a guide: e.g. [Guide to LlamIndex + Structured Data](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/structured_data.html)
2. an example notebook: e.g. [Email Info Extraction](/examples/usecases/email_data_extraction.ipynb)

### 6. 🧪 Add Experimental Features

If you have a crazy idea, make a PR for it!
Whether if it's the latest research, or what you thought of in the shower, we'd love to see creative ways to improve LlamaIndex.

### 7. 📄 Improve Code Quality & Documentation

We would love your help in making the project cleaner, more robust, and more understandable. If you find something confusing, it most likely is for other people as well. Help us be better!

## Development Guidelines

### Setting up environment

LlamaIndex is a Python package. We've tested primarily with Python versions >= 3.8. Here's a quick
and dirty guide to setting up your environment for local development.

1. Fork [LlamaIndex Github repo][ghr]\* and clone it locally. (New to GitHub / git? Here's [how][frk].)
2. In a terminal, `cd` into the directory of your local clone of your forked repo.
3. Install [pre-commit hooks][pch]\* by running `pre-commit install`. These hooks are small house-keeping scripts executed every time you make a git commit, which automates away a lot of chores.
4. `cd` into the specific package you want to work on. For example, if I want to work on the core package, I execute `cd llama-index-core/`. (New to terminal / command line? Here's a [getting started guide][gsg].)
5. Prepare a [virtual environment][vev].
   1. [Install Poetry][pet]\*. This will help you manage package dependencies.
   2. Execute `poetry shell`. This command will create a [virtual environment][vev] specific for this package, which keeps installed packages contained to this project. (New to Poetry, the dependency & packaging manager for Python? Read about its basic usage [here][bus].)
   3. Execute `poetry install --with dev,docs`\*. This will install all dependencies needed for local development. To see what will be installed, read the `pyproject.toml` under that directory.

[frk]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo
[ghr]: https://github.com/run-llama/llama_index/
[pch]: https://pre-commit.com/
[gsg]: https://www.freecodecamp.org/news/command-line-for-beginners/
[pet]: https://python-poetry.org/docs/#installation
[vev]: https://python-poetry.org/docs/managing-environments/
[bus]: https://python-poetry.org/docs/basic-usage/

Steps marked with an asterisk (`*`) are one-time tasks. You don't have to repeat them when you attempt to contribute on something else next time.

Now you should be set!

### Validating your Change

Let's make sure to `format/lint` our change. For bigger changes,
let's also make sure to `test` it and perhaps create an `example notebook`.

#### Formatting/Linting

We run an assortment of linters: `black`, `ruff`, `mypy`.

If you have installed pre-commit hooks in this repo, they should have taken care of the formatting and linting automatically.

If -- for whatever reason -- you would like to do it manually, you can format and lint your changes with the following commands in the root directory:

```bash
make format; make lint
```

Under the hood, we still install pre-commit hooks for you, so that you don't have to do this manually next time.

#### Testing

If you modified or added code logic, **create test(s)**, because they help preventing other maintainers from accidentally breaking the nice things you added / re-introducing the bugs you fixed.

- In almost all cases, add **unit tests**.
- If your change involves adding a new integration, also add **integration tests**. When doing so, please [mock away][mck] the remote system that you're integrating LlamaIndex with, so that when the remote system changes, LlamaIndex developers won't see test failures.

Reciprocally, you should **run existing tests** (from every package that you touched) before making a git commit, so that you can be sure you didn't break someone else's good work.

(By the way, when a test is run with the goal of detecting whether something broke in a new version of the codebase, it's referred to as a "[regression test][reg]". You'll also hear people say "the test _regressed_" as a more diplomatic way of saying "the test _failed_".)

Our tests are stored in the `tests` folders under each package directory. We use the testing framework [pytest][pyt], so you can **just run `pytest` in each package you touched** to run all its tests.

Just like with formatting and linting, if you prefer to do things the [make][mkf] way, run:

```shell
make test
```

Regardless of whether you have run them locally, a [CI system][cis] will run all affected tests on your PR when you submit one anyway. There, tests are orchestrated with [Pants][pts], the build system of our choice. There is a slight chance that tests broke on CI didn't break on your local machine or the other way around. When that happens, please take our CI as the source of truth. This is because our release pipeline (which builds the packages users are going to download from PyPI) are run in the CI, not on your machine (even if you volunteer), so it's the CI that is the golden standard.

[reg]: https://www.browserstack.com/guide/regression-testing
[mck]: https://pytest-mock.readthedocs.io/en/latest/
[pyt]: https://docs.pytest.org/
[mkf]: https://makefiletutorial.com/
[cis]: https://www.atlassian.com/continuous-delivery/continuous-integration
[pts]: https://www.pantsbuild.org/

### Creating an Example Notebook

For changes that involve entirely new features, it may be worth adding an example Jupyter notebook to showcase
this feature.

Example notebooks can be found in [this folder](https://github.com/run-llama/llama_index/tree/main/docs/docs/examples).

### Creating a pull request

See [these instructions](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
to open a pull request against the main LlamaIndex repo.
