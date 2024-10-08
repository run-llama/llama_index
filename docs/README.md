# Documentation

This directory contains the documentation source code for LlamaIndex, available at https://docs.llamaindex.ai.

This guide is made for anyone who's interested in running LlamaIndex documentation locally,
making changes to it and making contributions. LlamaIndex is made by the thriving community
behind it, and you're always welcome to make contributions to the project and the
documentation.

## Build Docs

If you haven't already, clone the LlamaIndex Github repo to a local directory:

```
git clone https://github.com/run-llama/llama_index.git && cd llama_index
```

Documentation has its own, dedicated Python virtual environment, and all the tools and scripts are available from the
`docs` directory:

```
cd llama_index/docs
```

From now on, we assume all the commands will be executed from the `docs` directory.

Install all dependencies required for building docs (mainly `mkdocs` and its extension):

- [Install poetry](https://python-poetry.org/docs/#installation) - this will help you manage package dependencies
- `poetry install` - this will install all dependencies needed for building docs

To build the docs and browse them locally run:

```
poetry run mkdocs serve --dirty
```

> [!NOTE]
> With `--dirty` mkdocs will rebuild only files that have changed, decreasing the time it takes to iterate on a page.

You can now open your browser at http://localhost:8000/ to view the generated docs. The local server will rebuild the
docs and refresh your browser every time you make changes to the docs.
