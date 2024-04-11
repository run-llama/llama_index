# Documentation Guide

## A guide for docs contributors

The `docs` directory contains the sphinx source text for LlamaIndex docs, visit
https://docs.llamaindex.ai/en/stable/ to read the full documentation.

This guide is made for anyone who's interested in running LlamaIndex documentation locally,
making changes to it and making contributions. LlamaIndex is made by the thriving community
behind it, and you're always welcome to make contributions to the project and the
documentation.

## Build Docs

If you haven't already, clone the LlamaIndex Github repo to a local directory:

```bash
git clone https://github.com/run-llama/llama_index.git && cd llama_index
```

Install all dependencies required for building docs (mainly `mkdocs` and its extension):

- [Install poetry](https://python-poetry.org/docs/#installation) - this will help you manage package dependencies
- `poetry shell` - this command creates a virtual environment, which keeps installed packages contained to this project
- `poetry install --with docs` - this will install all dependencies needed for building docs

Build with mkdocs:

```bash
cd docs
mkdocs serve
```

And open your browser at http://localhost:8000/ to view the generated docs.

This hosted version will re-build and update as changes are made to the docs.

## Config

All config for mkdocs is in the `mkdocs.yml` file.

Running the command `python docs/prepare_for_build.py` from the root of the llama-index repo will update the mkdocs.yml API Reference and examples nav with the latest changes, as well as writing new api reference files.
