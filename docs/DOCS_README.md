# Documentation Guide

## A guide for docs contributors

The `docs` directory contains the sphinx source text for LlamaIndex docs, visit
https://gpt-index.readthedocs.io/ to read the full documentation.

This guide is made for anyone who's interested in running LlamaIndex documentation locally,
making changes to it and make contributions. LlamaIndex is made by the thriving community
behind it, and you're always welcome to make contributions to the project and the 
documentation. 

## Build Docs

If you haven't already, clone the LlamaIndex Github repo to a local directory:

```bash
git clone https://github.com/jerryjliu/llama_index.git && cd llama_index
```

Install all dependencies required for building docs (mainly `sphinx` and its extension):

```bash
pip install -r docs/requirements.txt
```

Build the sphinx docs:

```bash
cd docs
make html
```

The docs HTML files are now generated under `docs/_build/html` directory, you can preview
it locally with the following command:

```bash
python -m http.server 8000 -d _build/html
```

And open your browser at http://0.0.0.0:8000/ to view the generated docs.


##### Watch Docs

We recommend using sphinx-autobuild during development, which provides a live-reloading 
server, that rebuilds the documentation and refreshes any open pages automatically when 
changes are saved. This enables a much shorter feedback loop which can help boost 
productivity when writing documentation.

Simply run the following command from LlamaIndex project's root directory: 
```bash
make watch-docs
```
