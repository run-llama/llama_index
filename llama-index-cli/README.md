# LlamaIndex CLI

## Installation

```sh
pip install llama-index-cli
```

## Usage

```sh
llamaindex-cli -h

usage: llamaindex-cli [-h] {rag,download-llamapack,download-llamadataset,upgrade,upgrade-file,new-package} ...

LlamaIndex CLI tool.

options:
  -h, --help            show this help message and exit

commands:
  {rag,download-llamapack,download-llamadataset,upgrade,upgrade-file,new-package}
    rag                 Ask a question to a document / a directory of documents.
    download-llamapack  Download a llama-pack
    download-llamadataset
                        Download a llama-dataset
    upgrade             Upgrade a directory containing notebooks or python files.
    upgrade-file        Upgrade a single notebook or python file.
    new-package         Initialize a new llama-index package
```
