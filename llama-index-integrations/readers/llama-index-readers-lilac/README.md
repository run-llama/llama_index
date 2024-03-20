# Lilac reader

```bash
pip install llama-index-readers-papers

pip install llama-index-readers-lilac
```

[Lilac](https://lilacml.com/) is an open-source product that helps you analyze, enrich, and clean unstructured data with AI.

It can be used to analyze, clean, structure, and label data that can be used in downstream LlamaIndex and LangChain applications.

## Lilac projects

This assumes you've already run Lilac locally, and have a project directory with a dataset. For more details on Lilac projects, see [Lilac Projects](https://lilacml.com/projects/projects.html)

You can use any LlamaIndex loader to load data into Lilac, clean data, and then bring it back into LlamaIndex Documents.

## Usage

### LlamaIndex => Lilac

See [this notebook](https://github.com/lilacai/lilac/blob/main/notebooks/LlamaIndexLoader.ipynb) for getting data into Lilac from LlamaHub.

```python
import lilac as ll

# See: https://llamahub.ai/l/papers-arxiv
from llama_index.readers.papers import ArxivReader

loader = ArxivReader()
documents = loader.load_data(search_query="au:Karpathy")

# Set the project directory for Lilac.
ll.set_project_dir("./data")

# This assumes you already have a lilac project set up.
# If you don't, use ll.init(project_dir='./data')
ll.create_dataset(
    config=ll.DatasetConfig(
        namespace="local",
        name="arxiv-karpathy",
        source=ll.LlamaIndexDocsSource(
            # documents comes from the loader.load_data call in the previous cell.
            documents=documents
        ),
    )
)

# You can start a lilac server with. Once you've cleaned the dataset, you can come back into GPTIndex.
ll.start_server(project_dir="./data")
```

### Lilac => LlamaIndex Documents

```python
from llama_index.core import VectorStoreIndex, download_loader

from llama_index.readers.lilac import LilacReader

loader = LilacReader()
documents = loader.load_data(
    project_dir="~/my_project",
    # The name of your dataset in the project dir.
    dataset="local/arxiv-karpathy",
)

index = VectorStoreIndex.from_documents(documents)

index.query("How are ImageNet labels validated?")
```

This loader is designed to be used as a way to load data into [GPT Index](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used in a [LangChain](https://github.com/hwchase17/langchain) Agent.
