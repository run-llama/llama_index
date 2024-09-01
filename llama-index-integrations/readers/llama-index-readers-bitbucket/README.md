# Bitbucket Loader

```bash
pip install llama-index-readers-bitbucket
```

This loader utilizes the Bitbucket API to load the files inside a Bitbucket repository as Documents in an index.

## Usage

To use this loader, you need to provide as environment variables the `BITBUCKET_API_KEY` and the `BITBUCKET_USERNAME`.

```python
import os
from llama_index.core import VectorStoreIndex, download_loader

os.environ["BITBUCKET_USERNAME"] = "myusername"
os.environ["BITBUCKET_API_KEY"] = "myapikey"

base_url = "https://myserver/bitbucket"
project_key = "mykey"

from llama_index.readers.bitbucket import BitbucketReader

loader = BitbucketReader(
    base_url=base_url,
    project_key=project_key,
    branch="refs/heads/develop",
    repository="ms-messaging",
)
documents = loader.load_data()

index = VectorStoreIndex.from_documents(documents)
```

This loader is designed to be used as a way to load data into [Llama Index](https://github.com/run-llama/llama_index/).

For a step-by-step guide, checkout this [tutorial](https://lejdiprifti.com/2023/12/16/ask-your-bitbucket-rag-with-llamaindex-and-bitbucket/)
