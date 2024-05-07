# Hatena Blog Loader

```bash
pip install llama-index-readers-hatena-blog
```

This loader fetches article from your own [Hatena Blog](https://hatenablog.com/) blog posts using the AtomPub API.

You can get AtomPub info from the admin page after logging into Hatena Blog.

## Usage

Here's an example usage of the HatenaBlogReader.

```python
import os

from llama_index.readers.hatena_blog import HatenaBlogReader

root_endpoint = os.getenv("ATOM_PUB_ROOT_ENDPOINT")
api_key = os.getenv("ATOM_PUB_API_KEY")
username = os.getenv("HATENA_BLOG_USERNAME")

reader = HatenaBlogReader(
    root_endpoint=root_endpoint, api_key=api_key, username=username
)
documents = reader.load_data()
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
