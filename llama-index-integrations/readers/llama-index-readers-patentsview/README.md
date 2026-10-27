# Patentsview Loader

```bash
pip install llama-index-readers-patentsview
```

This loader loads patent abstract from `a list of patent numbers` with API provided by [Patentsview](https://patentsview.org/).

> Note: The API now requires an API key which can be requested on [this page](https://patentsview-support.atlassian.net/servicedesk/customer/portal/1).
> PatentsviewReader looks for its value in the environmental variable `PATENTSVIEW_API_KEY`

> To use another host that serves the same `/api/v1/patent` endpoint and query
> language, pass `base_url="https://host"` (scheme and host) or set
> `PATENTSVIEW_BASE_URL`. The default is `https://search.patentsview.org`.

## Usage

Here'a an example usage of PatentsviewReader.

```python
from llama_index.readers.patentsview import PatentsviewReader

loader = PatentsviewReader(api_key=os.getenv("PATENTSVIEW_API_KEY"))
patents = ["8848839", "10452978"]
abstracts = loader.load_data(patents)
```

This loader is designed for loading data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index).
