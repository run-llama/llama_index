# LlamaIndex Readers Integration: Zyte-Serp

ZyteSerp can be used to add organic search results from Google Search. It takes a `query` and returns top search results urls.

## Instructions for ZyteSerpReader

### Setup and Installation

`pip install llama-index-readers-zyte-serp`

Secure an API key from [Zyte](https://www.zyte.com/zyte-api/) to access the Zyte services.

### Using ZyteSerpReader

- **Initialization**: Initialize the ZyteSerpReader by providing the API key and the option for extraction ("httpResponseBody" or "browserHtml").

  ```python
  from llama_index.readers.zyte_serp import ZyteSerpReader

  zyte_serp = ZyteSerpReader(
      api_key="your_api_key_here",
      extract_from="httpResponseBody",  # or "browserHtml"
  )
  ```

- **Loading Data**: To load search results, use the `load_data` method with the query you wish to search.

```python
documents = zyte_serp.load_data(query="llama index docs")
```

### Example Usage

Here is an example demonstrating how to initialize the ZyteSerpReader and get top search URLs.
Further the content from these URLs can be loaded using ZyteWebReader in "article" mode to obtain just the article content from webpage.

```python
from llama_index.readers.zyte_serp import ZyteSerpReader
from llama_index.readers.web.zyte.base import ZyteWebReader

# Initialize the ZyteSerpReader with your API key
zyte_serp = ZyteSerpReader(
    api_key="your_api_key_here",  # Replace with your actual API key
)

# Get the search results (URLs from google search results)
search_urls = zyte_serp.load_data(query="llama index docs")

# Display the results
print(search_urls)

urls = [result.text for result in search_urls]

# Initialize the ZyteWebReader to load the content from search results
zyte_web = ZyteWebReader(
    api_key="your_api_key_here",  # Replace with your actual API key
    mode="article",
)

documents = zyte_web.load_data(urls)
print(documents)
```
