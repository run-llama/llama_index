# LlamaIndex Readers Integration: Zyte-Serp

ZyteSerp can be used to add organic search results from Google Search. It takes a `query` and returns top search results urls.

## Instructions for ZyteSerpReader

### Setup and Installation

`pip install llama-index-readers-zyte-serp`

1. **Install zyte-api Package**: Ensure the `zyte-api` package is installed to use the ZyteSerpReader. Install it via pip with the following command:

   ```bash
   pip install zyte-api
   ```

2. **API Key**: Secure an API key from [Zyte](https://www.zyte.com/zyte-api/) to access the Zyte services.

### Using ZyteSerpReader

- **Initialization**: Initialize the ZyteWebReader by providing the API key, the desired mode of operation (`article`, `html-text`, or `html`), and any optional parameters for the Zyte API.

  ```python
  from llama_index.readers.zyte_serp import ZyteWebReader

  zyte_serp = ZyteSerpReader(
      api_key="your_api_key_here",
      extract_from="httpResponseBody",  # or "browserHtml"
  )
  ```

- **Loading Data**: To load data, use the `load_data` method with the URLs you wish to process.

```python
documents = zyte_serp.load_data(query="llama index docs")
```

### Example Usage

Here is an example demonstrating how to initialize the ZyteWebReader, load document from a URL.

```python
# Initialize the ZyteSerpReader with your API key
zyte_serp = ZyteSerpReader(
    api_key="your_api_key_here",  # Replace with your actual API key
)

# Load documents from Paul G
documents = zyte_serp.load_data(urls="llama index docs")

# Display the document
print(documents)
```
