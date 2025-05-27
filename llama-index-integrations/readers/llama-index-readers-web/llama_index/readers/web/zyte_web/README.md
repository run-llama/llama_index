# ZyteWebReader

## Instructions for ZyteWebReader

### Setup and Installation

`pip install llama-index-readers-web`

1. **Install zyte-api Package**: Ensure the `zyte-api` package is installed to use the ZyteWebReader. Install it via pip with the following command:

   ```bash
   pip install zyte-api
   ```

   Additionally if you are planning on using "html-text" mode, you'll also need to install `html2text`

   ```bash
   pip install html2text
   ```

2. **API Key**: Secure an API key from [Zyte](https://www.zyte.com/zyte-api/) to access the Zyte services.

### Using ZyteWebReader

- **Initialization**: Initialize the ZyteWebReader by providing the API key, the desired mode of operation (`article`, `html-text`, or `html`), and any optional parameters for the Zyte API.

  ```python
  from llama_index.readers.web.zyte_web.base import ZyteWebReader

  zyte_reader = ZyteWebReader(
      api_key="your_api_key_here",
      mode="article",  # or "html" or "html-text"
      n_conn=5,  # number of concurrent connections
      download_kwargs={"additional": "parameters"},
  )
  ```

- **Loading Data**: To load data, use the `load_data` method with the URLs you wish to process.

```python
documents = zyte_reader.load_data(urls=["http://example.com"])
```

### Example Usage

Here is an example demonstrating how to initialize the ZyteWebReader, load document from a URL.

```python
# Initialize the ZyteWebReader with your API key and desired mode
zyte_reader = ZyteWebReader(
    api_key="your_api_key_here",  # Replace with your actual API key
    mode="article",  # Choose between "article", "html-text", and "html"
    download_kwargs={
        "additional": "parameters"
    },  # Optional additional parameters
)

# Load documents from Paul Graham's essay URL
documents = zyte_reader.load_data(urls=["http://www.paulgraham.com/"])

# Display the document
print(documents)
```
