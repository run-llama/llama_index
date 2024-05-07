# Browserbase Web Reader

[Browserbase](https://browserbase.com) is a serverless platform for running headless browsers, it offers advanced debugging, session recordings, stealth mode, integrated proxies and captcha solving.

## Installation and Setup

- Get an API key from [browserbase.com](https://browserbase.com) and set it in environment variables (`BROWSERBASE_API_KEY`).
- Install the [Browserbase SDK](http://github.com/browserbase/python-sdk):

```
pip install browserbase
```

## Usage

### Loading documents

You can load webpages into LlamaIndex using `BrowserbaseWebReader`. Optionally, you can set `text_content` parameter to convert the pages to text-only representation.

```python
from llama_index.readers.web import BrowserbaseWebReader


reader = BrowserbaseWebReader()
docs = reader.load_data(
    urls=[
        "https://example.com",
    ],
    # Text mode
    text_content=False,
)
```

### Loading images

You can also load screenshots of webpages (as bytes) for multi-modal models.

```python
from browserbase import Browserbase
from base64 import b64encode

browser = Browserbase()
screenshot = browser.screenshot("https://browserbase.com")

# Optional. Convert to base64
img_encoded = b64encode(screenshot).decode()
```
