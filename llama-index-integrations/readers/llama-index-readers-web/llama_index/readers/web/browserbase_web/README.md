# Browserbase Web Reader

[Browserbase](https://browserbase.com) is a developer platform to reliably run, manage, and monitor headless browsers.

Power your AI data retrievals with:

- [Serverless Infrastructure](https://docs.browserbase.com/under-the-hood) providing reliable browsers to extract data from complex UIs
- [Stealth Mode](https://docs.browserbase.com/features/stealth-mode) with included fingerprinting tactics and automatic captcha solving
- [Session Debugger](https://docs.browserbase.com/features/sessions) to inspect your Browser Session with networks timeline and logs
- [Live Debug](https://docs.browserbase.com/guides/session-debug-connection/browser-remote-control) to quickly debug your automation

## Installation and setup

- Get an API key and Project ID from [browserbase.com](https://browserbase.com) and set it in environment variables (`BROWSERBASE_API_KEY`, `BROWSERBASE_PROJECT_ID`).
- Install the [Browserbase SDK](http://github.com/browserbase/python-sdk):

```bash
pip install browserbase
```

## Loading documents

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

### Parameters

- `urls` Required. A list of URLs to fetch.
- `text_content` Retrieve only text content. Default is `False`.
- `session_id` Optional. Provide an existing Session ID.
- `proxy` Optional. Enable/Disable Proxies.## Loading images

You can also load screenshots of webpages (as bytes) for multi-modal models.

```python
from browserbase import Browserbase
from base64 import b64encode

browser = Browserbase()
screenshot = browser.screenshot("https://browserbase.com")

# Optional. Convert to base64
img_encoded = b64encode(screenshot).decode()
```
