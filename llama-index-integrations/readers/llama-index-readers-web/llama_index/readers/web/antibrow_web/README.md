# AntiBrow Web Reader

[AntiBrow](https://antibrow.com) is an antidetect browser driven through the standard Playwright API, running on your own machine.

The unit of state is a profile, not a session:

- Persistent identity - the same profile name always gets the same fingerprint, cookies and storage, so a page behind a login loads already signed in
- One proxy per profile - answered inside the browser engine, so nothing is loaded as an extension
- Local - the browser runs where your code runs, so there are no browser-hours to buy

## Installation and setup

- Get an API key from [antibrow.com](https://antibrow.com/dashboard) and pass it as `api_key`, or run `python -m antibrow login --key ...` once to store it.
- Install the [AntiBrow SDK](https://github.com/antibrow/antibrow):

```bash
pip install antibrow
```

The browser engine downloads on the first launch and is cached.

## Usage

```python
from llama_index.readers.web import AntibrowWebReader

reader = AntibrowWebReader(profile="research-01")
documents = reader.load_data(urls=["https://example.com"], selector="main")
```

Each `Document` carries the final url, the page title, the HTTP status and the profile name in its metadata.

Two agents that must not share an identity need two profile names, not two readers on one profile. Pass `temporary=True` for a one-off anonymous read.

A persistent identity removes the tells that come from starting over every run - a fresh profile, a stock automation fingerprint, your own IP. It does not promise that a given site will accept an automated session.
