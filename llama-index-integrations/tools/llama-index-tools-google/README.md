# LlamaIndex Tools Integration: Google

### Provides a set of tools to interact with Google services.

- you need to enable each of the below services in your google cloud console, under a same API key for a service, in
  order to use them.

### Quick Start:

```python
# pip install llama-index-tools-google
from llama_index.tools.google import GmailToolSpec
from llama_index.tools.google import GoogleCalendarToolSpec
from llama_index.tools.google import GoogleSearchToolSpec
```

#### [custom search service](https://developers.google.com/custom-search/v1/overview)

```python
google_spec = GoogleSearchToolSpec(key="your-key", engine="your-engine")
```

- `key` collected from your service console
- `engine` which represents the search engine to use, you can create a custom search
  engine [here](https://cse.google.com/cse/all)

#### [calendar read, create]()

- requires OAuth 2.0 credentials, you can create them [here](https://console.developers.google.com/apis/credentials)
- store oAuth`credentials.json` in the same directory as the runnable agent.
- you will need to manually approve the Oath every time this tool is invoked

#### [gmail read, create]()

- same as calendar

### known defects

- the calendar tool create is not able to generate an event if the agent is not able to infer the timezome
