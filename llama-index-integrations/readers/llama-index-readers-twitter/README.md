# LlamaIndex Twitter and X readers

## Overview

This package loads recent posts from an X handle into LlamaIndex documents. Use
`TwitterTweetReader` with the official X API. Use `XquikTweetReader` with the
Xquik Python SDK.

### Installation

You can install the TwitterTweet Reader via pip:

```bash
pip install llama-index-readers-twitter
```

Install the optional Xquik client when you need Xquik:

```bash
pip install "llama-index-readers-twitter[xquik]"
```

Check the [X API documentation](https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api)
for official API access.

### Official X API

```python
from llama_index.readers.twitter import TwitterTweetReader

# Initialize TwitterTweetReader
reader = TwitterTweetReader(
    bearer_token="<Twitter Bearer Token>", num_tweets=100
)

# Load tweets of user twitter handles
documents = reader.load_data(twitterhandles=["user1", "user2"])
```

### Xquik

Set `XQUIK_API_KEY` in your secret store. The reader sends it only to Xquik
through the typed `x_twitter_scraper` SDK.

```python
import os

from llama_index.readers.twitter import XquikTweetReader

reader = XquikTweetReader(
    api_key=os.environ["XQUIK_API_KEY"],
    num_tweets=100,
)

documents = reader.load_data(twitterhandles=["user1", "@user2"])
```

Each handle produces one document. The reader loads one bounded page of 1 to
300 recent posts. It excludes replies and parent posts. Xquik meters returned
posts. See the [Xquik Python SDK guide](https://docs.xquik.com/sdks/python) for
current authentication and usage details.

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.

Xquik is an independent third-party service. Not affiliated with X Corp.
"Twitter" and "X" are trademarks of X Corp.
