# LlamaIndex Readers Integration: TwitterTweet

## Overview

The TwitterTweet Reader allows you to read tweets of a specified Twitter handle. It retrieves tweets from Twitter using the Twitter API.

### Installation

You can install the TwitterTweet Reader via pip:

```bash
pip install llama-index-readers-twitter
```

Check [Twitter API](https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api) on how to get access to twitter API.

### Usage

```python
from llama_index.readers.twitter_tweet import TwitterTweetReader

# Initialize TwitterTweetReader
reader = TwitterTweetReader(
    bearer_token="<Twitter Bearer Token>", num_tweets=100
)

# Load tweets of user twitter handles
documents = reader.load_data(twitterhandles=["user1", "user2"])
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
