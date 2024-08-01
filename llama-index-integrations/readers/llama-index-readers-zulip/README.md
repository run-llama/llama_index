## Zulip Loader

The Zulip Loader is a Python script that allows you to load data from Zulip streams using a Zulip bot's API token. It fetches messages from specified streams or all streams if none are specified, and returns a list of documents with the stream content.

### Prerequisites

Create a Zulip bot and obtain its API token. Follow the instructions in the Zulip documentation to create a bot and get the API key (token).

Set the ZULIP_TOKEN environment variable to your Zulip bot's API token:

```bash
export ZULIP_TOKEN="your-zulip-bot-api-token"
```

### Installation

You can install the Zulip Reader via pip:

```bash
pip install llama-index-readers-zulip
```

### Usage

Use the ZulipReader class to load data from Zulip streams:

```python
from zulip_loader import ZulipReader

# Initialize the ZulipReader with the bot's email and Zulip domain
reader = ZulipReader(
    zulip_email="your-bot-email@your-zulip-domain.zulipchat.com",
    zulip_domain="your-zulip-domain.zulipchat.com",
)

# Load data from all streams
data = reader.load_data(reader.get_all_streams())

# Load data from specific streams
stream_names = ["stream1", "stream2"]
data = reader.load_data(stream_names)
# This will return a list of documents containing the content of the specified streams.
```

For more customization, you can pass the `reverse_chronological` parameter to the load_data() method to indicate the order of messages in the output.

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
