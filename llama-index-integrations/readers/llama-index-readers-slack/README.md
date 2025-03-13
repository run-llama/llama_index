# Slack Loader

```bash
pip install llama-index-readers-slack
```

This loader loads conversations from Slack. The user specifies `channel_ids` and we fetch conversations from those `channel_ids`.

## Usage

Here's an example usage of the SlackReader.

```python
import os
from llama_index.readers.slack import SlackReader

slack_token = os.getenv("SLACK_TOKEN")
channel_ids = ["C08J3PZD5B2"]  # Replace with your channel_id
reader = SlackReader(slack_token=slack_token)
documents = reader.load_data(channel_ids=channel_ids)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).