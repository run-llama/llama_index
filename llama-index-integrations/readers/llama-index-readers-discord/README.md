# Discord Loader

```bash
pip install llama-index-readers-discord
```

This loader loads conversations from Discord. The user specifies `channel_ids` and we fetch conversations from
those `channel_ids`.

## Usage

Here's an example usage of the DiscordReader.

```python
import os

from llama_index.readers.discord import DiscordReader

discord_token = os.getenv("DISCORD_TOKEN")
channel_ids = [1057178784895348746]  # Replace with your channel_id
reader = DiscordReader(discord_token=discord_token)
documents = reader.load_data(channel_ids=channel_ids)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
