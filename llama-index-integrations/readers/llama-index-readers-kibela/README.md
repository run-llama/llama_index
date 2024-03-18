# Kibela Reader

```bash
pip install llama-index-readers-kibela
```

This reader fetches article from your [Kibela](https://kibe.la/) notes using the GraphQL API.

# Usage

Here's an example of how to use it. You can get your access token from [here](https://my.kibe.la/settings/access_tokens).

```python
import os
from llama_index.readers.kibela import KibelaReader

team = os.environ["KIBELA_TEAM"]
token = os.environ["KIBELA_TOKEN"]

reader = KibelaReader(team=team, token=token)
documents = reader.load_data()
```

Alternately, you can also use download_loader from llama_index

```python
import os

from llama_index.readers.kibela import KibelaReader

team = os.environ["KIBELA_TEAM"]
token = os.environ["KIBELA_TOKEN"]

reader = KibelaReader(team=team, token=token)
documents = reader.load_data()
```
