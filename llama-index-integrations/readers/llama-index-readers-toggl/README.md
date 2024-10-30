# Toggl reader

```bash
pip install llama-index-readers-toggl
```

This loader fetches time entries from Toggl workspace and project into `Document`s.

Before working with Toggl's API, you need to get API token:

1. Log in to Toggl
2. [Open profile](https://track.toggl.com/profile)
3. Scroll down and click `-- Click to reveal --` for API token

## Usage

```python
from llama_index.readers.toggl import TogglReader
from llama_index.readers.toggl.dto import TogglOutFormat
import datetime

reader = TogglReader(api_token="{{YOUR_API_TOKEN}}")

docs = reader.load_data(
    workspace_id="{{WORKSPACE_ID}}",
    project_id="{{PROJECT_ID}}",
    start_date=datetime.datetime.now() - datetime.timedelta(days=7),
    out_format=TogglOutFormat.markdown,
)
```

## Examples

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
