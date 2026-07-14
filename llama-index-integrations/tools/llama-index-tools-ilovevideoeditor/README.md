# LlamaIndex Tool: iLoveVideoEditor

This tool allows LlamaIndex agents to render MP4 videos from declarative [VideoJSON](https://ilovevideoeditor.com/docs/api-guide) specifications using the [iLoveVideoEditor](https://ilovevideoeditor.com) cloud rendering API.

## Features

- 🎬 **Programmatic video rendering**: Turn VideoJSON specs into MP4 videos via a single tool call
- ⏳ **Wait or queue**: Poll until the render finishes, or queue and check status later
- 🤖 **Agent-ready**: Works directly with LlamaIndex agents via `BaseToolSpec`
- 🛡️ **Error handling**: Graceful error handling with informative messages

## Requirements

- Python >= 3.10
- llama-index-core >= 0.13.0
- requests >= 2.31.0
- An iLoveVideoEditor API key (get one from the [dashboard](https://ilovevideoeditor.com/dashboard))

## Installation

```bash
pip install llama-index-tools-ilovevideoeditor
```

## Required Environment Variables

```env
ILOVEVIDEOEDITOR_API_KEY=vf_live_...
```

## Usage

```python
from llama_index.tools.ilovevideoeditor import ILoveVideoEditorTool
from llama_index.core.agent import ReActAgent

# Initialize the tool spec (reads ILOVEVIDEOEDITOR_API_KEY from the environment)
tool_spec = ILoveVideoEditorTool()

# Convert the spec to a list of FunctionTools
tools = tool_spec.to_tool_list()

# Create an agent with the rendering tools
agent = ReActAgent.from_tools(tools, verbose=True)

# The agent can now render videos from VideoJSON specs
agent.chat(
    "Render a 3-second 1080p video with the text 'Hello from LlamaIndex' "
    "using the iLoveVideoEditor tool."
)
```

Direct usage without an agent:

```python
from llama_index.tools.ilovevideoeditor import ILoveVideoEditorTool

tool = ILoveVideoEditorTool()

result = tool.render_video(
    {
        "name": "hello",
        "layers": [
            {
                "type": "text",
                "settings": {
                    "startTime": 0,
                    "duration": 3,
                    "text": "Hello from LlamaIndex",
                    "fontSize": 64,
                    "color": "#ffffff",
                },
            }
        ],
    }
)
# '{"jobId": "425ba18a-...", "status": "completed", "downloadUrl": "https://..."}'
```

Queue without waiting, then check status later:

```python
queued = tool.render_video(video_json, wait_for_completion=False)
# '{"jobId": "425ba18a-...", "status": "queued"}'

status = tool.get_render_status("425ba18a-...")
# '{"jobId": "425ba18a-...", "status": "completed", "downloadUrl": "https://..."}'
```

## Configuration options

- `api_key`: API key. Defaults to the `ILOVEVIDEOEDITOR_API_KEY` environment variable.
- `base_url`: API base URL. Defaults to the `ILOVEVIDEOEDITOR_API_BASE` environment variable or `https://api.ilovevideoeditor.com`.
- `max_wait_seconds`: Maximum time to poll for render completion (default `300`).
- `poll_interval_seconds`: Interval between status polls (default `2`).

## API reference

For the full VideoJSON schema and REST API reference, see the [iLoveVideoEditor API docs](https://ilovevideoeditor.com/docs/api/).

## Examples

See the [`examples`](./examples) directory for more usage examples.

## Development

Run tests:

```bash
make test
```

Run linters:

```bash
make lint
```

Format code:

```bash
make format
```
