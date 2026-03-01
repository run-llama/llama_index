# Reve AI Image Generation Tool

LlamaIndex tool integration for [Reve AI](https://reve.com) native image generation API.

Provides four operations:
- **Create** — generate images from text prompts
- **Edit** — modify existing images with text guidance
- **Remix** — remix existing images with text guidance
- **Check Credits** — query your remaining credit balance

## Installation

```bash
pip install llama-index-tools-reve
```

## Configuration

Set your Reve API key as an environment variable:

```bash
export REVE_API_KEY="your-api-key"
```

Or pass it directly:

```python
from llama_index.tools.reve import ReveToolSpec

tool_spec = ReveToolSpec(api_key="your-api-key")
```

## Usage with LlamaIndex Agents

```python
import asyncio
from llama_index.tools.reve import ReveToolSpec

tool_spec = ReveToolSpec()
tools = tool_spec.to_tool_list()

# Use with any LlamaIndex agent
result = asyncio.run(tool_spec.reve_create_image(
    prompt="A futuristic city at sunset",
    aspect_ratio="16:9",
    test_time_scaling=5,
))
print(result)
```

## Usage as MCP Server

The package includes a standalone FastMCP server for use with Claude Desktop,
Cursor, or any MCP-compatible client.

### Claude Desktop / Cursor Configuration

Add to your MCP client config:

```json
{
  "mcpServers": {
    "reve_mcp": {
      "command": "python3",
      "args": ["/path/to/reve_mcp_server.py"],
      "env": {
        "REVE_API_KEY": "your-api-key"
      }
    }
  }
}
```

### Run Directly

```bash
REVE_API_KEY=your-key python reve_mcp_server.py
```

## Parameters

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Text description for generation/edit/remix |
| `aspect_ratio` | str | `"1:1"` | e.g. `"1:1"`, `"16:9"`, `"9:16"` |
| `test_time_scaling` | int | None | Quality 1-15. 3-5 = high, 10+ = max |
| `upscale_factor` | int | None | Post-processing upscale 2-4x |
| `remove_background` | bool | None | Remove image background |
| `fit_max_dim` | int | None | Resize to fit max dimension (pixels) |

### Edit / Remix Additional Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_url` | str | URL of the source image (auto-fetched and base64-encoded) |

## Response Format

All tools return JSON with the API response body plus metadata:

```json
{
  "data": [{"url": "https://..."}],
  "meta": {
    "credits_used": "2",
    "credits_remaining": "498",
    "model_version": "reve-v1",
    "request_id": "abc123",
    "content_violation": "false"
  }
}
```
