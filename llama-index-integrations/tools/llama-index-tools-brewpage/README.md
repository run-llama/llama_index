# BrewPage Tool Spec

This package provides a LlamaIndex ToolSpec for [BrewPage](https://brewpage.app), a free HTML/Markdown/JSON/file hosting service with no authentication required.

## Installation

```bash
pip install llama-index-tools-brewpage
```

## Quick Start

```python
from llama_index.tools.brewpage import BrewPageToolSpec

# Create the tool spec
tool = BrewPageToolSpec()

# Publish HTML content
link = tool.publish_content(
    "<h1>Hello, World!</h1>",
    namespace="public",
    ttl_days=15
)
print(f"Published at: {link}")
# Output: Published at: https://brewpage.app/public/abc123def4

# Retrieve content
content = tool.get_content("public", "abc123def4")
print(content)
# Output: <h1>Hello, World!</h1>
```

## Features

- **Publish content**: Host HTML or Markdown with automatic short URL generation
- **Retrieve content**: Fetch hosted content by namespace and ID
- **Custom TTL**: Set content expiration (1-30 days, default 15 days)
- **No authentication**: Free tier, no API keys required
- **Size limits**: Up to 5 MB per resource

## API Reference

### `publish_content(content, namespace="public", ttl_days=None)`

Publish HTML or Markdown content to BrewPage.

**Parameters:**
- `content` (str): The HTML or Markdown content to publish (max 5 MB)
- `namespace` (str): Namespace for the content (default: "public")
- `ttl_days` (int, optional): Days until content expires (default: 15, max: 30)

**Returns:** (str) Short URL link to the published content

**Example:**
```python
link = tool.publish_content(
    "# My Documentation\n\nThis is a test.",
    namespace="public",
    ttl_days=7
)
```

### `get_content(namespace, short_id)`

Retrieve content from BrewPage.

**Parameters:**
- `namespace` (str): Namespace where content is stored
- `short_id` (str): 10-character short ID of the content

**Returns:** (str) The full HTML or Markdown content

**Example:**
```python
content = tool.get_content("public", "abc123def4")
```

## Use Cases

- **AI Documentation**: Generate and host documentation from LLM outputs
- **Quick Sharing**: Create shareable links to AI-generated content
- **Template Hosting**: Store HTML/Markdown templates for reuse
- **Content Management**: Organize content by namespace with automatic expiration

## Namespaces

- **`public`**: Recommended for shared content (publicly accessible)
- **Private namespaces**: For personal use only (e.g., "my-private-ns")

Note: The "public" namespace is shared; content in other namespaces is private to the creator.

## Links

- **BrewPage**: https://brewpage.app
- **OpenAPI Spec**: https://raw.githubusercontent.com/kochetkov-ma/brewpage-openapi/main/openapi/openapi.yaml
- **GitHub**: https://github.com/kochetkov-ma/brewpage-openapi

## License

MIT
