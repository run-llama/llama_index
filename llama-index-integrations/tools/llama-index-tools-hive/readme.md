# Hive Tool for LlamaIndex

This tool provides integration with Hive Intelligence, enabling powerful crypto and Web3 intelligence capabilities.

## Installation

Install the package using pip:

```bash
pip install llama-index-tools-hive
```

## Setup

Set your Hive API key as an environment variable:

```bash
export HIVE_API_KEY=your_api_key_here
```

On Windows:

```cmd
set HIVE_API_KEY=your_api_key_here
```

## Basic Usage

### Import and Initialize

```python
from llama_index.tools.hive import HiveToolSpec, HiveSearchMessage

# Initialize with API key and optional search parameters
hive_tool = HiveToolSpec(
    api_key="your_api_key_here",  # or it will use HIVE_API_KEY env var
)
```

### Simple Search

```python
# Single prompt search
results = hive_tool.search(
    prompt="What is the current price of Ethereum?", include_data_sources=True
)
print("Search results:", results)
```

### Chat-style Conversation

```python
# Create conversation messages
chat_msgs = [
    HiveSearchMessage(role="user", content="Price of what?"),
    HiveSearchMessage(role="assistant", content="Please specify asset."),
    HiveSearchMessage(role="user", content="BTC"),
]

# Execute chat search
results = hive_tool.search(messages=chat_msgs, include_data_sources=True)
print("Chat response:", results)
```

## Advanced Usage

### Search with Custom Parameters

```python
results = hive_tool.search(
    prompt="Analyze the latest trends in DeFi",
    include_data_sources=True,  # Include sources in response
)
```

## API Reference

### HiveToolSpec

#### `__init__(self, api_key: str, temperature: Optional[float] = None, top_k: Optional[int] = None, top_p: Optional[float] = None)`

- `api_key`: Your Hive Intelligence API key
- `temperature`: Controls randomness (0.0 to 1.0) - optional
- `top_k`: Limit number of results returned - optional
- `top_p`: Nucleus sampling parameter - optional

#### `search(self, prompt: str = None, messages: List[HiveSearchMessage] = None, include_data_sources: bool = False) -> HiveSearchResponse`

Parameters:

- `prompt`: The search query or question (string)
- `messages`: List of HiveSearchMessage objects for chat-style conversations
- `include_data_sources`: Whether to include data sources in the response

Returns:

- `HiveSearchResponse` containing the search results

## Examples

### Full Example Script

```python
from llama_index.tools.hive import HiveToolSpec, HiveSearchMessage


def main():
    # Initialize with API key and search parameters
    hive = HiveToolSpec(api_key="your_api_key_here")

    # Simple search
    print("--- Simple Search ---")
    results = hive.search(
        prompt="What is the current price of Bitcoin?",
        include_data_sources=True,
    )
    print("Results:", results)

    # Chat conversation
    print("\n--- Chat Conversation ---")
    chat = [
        HiveSearchMessage(role="user", content="Tell me about"),
        HiveSearchMessage(
            role="assistant", content="What would you like to know about?"
        ),
        HiveSearchMessage(role="user", content="Ethereum upgrades"),
    ]
    results = hive.search(messages=chat)
    print("Chat response:", results)


if __name__ == "__main__":
    main()
```
