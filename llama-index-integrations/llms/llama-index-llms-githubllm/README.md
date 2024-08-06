# LlamaIndex Llms Integration: Githubllm

# GithubLLM

GithubLLM is a custom LLM (Language Model) interface that allows you to interact with AI models hosted on GitHub's inference endpoint, with automatic fallback to Azure when rate limits are reached.

## Features

- Seamless integration with GitHub-hosted AI models
- Automatic fallback to Azure when GitHub rate limits are reached
- Support for both completion and chat-based interactions
- Streaming support for both completion and chat responses
- Easy integration with LlamaIndex ecosystem

## Installation

```bash
pip install llama-index-llms-githubllm
```

## Usage

```python
from llama_index.llms.github import GithubLLM

# Initialize the LLM
llm = GithubLLM(
    model="gpt-4o",
    system_prompt="You are a helpful assistant.",
    use_azure_fallback=True,
)

# Completion
response = llm.complete("What is the capital of France?")
print(response.text)

# Chat
messages = [
    ChatMessage(role="user", content="Tell me about the French Revolution."),
    ChatMessage(
        role="assistant",
        content="The French Revolution was a period of major social and political upheaval in France...",
    ),
    ChatMessage(role="user", content="What were the main causes?"),
]
response = llm.chat(messages)
print(response.message.content)

# Streaming
for chunk in llm.stream_chat(
    [
        ChatMessage(
            role="user", content="Can you elaborate on the Reign of Terror?"
        )
    ]
):
    print(chunk.message.content, end="", flush=True)
```

## Configuration

- Set `GITHUB_TOKEN` environment variable for GitHub API access
- Set `AZURE_API_KEY` environment variable for Azure fallback

## Rate Limits

GithubLLM respects the following rate limits:

| Model Type | Requests/min | Requests/day | Tokens/request (in/out) | Concurrent Requests |
| ---------- | ------------ | ------------ | ----------------------- | ------------------- |
| Low        | 15           | 150          | 8000/4000               | 5                   |
| High       | 10           | 50           | 8000/4000               | 2                   |
| Embedding  | 15           | 150          | 64000                   | 5                   |

Note: Rate limits may vary based on your GitHub account type (Free, Copilot Individual, Copilot Business, Copilot Enterprise).

## Going to Production

For production use, replace the GitHub token with a paid Azure account token. No other code changes are required.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This library is for prototyping and experimentation. Ensure compliance with GitHub's and Azure's terms of service when using in production.
