# LlamaIndex LLMs Integration: Chrome AI (Prompt API)

Use Chrome's built-in **Gemini Nano** model via the
[Chrome Prompt API](https://chromestatus.com/feature/5134603979063296)
directly from LlamaIndex — no API keys, no network calls, fully on-device.

```bash
pip install llama-index-llms-chrome-ai
playwright install chromium   # only needed for the Playwright driver
```

## Prerequisites

| Requirement | Details |
|---|---|
| Chrome version | 127 or later (138+ recommended) |
| Prompt API enabled | Origin-trial token **or** `--enable-features=PromptAPIForGeminiNano` flag |
| Gemini Nano downloaded | Open `chrome://components` → *Optimization Guide On Device Model* → **Check for update** |
| `playwright` package | `pip install playwright` |

## Usage

### Basic completion

```python
from llama_index.llms.chrome_ai import ChromeAI

llm = ChromeAI()
response = llm.complete("Explain quantum entanglement in one sentence.")
print(response.text)
```

### Chat

```python
from llama_index.core.base.llms.types import ChatMessage, MessageRole

messages = [
    ChatMessage(role=MessageRole.SYSTEM, content="You are a concise assistant."),
    ChatMessage(role=MessageRole.USER, content="What is the speed of light?"),
]
response = llm.chat(messages)
print(response.message.content)
```

### Streaming

```python
for chunk in llm.stream_complete("Write a haiku about the ocean."):
    print(chunk.delta, end="", flush=True)
print()
```

### Async

```python
import asyncio

async def main():
    llm = ChromeAI()
    response = await llm.acomplete("Name three programming languages.")
    print(response.text)

asyncio.run(main())
```

### Check model availability

```python
llm = ChromeAI()
print(llm.check_availability())
# "available" | "downloadable" | "downloading" | "unavailable"
```

## Configuration

```python
llm = ChromeAI(
    # Sampling temperature (0.0–2.0). None = Chrome AI default.
    temperature=0.8,
    # Top-k parameter. None = Chrome AI default.
    top_k=3,
    # Path to Chrome binary. None = use system Chrome via Playwright channel.
    chrome_executable_path="/usr/bin/google-chrome",
    # Run browser headlessly (default: True).
    headless=True,
    # Per-operation timeout in seconds.
    timeout=60.0,
    # Extra Chrome launch flags.
    additional_launch_args=["--enable-features=PromptAPIForGeminiNano"],
)
```

## How it works

`ChromeAI` uses [Playwright](https://playwright.dev/python/) to launch a Chrome
browser instance and calls `window.LanguageModel` (Chrome 138+) or
`window.ai.languageModel` (Chrome 127–137) via JavaScript evaluation.
Streaming is implemented by exposing a Python callback to JavaScript that
receives each token delta as it is generated.

Because the model runs entirely inside Chrome, no data leaves the device.
