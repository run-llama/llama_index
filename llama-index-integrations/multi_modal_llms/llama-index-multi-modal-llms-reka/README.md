# LlamaIndex Multi_Modal_Llms Integration: Reka

LlamaIndex Multi-Modal LLMs Integration: Reka
This package provides integration between the Reka multi-modal language model and LlamaIndex, allowing you to use Reka's powerful language models with image input capabilities in your LlamaIndex applications.
Installation
To use this integration, you need to install the llama-index-multi-modal-llms-reka package:

```bash
pip install llama-index-multi-modal-llms-reka
```

To obtain an API key, please visit https://platform.reka.ai/
Our baseline models always available for public access are:

- `reka-edge`
- `reka-flash`
- `reka-core`

Other models may be available. The Get Models API allows you to list what models you have available to you. Using the Python SDK, it can be accessed as follows:

```python
from reka.client import Reka

client = Reka()
print(client.models.get())
```

# Usage

Here are some examples of how to use the Reka Multi-Modal LLM integration with LlamaIndex:
Initialize the Reka Multi-Modal LLM client

```python
import os
from llama_index.llms.reka import RekaMultiModalLLM

api_key = os.getenv("REKA_API_KEY")
reka_mm_llm = RekaMultiModalLLM(model="reka-flash", api_key=api_key)
```

# Chat completion with image

```python
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import ImageDocument

# Create an ImageDocument with the image URL or local file path
image_doc = ImageDocument(image_url="https://example.com/image.jpg")
# Or for a local file:
image_doc = ImageDocument(image_path="/path/to/local/image.jpg")

messages = [
    ChatMessage(
        role=MessageRole.SYSTEM, content="You are a helpful assistant."
    ),
    ChatMessage(
        role=MessageRole.USER, content="What do you see in this image?"
    ),
]

response = reka_mm_llm.chat(messages, image_documents=[image_doc])
print(response.message.content)
```

# Text completion with image

```python
from llama_index.core.schema import ImageDocument

image_doc = ImageDocument(image_url="https://example.com/image.jpg")
prompt = "Describe the contents of this image:"

response = reka_mm_llm.complete(prompt, image_documents=[image_doc])
print(response.text)
```

# Streaming Responses

Streaming chat completion with image

```python
messages = [
    ChatMessage(
        role=MessageRole.SYSTEM, content="You are a helpful assistant."
    ),
    ChatMessage(
        role=MessageRole.USER, content="Describe the colors in this image."
    ),
]

for chunk in reka_mm_llm.stream_chat(messages, image_documents=[image_doc]):
    print(chunk.delta, end="", flush=True)
```

Streaming text completion with image

```
prompt = "List the objects you can see in this image:"

for chunk in reka_mm_llm.stream_complete(prompt, image_documents=[image_doc]):
    print(chunk.delta, end="", flush=True)
```

# Asynchronous Usage

```python
import asyncio


async def main():
    # Async chat completion with image
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM, content="You are a helpful assistant."
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="What's the main subject of this image?",
        ),
    ]
    response = await reka_mm_llm.achat(messages, image_documents=[image_doc])
    print(response.message.content)

    # Async text completion with image
    prompt = "Describe the background of this image:"
    response = await reka_mm_llm.acomplete(prompt, image_documents=[image_doc])
    print(response.text)

    # Async streaming chat completion with image
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM, content="You are a helpful assistant."
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="What objects are visible in this image?",
        ),
    ]
    async for chunk in await reka_mm_llm.astream_chat(
        messages, image_documents=[image_doc]
    ):
        print(chunk.delta, end="", flush=True)

    # Async streaming text completion with image
    prompt = "List the colors present in this image:"
    async for chunk in await reka_mm_llm.astream_complete(
        prompt, image_documents=[image_doc]
    ):
        print(chunk.delta, end="", flush=True)


asyncio.run(main())
```

# Running Tests

To run the tests for this integration, you'll need to have pytest and pytest-asyncio installed. You can install them using pip:

```
pip install pytest pytest-asyncio
```

Then, set your Reka API key as an environment variable:

```
export REKA_API_KEY=your_api_key_here
```

Now you can run the tests using pytest:

```
pytest tests/test_multi_modal_llms_reka.py -v
```

To run only mock integration tests without remote connections:

```
pytest tests/test_multi_modal_llms_reka.py -v -k "mock"
```

Note: The test file should be named test_multi_modal_llms_reka.py and placed in the appropriate directory.

The Reka Multi-Modal LLM supports various image input formats, including URLs, local file paths, and base64-encoded image strings.
When using local file paths, make sure the files are accessible to your application.
The model can process multiple images in a single request by passing a list of ImageDocument objects.

# Contributing

Contributions to improve this integration are welcome. Please ensure that you add or update tests as necessary when making changes.
When adding new features or modifying existing ones, please update this README to reflect those changes.
