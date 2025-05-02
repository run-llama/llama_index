# LlamaIndex Llms Integration: Gigachat

This package provides a Gigachat integration as LLM Module.

## Installation

using poetry:

```shell
poetry add llama-index-llms-gigachat-ru
```

or using pip:

```shell
pip install llama-index-llms-gigachat-ru
```

## Basic Usage

To initialize the Gigachat integration, you need to provide the `credentials`
and optionally set `verify_ssl_certs` to `False`
if you want to disable SSL certificate verification.

Then you will be able to use the `complete` method to get the completion.

```python
from llama_index.llms.gigachat import GigaChatLLM

llm = GigaChatLLM(
    credentials="NjI0M2M4MzctNmEwMi00ZjhmLWIzYmEtMTBlMzdhZjI4NzNhOjgzYjM3YzFkLWQ3MTEtNGVhYi04Y2Q0LTkwODM5ZjI4MDg1Zg==",
    verify_ssl_certs=False,
)
resp = llm.complete("What is the capital of France?")
print(resp)
```

Also, you can use it asynchronously:

```python
import asyncio
from llama_index.llms.gigachat import GigaChatLLM

llm = GigaChatLLM(
    credentials="NjI0M2M4MzctNmEwMi00ZjhmLWIzYmEtMTBlMzdhZjI4NzNhOjgzYjM3YzFkLWQ3MTEtNGVhYi04Y2Q0LTkwODM5ZjI4MDg1Zg==",
    verify_ssl_certs=False,
)


async def main():
    resp = await llm.acomplete("What is the capital of France?")
    print(resp)


asyncio.run(main())
```

And as a chat module:

```python
import asyncio

from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.gigachat import GigaChatLLM

llm = GigaChatLLM(
    credentials="NjI0M2M4MzctNmEwMi00ZjhmLWIzYmEtMTBlMzdhZjI4NzNhOjgzYjM3YzFkLWQ3MTEtNGVhYi04Y2Q0LTkwODM5ZjI4MDg1Zg==",
    verify_ssl_certs=False,
)

chat = SimpleChatEngine.from_defaults(
    llm=llm,
)


async def main():
    resp = await chat.achat("What is the capital of France?")
    print(resp)


asyncio.run(main())
```

And as streaming completion:

```python
import asyncio

from llama_index.llms.gigachat import GigaChatLLM

llm = GigaChatLLM(
    credentials="NjI0M2M4MzctNmEwMi00ZjhmLWIzYmEtMTBlMzdhZjI4NzNhOjgzYjM3YzFkLWQ3MTEtNGVhYi04Y2Q0LTkwODM5ZjI4MDg1Zg==",
    verify_ssl_certs=False,
)


async def main():
    async for resp in await llm.astream_complete(
        "What is the capital of France?"
    ):
        print(resp)


asyncio.run(main())
```
