import time
import asyncio
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llama_index.llms.sambanovacloud import SambaNovaCloud
import pytest


sambanova_api_key = os.environ.get("SAMBANOVA_API_KEY", None)


@pytest.mark.asyncio()
async def run_async_test(fn, chat_msgs, number, verbose=False):
    tasks = [fn(chat_msgs) for _ in range(number)]
    if verbose:
        for task in asyncio.as_completed(tasks):
            result = await task  # Wait for the next completed task
            print(result)
    else:
        await asyncio.gather(*tasks)


def run_sync_test(fn, chat_msgs, number):
    for _ in range(number):
        fn(chat_msgs)


def get_execution_time(fn, chat_msgs, is_async=False, number=10):
    start_time = time.perf_counter()
    if is_async:
        asyncio.run(run_async_test(fn, chat_msgs, number))
    else:
        run_sync_test(fn, chat_msgs, number)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(
        f"Execution time {'for async test' if is_async else ''}: {execution_time:.6f} seconds"
    )


@pytest.mark.skipif(not sambanova_api_key, reason="No openai api key set")
def test_sambanovacloud():
    # chat interaction example
    user_message = ChatMessage(
        role=MessageRole.USER, content="Tell me about Naruto Uzumaki in one sentence"
    )
    chat_text_msgs = [
        ChatMessage(role=MessageRole.SYSTEM, content=("You're a helpful assistant")),
        user_message,
    ]

    sambanovacloud_client = SambaNovaCloud()

    # sync
    print(f"chat response: {sambanovacloud_client.chat(chat_text_msgs)}\n")
    print(
        f"stream chat response: {[x.message.content for x in sambanovacloud_client.stream_chat(chat_text_msgs)]}\n"
    )

    print(
        f"complete response: {sambanovacloud_client.complete(user_message.content)}\n"
    )
    print(
        f"stream complete response: {[x.text for x in sambanovacloud_client.stream_complete(user_message.content)]}\n"
    )

    # async
    print(
        f"async chat response: {asyncio.run(sambanovacloud_client.achat(chat_text_msgs))}\n"
    )
    print(
        f"async complete response: {asyncio.run(sambanovacloud_client.acomplete(user_message.content))}\n"
    )


@pytest.mark.skipif(not sambanova_api_key, reason="No openai api key set")
def test_sambanovacloud_performance():
    # chat interaction example
    user_message = ChatMessage(
        role=MessageRole.USER, content="Tell me about Naruto Uzumaki in one sentence"
    )
    chat_text_msgs = [
        ChatMessage(role=MessageRole.SYSTEM, content=("You're a helpful assistant")),
        user_message,
    ]

    sambanovacloud_client = SambaNovaCloud()

    # chat
    get_execution_time(sambanovacloud_client.chat, chat_text_msgs, number=5)
    get_execution_time(
        sambanovacloud_client.achat, chat_text_msgs, is_async=True, number=5
    )

    # complete
    get_execution_time(sambanovacloud_client.complete, user_message.content, number=5)
    get_execution_time(
        sambanovacloud_client.acomplete, user_message.content, is_async=True, number=5
    )


@pytest.mark.skipif(not sambanova_api_key, reason="No openai api key set")
def test_hiperparameters():
    user_message = ChatMessage(
        role=MessageRole.USER, content="Tell me about Naruto Uzumaki in one sentence"
    )
    chat_text_msgs = [
        ChatMessage(role=MessageRole.SYSTEM, content=("You're a helpful assistant")),
        user_message,
    ]

    model_list = ["llama3-8b", "llama3-70b"]
    max_tokens_list = [10, 100]
    temperature_list = [0, 1]
    top_p_list = [0, 1]
    top_k_list = [1, 50]
    stream_options_list = [{"include_usage": False}, {"include_usage": True}]

    for model in model_list:
        sambanovacloud_client = SambaNovaCloud(
            model=model,
            max_tokens=max_tokens_list[0],
            temperature=temperature_list[0],
            top_p=top_p_list[0],
            top_k=top_k_list[0],
            stream_options=stream_options_list[0],
        )
        print(
            f"model: {model}, generation: {sambanovacloud_client.chat(chat_text_msgs)}"
        )

    for max_tokens in max_tokens_list:
        sambanovacloud_client = SambaNovaCloud(
            model=model_list[0],
            max_tokens=max_tokens,
            temperature=temperature_list[0],
            top_p=top_p_list[0],
            top_k=top_k_list[0],
            stream_options=stream_options_list[0],
        )
        print(
            f"max_tokens: {max_tokens}, generation: {sambanovacloud_client.chat(chat_text_msgs)}"
        )

    for temperature in temperature_list:
        sambanovacloud_client = SambaNovaCloud(
            model=model_list[0],
            max_tokens=max_tokens_list[0],
            temperature=temperature,
            top_p=top_p_list[0],
            top_k=top_k_list[0],
            stream_options=stream_options_list[0],
        )
        print(
            f"temperature: {temperature}, generation: {sambanovacloud_client.chat(chat_text_msgs)}"
        )

    for top_p in top_p_list:
        sambanovacloud_client = SambaNovaCloud(
            model=model_list[0],
            max_tokens=max_tokens_list[0],
            temperature=temperature_list[0],
            top_p=top_p,
            top_k=top_k_list[0],
            stream_options=stream_options_list[0],
        )
        print(
            f"top_p: {top_p}, generation: {sambanovacloud_client.chat(chat_text_msgs)}"
        )

    for top_k in top_k_list:
        sambanovacloud_client = SambaNovaCloud(
            model=model_list[0],
            max_tokens=max_tokens_list[0],
            temperature=temperature_list[0],
            top_p=top_p_list[0],
            top_k=top_k,
            stream_options=stream_options_list[0],
        )
        print(
            f"top_k: {top_k}, generation: {sambanovacloud_client.chat(chat_text_msgs)}"
        )

    for stream_options in stream_options_list:
        sambanovacloud_client = SambaNovaCloud(
            model=model_list[0],
            max_tokens=max_tokens_list[0],
            temperature=temperature_list[0],
            top_p=top_p_list[0],
            top_k=top_k_list[0],
            stream_options=stream_options,
        )
        print(
            f"stream_options: {stream_options}, generation: {sambanovacloud_client.chat(chat_text_msgs)}"
        )


if __name__ == "__main__":
    test_sambanovacloud()
    test_sambanovacloud_performance()
    test_hiperparameters()
