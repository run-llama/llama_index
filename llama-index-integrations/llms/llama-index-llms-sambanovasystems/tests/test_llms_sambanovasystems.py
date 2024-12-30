import asyncio
import time
import os
import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)
from llama_index.core.llms.llm import LLM
from llama_index.llms.sambanovasystems import SambaNovaCloud, SambaStudio


sambanova_api_key = os.environ.get("SAMBANOVA_API_KEY", None)
sambastudio_url = os.environ.get("SAMBASTUDIO_URL", None)
sambastudio_api_key = os.environ.get("SAMBASTUDIO_API_KEY", None)


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


@pytest.mark.skipif(not sambanova_api_key, reason="No api key set")
def test_calls(sambanova_client: LLM):
    # chat interaction example
    user_message = ChatMessage(
        role=MessageRole.USER, content="Tell me about Naruto Uzumaki in one sentence"
    )
    chat_text_msgs = [
        ChatMessage(role=MessageRole.SYSTEM, content=("You're a helpful assistant")),
        user_message,
    ]

    # sync
    print(f"chat response: {sambanova_client.chat(chat_text_msgs)}\n")
    print(
        f"stream chat response: {[x.message.content for x in sambanova_client.stream_chat(chat_text_msgs)]}\n"
    )

    print(f"complete response: {sambanova_client.complete(user_message.content)}\n")
    print(
        f"stream complete response: {[x.text for x in sambanova_client.stream_complete(user_message.content)]}\n"
    )

    # async
    print(
        f"async chat response: {asyncio.run(sambanova_client.achat(chat_text_msgs))}\n"
    )
    print(
        f"async complete response: {asyncio.run(sambanova_client.acomplete(user_message.content))}\n"
    )


@pytest.mark.skipif(not sambanova_api_key, reason="No api key set")
def test_performance(sambanova_client: LLM):
    # chat interaction example
    user_message = ChatMessage(
        role=MessageRole.USER, content="Tell me about Naruto Uzumaki in one sentence"
    )
    chat_text_msgs = [
        ChatMessage(role=MessageRole.SYSTEM, content=("You're a helpful assistant")),
        user_message,
    ]

    # chat
    get_execution_time(sambanova_client.chat, chat_text_msgs, number=5)
    get_execution_time(sambanova_client.achat, chat_text_msgs, is_async=True, number=5)

    # complete
    get_execution_time(sambanova_client.complete, user_message.content, number=5)
    get_execution_time(
        sambanova_client.acomplete, user_message.content, is_async=True, number=5
    )


@pytest.mark.skipif(not sambanova_api_key, reason="No api key set")
def test_hiperparameters(sambanova_cls: LLM, testing_model: str):
    user_message = ChatMessage(
        role=MessageRole.USER, content="Tell me about Naruto Uzumaki in one sentence"
    )
    chat_text_msgs = [
        ChatMessage(role=MessageRole.SYSTEM, content=("You're a helpful assistant")),
        user_message,
    ]

    max_tokens_list = [10, 100]
    temperature_list = [0, 1]
    top_p_list = [0, 1]
    top_k_list = [1, 50]
    stream_options_list = [{"include_usage": False}, {"include_usage": True}]

    sambanovacloud_client = sambanova_cls(
        model=testing_model,
        max_tokens=max_tokens_list[0],
        temperature=temperature_list[0],
        top_p=top_p_list[0],
        top_k=top_k_list[0],
        stream_options=stream_options_list[0],
    )
    print(
        f"model: {testing_model}, generation: {sambanovacloud_client.chat(chat_text_msgs)}"
    )

    for max_tokens in max_tokens_list:
        sambanovacloud_client = sambanova_cls(
            model=testing_model,
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
        sambanovacloud_client = sambanova_cls(
            model=testing_model,
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
        sambanovacloud_client = sambanova_cls(
            model=testing_model,
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
        sambanovacloud_client = sambanova_cls(
            model=testing_model,
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
        sambanovacloud_client = sambanova_cls(
            model=testing_model,
            max_tokens=max_tokens_list[0],
            temperature=temperature_list[0],
            top_p=top_p_list[0],
            top_k=top_k_list[0],
            stream_options=stream_options,
        )
        print(
            f"stream_options: {stream_options}, generation: {sambanovacloud_client.chat(chat_text_msgs)}"
        )


@pytest.mark.skipif(not sambanova_api_key, reason="No api key set")
def test_sambanovacloud():
    testing_model = "llama3-8b"
    sambanova_client = SambaNovaCloud()
    test_calls(sambanova_client)
    test_performance(sambanova_client)
    test_hiperparameters(SambaNovaCloud, testing_model)


@pytest.mark.skipif(not sambastudio_api_key, reason="No api key set")
def test_sambastudio():
    testing_model = "Meta-Llama-3-70B-Instruct-4096"
    sambanova_client = SambaStudio(model=testing_model)
    test_calls(sambanova_client)
    test_performance(sambanova_client)
    test_hiperparameters(SambaStudio, testing_model)


def test_init():
    _ = SambaNovaCloud(sambanova_api_key="fake")
    _ = SambaStudio(sambastudio_url="fake/stream/url", sambastudio_api_key="fake")


if __name__ == "__main__":
    test_sambanovacloud()
    test_sambastudio()
