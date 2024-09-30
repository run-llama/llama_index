# Transform a string into input zephyr-specific input
def completion_to_prompt(completion):
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"


# Transform a list of chat messages into zephyr-specific input
def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}</s>\n"

    # ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    # add final assistant prompt
    prompt = prompt + "<|assistant|>\n"

    return prompt


import logging
import argparse
from llama_index.llms.gaudi import GaudiLLM
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.gaudi.utils import (
    setup_parser,
)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GaudiLLM Basic Usage Example")
    args = setup_parser(parser)
    args.model_name_or_path = "HuggingFaceH4/zephyr-7b-alpha"

    llm = GaudiLLM(
        args=args,
        logger=logger,
        model_name="HuggingFaceH4/zephyr-7b-alpha",
        tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
        query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
        context_window=3900,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        messages_to_prompt=messages_to_prompt,
        device_map="auto",
    )

    query = "Is the ocean blue?"
    print("\n----------------- Complete ------------------")
    completion_response = llm.complete(query)
    print(completion_response.text)
    print("\n----------------- Stream Complete ------------------")
    response_iter = llm.stream_complete(query)
    for response in response_iter:
        print(response.delta, end="", flush=True)
    print("\n----------------- Chat ------------------")
    from llama_index.core.llms import ChatMessage

    message = ChatMessage(role="user", content=query)
    resp = llm.chat([message])
    print(resp)
    print("\n----------------- Stream Chat ------------------")
    message = ChatMessage(role="user", content=query)
    resp = llm.stream_chat([message], max_tokens=256)
    for r in resp:
        print(r.delta, end="")
