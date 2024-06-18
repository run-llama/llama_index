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


from llama_index.llms.ipex_llm import IpexLLM
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IpexLLM Basic Usage Example")
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="HuggingFaceH4/zephyr-7b-alpha",
        help="The huggingface repo id for the LLM model to be downloaded"
        ", or the path to the huggingface checkpoint folder",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        choices=["cpu", "xpu"],
        help="The device (Intel CPU or Intel GPU) the LLM model runs on",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="What is IPEX-LLM?",
        help="The sentence you prefer for query the LLM",
    )

    args = parser.parse_args()
    model_name = args.model_name
    device = args.device
    query = args.query

    llm = IpexLLM.from_model_id(
        model_name=model_name,
        tokenizer_name=model_name,
        context_window=512,
        max_new_tokens=128,
        generate_kwargs={"do_sample": False},
        completion_to_prompt=completion_to_prompt,
        messages_to_prompt=messages_to_prompt,
        device_map=device,
    )

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
