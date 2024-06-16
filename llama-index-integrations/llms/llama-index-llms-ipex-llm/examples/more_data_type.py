import argparse
from llama_index.llms.ipex_llm import IpexLLM


# Transform a string into input llama2-specific input
def completion_to_prompt(completion):
    return f"<s>[INST] <<SYS>>\n    \n<</SYS>>\n\n{completion} [/INST]"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="More Data Types Example")
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="The huggingface repo id for the large language model to be downloaded"
        ", or the path to the huggingface checkpoint folder",
    )
    parser.add_argument(
        "--tokenizer-name",
        "-t",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="The huggingface repo id or the path to the checkpoint containing the tokenizer"
        "usually it is the same as the model_name",
    )
    parser.add_argument(
        "--low-bit",
        "-l",
        type=str,
        default="asym_int4",
        choices=[
            "sym_int4",
            "asym_int4",
            "sym_int5",
            "asym_int5",
            "sym_int8",
            "fp4",  # only available on GPU
            "fp8",  # only available on GPU
            "fp16",  # only available on GPU
            "bf16",  # only available on GPU
            "fp8_e4m3",  # only available on GPU
            "fp8_e5m2",  # only available on GPU
            "nf3",  # only available on GPU
            "nf4",  # only available on GPU
        ],
        help="The quantization type the model will convert to.",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="xpu",
        choices=["cpu", "xpu"],
        help="The device the model will run on.",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="What is AI?",
        help="The sentence you prefer for query the LLM",
    )

    args = parser.parse_args()
    model_name = args.model_name
    tokenizer_name = args.tokenizer_name
    low_bit = args.low_bit
    device = args.device
    query = args.query

    # load the model using low-bit format specified
    llm = IpexLLM.from_model_id(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        context_window=512,
        max_new_tokens=64,
        load_in_low_bit=low_bit,
        completion_to_prompt=completion_to_prompt,
        generate_kwargs={"do_sample": False},
        device_map=device,
    )

    print(
        "\n----------------------- Text Stream Completion ---------------------------"
    )
    response_iter = llm.stream_complete(query)
    for response in response_iter:
        print(response.delta, end="", flush=True)
