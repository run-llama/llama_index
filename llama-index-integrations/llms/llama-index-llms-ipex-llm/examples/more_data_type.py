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
        choices=["sym_int4", "asym_int4", "sym_int5", "asym_int5", "sym_int8"],
        help="The quantization type the model will convert to.",
    )

    args = parser.parse_args()
    model_name = args.model_name
    tokenizer_name = args.tokenizer_name
    low_bit = args.low_bit

    # load the model using low-bit format specified
    llm = IpexLLM.from_model_id(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        context_window=512,
        max_new_tokens=64,
        load_in_low_bit=low_bit,
        completion_to_prompt=completion_to_prompt,
        generate_kwargs={"do_sample": False},
    )

    print(
        "\n----------------------- Text Stream Completion ---------------------------"
    )
    response_iter = llm.stream_complete("Explain what is AI?")
    for response in response_iter:
        print(response.delta, end="", flush=True)
