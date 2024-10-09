# LlamaIndex Llms Integration with Intel Gaudi

## Installation

```bash
pip install --upgrade-strategy eager optimum[habana]
pip install llama-index-llms-gaudi
pip install llama-index-llms-huggingface
```

## Usage

```python
import argparse
import os, logging
from llama_index.llms.gaudi import GaudiLLM


def setup_parser(parser):
    parser.add_argument(...)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GaudiLLM Basic Usage Example"
    )
    args = setup_parser(parser)
    args.model_name_or_path = "HuggingFaceH4/zephyr-7b-alpha"

    llm = GaudiLLM(
        args=args,
        logger=logger,
        model_name="HuggingFaceH4/zephyr-7b-alpha",
        tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
        query_wrapper_prompt=PromptTemplate(
            "<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"
        ),
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
```

## Examples

- [More Examples](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/llms/llama-index-llms-gaudi/examples)
