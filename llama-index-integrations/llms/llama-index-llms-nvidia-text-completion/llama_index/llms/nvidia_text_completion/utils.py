from llama_index.core.bridge.pydantic import BaseModel


class Model(BaseModel):
    id: str


COMPLETION_MODEL_TABLE = {
    "bigcode/starcoder2-7b": Model(
        id="bigcode/starcoder2-7b",
        model_type="completions",
        client="NVIDIA",
    ),
    "bigcode/starcoder2-15b": Model(
        id="bigcode/starcoder2-15b",
        model_type="completions",
        client="NVIDIA",
    ),
}

completions_arguments = {
    "frequency_penalty",
    "max_tokens",
    "presence_penalty",
    "seed",
    "stop",
    "temperature",
    "top_p",
    "best_of",
    "echo",
    "logit_bias",
    "logprobs",
    "n",
    "suffix",
    "user",
    "stream",
}
