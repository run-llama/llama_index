from typing import Dict, Sequence, List

from llama_index.core.base.llms.types import ChatMessage

from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message import ChatCompletionMessage

AI_PLAYGROUND_MODELS: Dict[str, int] = {
    "playground_llama2_13b": 4096,
    "playground_llama2_code_13b": 100_000,
    "playground_llama2_70b": 4096,
    "playground_mistral_7b": 8192,
    "playground_nv_llama2_rlhf_70b": 4096,
    "playground_yi_34b": 200_000,
    "playground_nemotron_steerlm_8b": 3072,
    "playground_mixtral_8x7b": 32_000,
    "playground_llama2_code_34b": 100_000,
    "playground_steerlm_llama_70b": 3072,
}

def playground_modelname_to_contextsize(modelname: str) -> int:
    if modelname not in AI_PLAYGROUND_MODELS:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid AI Playground model name."
            "Known models are: " + ", ".join(AI_PLAYGROUND_MODELS.keys())
        )

    return AI_PLAYGROUND_MODELS[modelname]
