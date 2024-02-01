from typing import Dict, Type

from llama_index.core.llms.custom import CustomLLM
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.mock import MockLLM

RECOGNIZED_LLMS: Dict[str, Type[LLM]] = {
    MockLLM.class_name(): MockLLM,
    CustomLLM.class_name(): CustomLLM,
}


def load_llm(data: dict) -> LLM:
    """Load LLM by name."""
    if isinstance(data, LLM):
        return data
    llm_name = data.get("class_name", None)
    if llm_name is None:
        raise ValueError("LLM loading requires a class_name")

    if llm_name not in RECOGNIZED_LLMS:
        raise ValueError(f"Invalid LLM name: {llm_name}")

    return RECOGNIZED_LLMS[llm_name].from_dict(data)
