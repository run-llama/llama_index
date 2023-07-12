from typing import Dict, Type

from llama_index.agent import OpenAIAgent, ReActAgent
from llama_index.agent.types import BaseAgent
from llama_index.llms import Anthropic, OpenAI
from llama_index.llms.base import LLM

OPENAI_MODELS = [
    "text-davinci-003",
    "gpt-3.5-turbo-0613",
    "gpt-4-0613",
]
ANTHROPIC_MODELS = ["claude-instant-1", "claude-2"]
ALL_MODELS = OPENAI_MODELS + ANTHROPIC_MODELS

AGENTS: Dict[str, Type[BaseAgent]] = {
    "react": ReActAgent,
    "openai": OpenAIAgent,
}


def get_model(model: str) -> LLM:
    llm: LLM
    if model in OPENAI_MODELS:
        llm = OpenAI(model=model)
    elif model in ANTHROPIC_MODELS:
        llm = Anthropic(model=model)
    else:
        raise ValueError(f"Unknown model {model}")
    return llm


def is_valid_combination(agent: str, model: str) -> bool:
    if agent == "openai" and model not in ["gpt-3.5-turbo-0613", "gpt-4-0613"]:
        print(f"{agent} does not work with {model}")
        return False
    return True
