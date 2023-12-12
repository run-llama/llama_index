"""Question gen."""

from llama_index.question_gen.openai_generator import OpenAIQuestionGenerator
from llama_index.question_gen.llm_generators import LLMQuestionGenerator

__all__ = ["OpenAIQuestionGenerator", "LLMQuestionGenerator"]