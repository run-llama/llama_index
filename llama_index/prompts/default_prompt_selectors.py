"""Prompt selectors."""
from llama_index.prompts.chat_prompts import (
    CHAT_TEXT_QA_PROMPT,
    CHAT_REFINE_PROMPT,
    CHAT_REFINE_TABLE_CONTEXT_PROMPT,
)
from llama_index.prompts.default_prompts import (
    DEFAULT_TEXT_QA_PROMPT,
    DEFAULT_REFINE_PROMPT,
    DEFAULT_REFINE_TABLE_CONTEXT_PROMPT,
)
from llama_index.prompts.prompt_selector import PromptSelector, is_chat_model
from llama_index.prompts.prompt_type import PromptType
from llama_index.prompts.prompts import (
    QuestionAnswerPrompt,
    RefinePrompt,
    RefineTableContextPrompt,
)

DEFAULT_TEXT_QA_PROMPT_SEL_LC = PromptSelector(
    default_prompt=DEFAULT_TEXT_QA_PROMPT.get_langchain_prompt(),
    conditionals=[(is_chat_model, CHAT_TEXT_QA_PROMPT.get_langchain_prompt())],
)
DEFAULT_TEXT_QA_PROMPT_SEL = QuestionAnswerPrompt(
    langchain_prompt_selector=DEFAULT_TEXT_QA_PROMPT_SEL_LC,
    prompt_type=PromptType.QUESTION_ANSWER,
)

DEFAULT_REFINE_PROMPT_SEL_LC = PromptSelector(
    default_prompt=DEFAULT_REFINE_PROMPT.get_langchain_prompt(),
    conditionals=[(is_chat_model, CHAT_REFINE_PROMPT.get_langchain_prompt())],
)
DEFAULT_REFINE_PROMPT_SEL = RefinePrompt(
    langchain_prompt_selector=DEFAULT_REFINE_PROMPT_SEL_LC,
    prompt_type=PromptType.REFINE,
)

DEFAULT_REFINE_TABLE_CONTEXT_PROMPT_SEL_LC = PromptSelector(
    default_prompt=DEFAULT_REFINE_TABLE_CONTEXT_PROMPT.get_langchain_prompt(),
    conditionals=[
        (is_chat_model, CHAT_REFINE_TABLE_CONTEXT_PROMPT.get_langchain_prompt())
    ],
)

DEFAULT_REFINE_TABLE_CONTEXT_PROMPT_SEL = RefineTableContextPrompt(
    langchain_prompt_selector=DEFAULT_REFINE_TABLE_CONTEXT_PROMPT_SEL_LC,
    prompt_type=PromptType.TABLE_CONTEXT,
)
