"""Wrapper functions around an LLM chain."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict

import openai
from langchain import Cohere, LLMChain, OpenAI
from langchain.llms import AI21
from langchain.llms.base import BaseLLM

from gpt_index.constants import MAX_CHUNK_SIZE, NUM_OUTPUTS
from gpt_index.prompts.base import Prompt
from gpt_index.utils import globals_helper, retry_on_exceptions_with_backoff

# for chat history support
from typing import List

from gpt_index.prompts.base import Prompt
from gpt_index.prompts.prompt_type import PromptType

class QuestionAnswerWithHistoryPrompt(Prompt):
    """Question Answer prompt.

    Prompt to answer a question `query_str` given a context `context_str`.

    Required template variables: `context_str`, `query_str`

    Args:
        template (str): Template for the prompt.
        **prompt_kwargs: Keyword arguments for the prompt.

    """

    prompt_type: PromptType = PromptType.QUESTION_ANSWER
    input_variables: List[str] = ["context_str", "history", "query_str"]

DEFAULT_TEXT_QA_PROMPT_WITH_HISTORY_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "chat history.\n"
    "---------------------\n"
    "{history}"
    "\n---------------------\n"
    "Given the context information or chat history and not prior knowledge, "
    "answer the question as honest and faithful personal counselor: {query_str}\n"
    "if context information or chat history don't have answer, you may use prior knowledge.\n"
)
DEFAULT_TEXT_QA_WITH_HISTORY_PROMPT = QuestionAnswerWithHistoryPrompt(DEFAULT_TEXT_QA_PROMPT_WITH_HISTORY_TMPL)


@dataclass
class LLMMetadata:
    """LLM metadata.

    We extract this metadata to help with our prompts.

    """

    max_input_size: int = MAX_CHUNK_SIZE
    num_output: int = NUM_OUTPUTS


def _get_llm_metadata(llm: BaseLLM) -> LLMMetadata:
    """Get LLM metadata from llm."""
    if not isinstance(llm, BaseLLM):
        raise ValueError("llm must be an instance of langchain.llms.base.LLM")
    if isinstance(llm, OpenAI):
        return LLMMetadata(
            max_input_size=llm.modelname_to_contextsize(llm.model_name),
            num_output=llm.max_tokens,
        )
    elif isinstance(llm, Cohere):
        # TODO: figure out max input size for cohere
        return LLMMetadata(num_output=llm.max_tokens)
    elif isinstance(llm, AI21):
        # TODO: figure out max input size for AI21
        return LLMMetadata(num_output=llm.maxTokens)
    else:
        return LLMMetadata()


class LLMPredictor:
    """LLM predictor class.

    Wrapper around an LLMChain from Langchain.

    Args:
        llm (Optional[langchain.llms.base.LLM]): LLM from Langchain to use
            for predictions. Defaults to OpenAI's text-davinci-003 model.
            Please see `Langchain's LLM Page
            <https://langchain.readthedocs.io/en/latest/modules/llms.html>`_
            for more details.

        retry_on_throttling (bool): Whether to retry on rate limit errors.
            Defaults to true.

    """

    def __init__(
        self, llm: Optional[BaseLLM] = None, retry_on_throttling: bool = True, chat_history: int = 0
    ) -> None:
        """Initialize params."""
        self._llm = llm or OpenAI(temperature=0, model_name="text-davinci-003")
        self.retry_on_throttling = retry_on_throttling
        self._total_tokens_used = 0
        self.flag = True
        self._last_token_usage: Optional[int] = None
        self._chat_history = chat_history
        if chat_history > 0:
            self.history = []
        
    def get_llm_metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # TODO: refactor mocks in unit tests, this is a stopgap solution
        if hasattr(self, "_llm"):
            return _get_llm_metadata(self._llm)
        else:
            return LLMMetadata()

    def _predict(self, prompt: Prompt, **prompt_args: Any) -> str:
        """Inner predict function.

        If retry_on_throttling is true, we will retry on rate limit errors.

        """
        if self._chat_history:
            print(f'===prompt===\n{prompt}')
            print(f'===prompt_args===\n{prompt_args}')
            prompt_with_history = DEFAULT_TEXT_QA_WITH_HISTORY_PROMPT   
            llm_chain = LLMChain(prompt=prompt_with_history.get_langchain_prompt(), llm=self._llm, verbose=True) 
        else:
            llm_chain = LLMChain(prompt=prompt.get_langchain_prompt(), llm=self._llm) 

        # Note: we don't pass formatted_prompt to llm_chain.predict because
        # langchain does the same formatting under the hood
        full_prompt_args = prompt.get_full_format_args(prompt_args)

        if self._chat_history > 0:  
            history_text = ""
            for ht in self.history:
                history_text += f"{ht}\n"
            partial_dict: Dict[str, Any] = {"history": history_text}
            full_prompt_args.update(partial_dict)

        if self.retry_on_throttling:
            llm_prediction = retry_on_exceptions_with_backoff(
                lambda: llm_chain.predict(**full_prompt_args),
                [openai.error.RateLimitError],
            )
        else:
            llm_prediction = llm_chain.predict(**full_prompt_args)

        if self._chat_history > 0:
            q = full_prompt_args["query_str"]
            a = llm_prediction.strip()
            self.history.append(f"Customer:{q}\nCounselor:{a}")
            while len(self.history) > self._chat_history:
                self.history.pop(0)
        return llm_prediction

    def predict(self, prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
        """Predict the answer to a query.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

        """
        formatted_prompt = prompt.format(**prompt_args)
        llm_prediction = self._predict(prompt, **prompt_args)

        # We assume that the value of formatted_prompt is exactly the thing
        # eventually sent to OpenAI, or whatever LLM downstream
        prompt_tokens_count = self._count_tokens(formatted_prompt)
        prediction_tokens_count = self._count_tokens(llm_prediction)
        self._total_tokens_used += prompt_tokens_count + prediction_tokens_count
        return llm_prediction, formatted_prompt

    @property
    def total_tokens_used(self) -> int:
        """Get the total tokens used so far."""
        return self._total_tokens_used

    def _count_tokens(self, text: str) -> int:
        tokens = globals_helper.tokenizer(text)
        return len(tokens)

    @property
    def last_token_usage(self) -> int:
        """Get the last token usage."""
        if self._last_token_usage is None:
            return 0
        return self._last_token_usage

    @last_token_usage.setter
    def last_token_usage(self, value: int) -> None:
        """Set the last token usage."""
        self._last_token_usage = value
