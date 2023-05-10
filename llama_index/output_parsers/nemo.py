"""NeMo Hallucincation output parser.

See https://github.com/NVIDIA/NeMo-Guardrails.

"""
try:
    from nemoguardrails.actions.hallucination import check_hallucination
    from nemoguardrails.actions.fact_checking import check_facts
except ImportError:
    check_hallucination = None

import asyncio
from typing import Any, Optional
from langchain.llms import OpenAI

from llama_index.output_parsers.base import BaseOutputParser

DEFAULT_HALLUCINATION_RESPONSE = (
    "This response may have been hallucincated, "
    "and should be validated and fact-checked by an expert."
)

DEFAULT_FACT_CHECK_RESPONSE = (
    "This answer is not supported by evidence found in the retrieved "
    "context, and should be validated and fact-checked by an expert."
)


class NeMoGaurdrailsOutputParser(BaseOutputParser):
    """NeMo Guardrails output parser.

    Args:
        llm (Optional[OpenAI]):
            The LLM to use for hallucination checking.
            Currently NeMo only supports OpenAI LLMs.
        check_hallucination (bool):
            Flag for whether to check for hallucination or not.
            Defaults to True.
        check_facts (bool):
            Flag for whether to check that the response is supported by the context.
            Defaults to True.
        remove_failed_facts (bool):
            Flag to remove responses that fail the fact check. Defaults to False.
        hallucination_str (str):
            String to indicate a hallucinated response. Is appened to normal output.
        grounded_str (str):
            String to indicate an un-grounded response. Is appened to normal output.
    """

    def __init__(
        self,
        llm: Optional[OpenAI] = None,
        check_hallucination: bool = True,
        check_facts: bool = True,
        remove_failed_facts: bool = False,
        hallucination_str: str = DEFAULT_HALLUCINATION_RESPONSE,
        fact_check_str: str = DEFAULT_FACT_CHECK_RESPONSE,
    ) -> None:
        self.llm = llm or OpenAI(model_name="text-davinci-003", temperature=0.0)

        self.check_hallucination = check_hallucination
        self.hallucination_str = hallucination_str

        self.check_facts = check_facts
        self.fact_check_str = fact_check_str
        self.remove_failed_facts = remove_failed_facts

    def parse(self, output: str, formatted_prompt: str) -> Any:
        """Parse, validate, and correct errors programmatically."""
        # TODO: super hacky, since NeMo functions are mostly async
        new_output = output
        if self.check_hallucination:
            hallucination_context = {
                "last_bot_message": output,
                "_last_bot_prompt": formatted_prompt,
            }
            is_hallucinating = asyncio.run(
                check_hallucination(hallucination_context, llm=self.llm)
            )
            if is_hallucinating:
                new_output = f"{new_output}\n{self.hallucination_str}"

        if self.check_facts:
            facts_context = {
                "last_bot_message": output,
                "relevant_chunks": formatted_prompt,
            }
            is_contradiction = asyncio.run(check_facts(facts_context, llm=self.llm))
            if is_contradiction is True and self.remove_failed_facts:
                new_output = ""
            elif is_contradiction is True:
                new_output = f"{new_output}\n{self.fact_check_str}"

        return new_output

    def format(self, query: str) -> str:
        """Unused for NeMoGaurdrailsOutputParser."""
        return query
