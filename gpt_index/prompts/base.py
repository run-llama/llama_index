"""Base module for prompts."""
from copy import deepcopy
from string import Formatter
from typing import Any, Dict, List, Optional, Type, TypeVar

from langchain import BasePromptTemplate as BaseLangchainPrompt
from langchain import PromptTemplate as LangchainPrompt
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.schema import BaseLanguageModel

from gpt_index.output_parsers.base import BaseOutputParser
from gpt_index.prompts.prompt_type import PromptType

PMT = TypeVar("PMT", bound="Prompt")


class Prompt:
    """Prompt class for LlamaIndex.

    Wrapper around langchain's prompt class. Adds ability to:
        - enforce certain prompt types
        - partially fill values
        - define stop token

    """

    input_variables: List[str]
    prompt_type: str = PromptType.CUSTOM

    def __init__(
        self,
        template: Optional[str] = None,
        langchain_prompt: Optional[BaseLangchainPrompt] = None,
        langchain_prompt_selector: Optional[ConditionalPromptSelector] = None,
        stop_token: Optional[str] = None,
        output_parser: Optional[BaseOutputParser] = None,
        **prompt_kwargs: Any,
    ) -> None:
        """Init params."""
        # first check if langchain_prompt_selector is provided
        # TODO: self.prompt is deprecated, switch to prompt_selector under the hood
        if langchain_prompt_selector is not None:
            self.prompt_selector = langchain_prompt_selector
            self.prompt: BaseLangchainPrompt = self.prompt_selector.default_prompt
        # then check if template is provided
        elif langchain_prompt is None:
            if template is None:
                raise ValueError(
                    "`template` must be specified if `langchain_prompt` is None"
                )
            # validate
            tmpl_vars = {
                v for _, v, _, _ in Formatter().parse(template) if v is not None
            }
            if tmpl_vars != set(self.input_variables):
                raise ValueError(
                    f"Invalid template: {template}, variables do not match the "
                    f"required input_variables: {self.input_variables}"
                )

            self.prompt = LangchainPrompt(
                input_variables=self.input_variables, template=template, **prompt_kwargs
            )
            self.prompt_selector = ConditionalPromptSelector(default_prompt=self.prompt)
        # finally, check if langchain_prompt is provided
        else:
            if template:
                raise ValueError(
                    f"Both template ({template}) and langchain_prompt "
                    f"({langchain_prompt}) are provided, only one should be."
                )
            self.prompt = langchain_prompt
            self.prompt_selector = ConditionalPromptSelector(default_prompt=self.prompt)

        # validate all prompts in prompt selector
        all_lc_prompts = [self.prompt_selector.default_prompt]
        for _, prompt in self.prompt_selector.conditionals:
            all_lc_prompts.append(prompt)
        for lc_prompt in all_lc_prompts:
            if set(lc_prompt.input_variables) != set(self.input_variables):
                raise ValueError(
                    f"Invalid prompt: {langchain_prompt}, variables do not match the "
                    f"required input_variables: {self.input_variables}"
                )
        self.partial_dict: Dict[str, Any] = {}
        self.prompt_kwargs = prompt_kwargs
        self.stop_token = stop_token

        self.output_parser = output_parser

    @classmethod
    def from_langchain_prompt(
        cls: Type[PMT], prompt: BaseLangchainPrompt, **kwargs: Any
    ) -> PMT:
        """Load prompt from LangChain prompt."""
        return cls(langchain_prompt=prompt, **kwargs)

    @classmethod
    def from_langchain_prompt_selector(
        cls: Type[PMT], prompt_selector: ConditionalPromptSelector, **kwargs: Any
    ) -> PMT:
        """Load prompt from LangChain prompt."""
        return cls(langchain_prompt_selector=prompt_selector, **kwargs)

    def partial_format(self: PMT, **kwargs: Any) -> PMT:
        """Format the prompt partially.

        Return an instance of itself.

        """
        for k in kwargs.keys():
            if k not in self.input_variables:
                raise ValueError(
                    f"Invalid input variable: {k}, not found in input_variables"
                )
        try:
            # NOTE: this is a hack to get around deepcopy failing on output parser
            output_parser = self.output_parser
            self.output_parser = None

            copy_obj = deepcopy(self)
            copy_obj.output_parser = output_parser
            copy_obj.partial_dict.update(kwargs)
            self.output_parser = output_parser
        except Exception as e:
            raise e

        return copy_obj

    @classmethod
    def from_prompt(
        cls: Type[PMT], prompt: "Prompt", llm: Optional[BaseLanguageModel] = None
    ) -> PMT:
        """Create a prompt from an existing prompt.

        Use case: If the existing prompt is already partially filled,
        and the remaining fields satisfy the requirements of the
        prompt class, then we can create a new prompt from the existing
        partially filled prompt.

        """
        lc_prompt = prompt.get_langchain_prompt(llm=llm)
        tmpl_vars = lc_prompt.input_variables
        format_dict = {}
        for var in tmpl_vars:
            if var not in prompt.partial_dict:
                format_dict[var] = f"{{{var}}}"

        template_str = prompt.format(llm=llm, **format_dict)
        cls_obj: PMT = cls(template_str, **prompt.prompt_kwargs)
        return cls_obj

    def get_langchain_prompt(
        self, llm: Optional[BaseLanguageModel] = None
    ) -> BaseLangchainPrompt:
        """Get langchain prompt."""
        if llm is None:
            return self.prompt_selector.default_prompt
        return self.prompt_selector.get_prompt(llm=llm)

    def format(self, llm: Optional[BaseLanguageModel] = None, **kwargs: Any) -> str:
        """Format the prompt."""
        kwargs.update(self.partial_dict)
        lc_prompt = self.get_langchain_prompt(llm=llm)
        return lc_prompt.format(**kwargs)

    def get_full_format_args(self, kwargs: Dict) -> Dict[str, Any]:
        """Get dict of all format args.

        Hack to pass into Langchain to pass validation.

        """
        kwargs.update(self.partial_dict)
        if self.stop_token is not None:
            kwargs["stop"] = self.stop_token
        return kwargs
