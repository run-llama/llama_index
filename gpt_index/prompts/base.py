"""Base module for prompts."""
from copy import deepcopy
from string import Formatter
from typing import Any, Dict, List, Optional, Type, TypeVar

from langchain import PromptTemplate as LangchainPrompt

from gpt_index.prompts.prompt_type import PromptType

PMT = TypeVar("PMT", bound="Prompt")


class Prompt:
    """Prompt class for LlamaIndex.

    Wrapper around langchain's prompt class. Adds ability to:
        - enforce certain prompt types
        - partially fill values

    """

    input_variables: List[str]
    prompt_type: PromptType = PromptType.CUSTOM

    def __init__(
        self,
        template: Optional[str] = None,
        langchain_prompt: Optional[LangchainPrompt] = None,
        **prompt_kwargs: Any,
    ) -> None:
        """Init params."""
        if langchain_prompt is None:
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

            self.prompt: LangchainPrompt = LangchainPrompt(
                input_variables=self.input_variables, template=template, **prompt_kwargs
            )
        else:
            if template:
                raise ValueError(
                    f"Both template ({template}) and langchain_prompt "
                    f"({langchain_prompt}) are provided, only one should be."
                )
            if set(langchain_prompt.input_variables) != set(self.input_variables):
                raise ValueError(
                    f"Invalid prompt: {langchain_prompt}, variables do not match the "
                    f"required input_variables: {self.input_variables}"
                )
            self.prompt = langchain_prompt
        self.partial_dict: Dict[str, Any] = {}
        self.prompt_kwargs = prompt_kwargs

    @classmethod
    def from_langchain_prompt(
        cls: Type[PMT], prompt: LangchainPrompt, **kwargs: Any
    ) -> PMT:
        """Load prompt from LangChain prompt."""
        return cls(langchain_prompt=prompt, **kwargs)

    def partial_format(self: PMT, **kwargs: Any) -> PMT:
        """Format the prompt partially.

        Return an instance of itself.

        """
        for k in kwargs.keys():
            if k not in self.input_variables:
                raise ValueError(
                    f"Invalid input variable: {k}, not found in input_variables"
                )

        copy_obj = deepcopy(self)
        copy_obj.partial_dict.update(kwargs)
        return copy_obj

    @classmethod
    def from_prompt(cls: Type[PMT], prompt: "Prompt") -> PMT:
        """Create a prompt from an existing prompt.

        Use case: If the existing prompt is already partially filled,
        and the remaining fields satisfy the requirements of the
        prompt class, then we can create a new prompt from the existing
        partially filled prompt.

        """
        template = prompt.prompt.template
        tmpl_vars = {v for _, v, _, _ in Formatter().parse(template) if v is not None}
        format_dict = {}
        for var in tmpl_vars:
            if var not in prompt.partial_dict:
                format_dict[var] = f"{{{var}}}"

        template_str = prompt.format(**format_dict)
        cls_obj: PMT = cls(template_str, **prompt.prompt_kwargs)
        return cls_obj

    def get_langchain_prompt(self) -> LangchainPrompt:
        """Get langchain prompt."""
        return self.prompt

    def format(self, **kwargs: Any) -> str:
        """Format the prompt."""
        kwargs.update(self.partial_dict)
        return self.prompt.format(**kwargs)

    def get_full_format_args(self, kwargs: Dict) -> Dict[str, Any]:
        """Get dict of all format args.

        Hack to pass into Langchain to pass validation.

        """
        kwargs.update(self.partial_dict)
        return kwargs
