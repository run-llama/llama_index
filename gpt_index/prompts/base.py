"""Base module for prompts."""
from copy import deepcopy
from string import Formatter
from typing import Any, Dict, List, Type, TypeVar

from langchain import Prompt as LangchainPrompt

from gpt_index.prompts.prompt_type import PromptType

PMT = TypeVar("PMT", bound="Prompt")


class Prompt:
    """Prompt class for GPT Index.

    Wrapper around langchain's prompt class. Adds ability to:
        - enforce certain prompt types
        - partially fill values

    """

    input_variables: List[str]
    prompt_type: PromptType = PromptType.CUSTOM

    def __init__(self, template: str, **prompt_kwargs: Any) -> None:
        """Init params."""
        # validate
        tmpl_vars = {v for _, v, _, _ in Formatter().parse(template) if v is not None}
        if tmpl_vars != set(self.input_variables):
            raise ValueError(
                f"Invalid template: {template}, variables do not match the "
                f"required input_variables: {self.input_variables}"
            )

        self.prompt: LangchainPrompt = LangchainPrompt(
            input_variables=self.input_variables, template=template, **prompt_kwargs
        )
        self.partial_dict: Dict[str, Any] = {}
        self.prompt_kwargs = prompt_kwargs

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
