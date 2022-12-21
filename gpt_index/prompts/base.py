"""Base module for prompts."""
from copy import deepcopy
from typing import Any, Dict, List, Optional

from langchain import Prompt as LangchainPrompt
from pydantic import Field

from gpt_index.prompts.prompt_type import PromptType


class Prompt(LangchainPrompt):
    """Prompt class for GPT Index.

    Extends langchain's prompt, but also adds ability to partially fill values.

    """

    prompt_type: PromptType = PromptType.SUMMARY
    partial_dict: Dict[str, Any] = Field(default_factory=dict)

    def partial_format(self, **kwargs: Any) -> "Prompt":
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

    def format(self, **kwargs: Any) -> str:
        """Format the prompt."""
        kwargs.update(self.partial_dict)
        return super().format(**kwargs)

    def get_full_format_args(self, kwargs: Any) -> Dict[str, Any]:
        """Get dict of all format args.

        Hack to pass into Langchain to pass validation.

        """
        kwargs.update(self.partial_dict)
        return kwargs


def validate_prompt(
    prompt: Prompt,
    required_fields: List[str],
    optional_fields: Optional[List[str]] = None,
) -> None:
    """Validate prompts."""
    # make sure all required fields are in input_variables
    for req_field in required_fields:
        if req_field not in prompt.input_variables:
            raise ValueError(f"`{req_field}` must be provided in prompt.")

    optional_fields = optional_fields or []
    valid_fields = required_fields + optional_fields

    # make sure all input_variables are either required or optional
    for input_var in prompt.input_variables:
        if input_var not in valid_fields:
            raise ValueError(
                f"`{input_var}` is not a valid input variable for this prompt."
                f"Set of valid fields: {valid_fields}."
            )
