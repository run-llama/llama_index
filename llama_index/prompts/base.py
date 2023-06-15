"""Base module for prompts."""
from copy import deepcopy
from typing import Any, Dict, Optional

from langchain import BasePromptTemplate as BaseLangchainPrompt
from langchain import PromptTemplate as LangchainPrompt
from langchain.base_language import BaseLanguageModel
from langchain.chains.prompt_selector import ConditionalPromptSelector

from llama_index.output_parsers.base import BaseOutputParser
from llama_index.prompts.prompt_type import PromptType


class Prompt:
    """Prompt class for LlamaIndex.

    Wrapper around langchain's prompt class. Adds ability to:
        - enforce certain prompt types
        - partially fill values
        - define stop token

    """

    def __init__(
        self,
        template: Optional[str] = None,
        langchain_prompt: Optional[BaseLangchainPrompt] = None,
        langchain_prompt_selector: Optional[ConditionalPromptSelector] = None,
        stop_token: Optional[str] = None,
        output_parser: Optional[BaseOutputParser] = None,
        prompt_type: str = PromptType.CUSTOM,
        metadata: Optional[Dict[str, Any]] = None,
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

            self.prompt = LangchainPrompt.from_template(
                template=template, **prompt_kwargs
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

        self.partial_dict: Dict[str, Any] = {}
        self.prompt_kwargs = prompt_kwargs
        self.stop_token = stop_token
        # NOTE: this is only used for token counting and testing
        self.prompt_type = prompt_type

        self.output_parser = output_parser

        self._original_template = template

        # Metadata is used to pass arbitrary information to other consumers of the
        # prompt. For example, VellumPromptRegistry uses this to access vellum-specific
        # identifiers that users can pass along with the prompt.
        self.metadata = metadata or {}

    @property
    def original_template(self) -> str:
        """Return the originally specified template, if supplied."""

        if not self._original_template:
            raise ValueError("No original template specified.")

        return self._original_template

    @classmethod
    def from_langchain_prompt(
        cls, prompt: BaseLangchainPrompt, **kwargs: Any
    ) -> "Prompt":
        """Load prompt from LangChain prompt."""
        return cls(langchain_prompt=prompt, **kwargs)

    @classmethod
    def from_langchain_prompt_selector(
        cls, prompt_selector: ConditionalPromptSelector, **kwargs: Any
    ) -> "Prompt":
        """Load prompt from LangChain prompt."""
        return cls(langchain_prompt_selector=prompt_selector, **kwargs)

    def partial_format(self, **kwargs: Any) -> "Prompt":
        """Format the prompt partially.

        Return an instance of itself.

        """
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
        cls,
        prompt: "Prompt",
        llm: Optional[BaseLanguageModel] = None,
        prompt_type: Optional[PromptType] = None,
    ) -> "Prompt":
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
        cls_obj = cls(
            template_str,
            prompt_type=prompt_type or PromptType.CUSTOM,
            **prompt.prompt_kwargs,
        )
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
