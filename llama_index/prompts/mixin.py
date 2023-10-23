"""Prompt Mixin."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union

from llama_index.prompts.base import BasePromptTemplate

HasPromptType = Union["PromptMixin", BasePromptTemplate]
PromptDictType = Dict[str, BasePromptTemplate]
PromptMixinType = Dict[str, "PromptMixin"]


class PromptMixin(ABC):
    """Prompt mixin.

    This mixin is used in other modules, like query engines, response synthesizers.
    This shows that the module supports getting, setting prompts,
    both within the immediate module as well as child modules.

    """

    def _validate_prompts(
        self, prompts_dict: PromptDictType, module_dict: PromptMixinType
    ) -> None:
        """Validate prompts."""
        # validate each prompt module to make sure there's no overlapping keys
        # (otherwise update will fail)
        # start a counter of keys
        all_prompt_keys = defaultdict(int)
        # add keys from prompt dict
        for key in prompts_dict:
            all_prompt_keys[key] += 1
        # add keys from each prompt module
        for module in module_dict.values():
            for key in module.get_prompts():
                all_prompt_keys[key] += 1
        # check for duplicates
        for key, count in all_prompt_keys.items():
            if count > 1:
                raise ValueError(f"Duplicate prompt key {key} found.")

    def get_prompts(self) -> Dict[str, BasePromptTemplate]:
        """Get a prompt."""
        prompts_dict = self._get_prompts()
        module_dict = self._get_prompt_modules()
        self._validate_prompts(prompts_dict, module_dict)

        all_prompts = prompts_dict
        for prompt_module in module_dict.values():
            all_prompts.update(prompt_module.get_prompts())

        return all_prompts

    def update_prompts(self, **prompts: BasePromptTemplate) -> None:
        """Update prompts.

        Other prompts will remain in place.

        """
        prompt_dict = self._get_prompts()
        prompt_modules = self._get_prompt_modules()

        # update prompts for current module
        self._update_prompts(**{key: prompts[key] for key in prompt_dict})

        # now update prompts for submodules
        for module in prompt_modules.values():
            module.update_prompts(**prompts)

    @abstractmethod
    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""

    @abstractmethod
    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules.

        Return a dictionary of sub-modules within the current module
        that also implement PromptMixin (so that their prompts can also be get/set).

        Can be blank if no sub-modules.

        """

    @abstractmethod
    def _update_prompts(self, **prompts: BasePromptTemplate) -> None:
        """Update prompts."""
        # TODO: make abstractmethod, implement in all subclasses
