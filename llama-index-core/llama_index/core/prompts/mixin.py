"""Prompt Mixin."""

from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Union

from llama_index.core.prompts.base import BasePromptTemplate

HasPromptType = Union["PromptMixin", BasePromptTemplate]
PromptDictType = Dict[str, BasePromptTemplate]
PromptMixinType = Dict[str, "PromptMixin"]


class PromptMixin(ABC):
    """
    Prompt mixin.

    This mixin is used in other modules, like query engines, response synthesizers.
    This shows that the module supports getting, setting prompts,
    both within the immediate module as well as child modules.

    """

    def _validate_prompts(
        self,
        prompts_dict: PromptDictType,
        module_dict: PromptMixinType,
    ) -> None:
        """Validate prompts."""
        # check if prompts_dict, module_dict has restricted ":" token
        for key in prompts_dict:
            if ":" in key:
                raise ValueError(f"Prompt key {key} cannot contain ':'.")

        for key in module_dict:
            if ":" in key:
                raise ValueError(f"Prompt key {key} cannot contain ':'.")

    def get_prompts(self) -> Dict[str, BasePromptTemplate]:
        """Get a prompt."""
        prompts_dict = self._get_prompts()
        module_dict = self._get_prompt_modules()
        self._validate_prompts(prompts_dict, module_dict)

        # avoid modifying the original dict
        all_prompts = deepcopy(prompts_dict)
        for module_name, prompt_module in module_dict.items():
            # append module name to each key in sub-modules by ":"
            for key, prompt in prompt_module.get_prompts().items():
                all_prompts[f"{module_name}:{key}"] = prompt
        return all_prompts

    def update_prompts(self, prompts_dict: Dict[str, BasePromptTemplate]) -> None:
        """
        Update prompts.

        Other prompts will remain in place.

        """
        prompt_modules = self._get_prompt_modules()

        # update prompts for current module
        self._update_prompts(prompts_dict)

        # get sub-module keys
        # mapping from module name to sub-module prompt keys
        sub_prompt_dicts: Dict[str, PromptDictType] = defaultdict(dict)
        for key in prompts_dict:
            if ":" in key:
                module_name, sub_key = key.split(":")
                sub_prompt_dicts[module_name][sub_key] = prompts_dict[key]

        # now update prompts for submodules
        for module_name, sub_prompt_dict in sub_prompt_dicts.items():
            if module_name not in prompt_modules:
                raise ValueError(f"Module {module_name} not found.")
            module = prompt_modules[module_name]
            module.update_prompts(sub_prompt_dict)

    @abstractmethod
    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""

    @abstractmethod
    def _get_prompt_modules(self) -> PromptMixinType:
        """
        Get prompt sub-modules.

        Return a dictionary of sub-modules within the current module
        that also implement PromptMixin (so that their prompts can also be get/set).

        Can be blank if no sub-modules.

        """

    @abstractmethod
    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        """Update prompts."""
