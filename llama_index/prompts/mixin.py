"""Prompt Mixin."""

from abc import abstractmethod, ABC
from llama_index.prompts.base import BasePromptTemplate
from typing import Dict, Union, Tuple, Optional
from collections import defaultdict


HasPromptType = Union["PromptMixin", BasePromptTemplate]
PromptDictType = Dict[str, BasePromptTemplate]
PromptMixinType = Dict[str, "PromptMixin"]

class PromptMixin(ABC):
    """Prompt mixin.

    This mixin is used in other modules, like query engines, response synthesizers.
    This shows that the module supports getting, setting prompts,
    both within the immediate module as well as child modules.
    
    """

    def _validate_prompts(self, prompts_dict: PromptDictType, module_dict: PromptMixinType) -> None:
        """Validate prompts."""
        # validate each prompt module to make sure there's no overlapping keys
        # (otherwise update will fail)
        # start a counter of keys
        all_prompt_keys = defaultdict(int)
        # add keys from prompt dict
        for key in prompts_dict.keys():
            all_prompt_keys[key] += 1
        # add keys from each prompt module
        for module in module_dict.values():
            for key in module.get_prompts().keys():
                all_prompt_keys[key] += 1
        # check for duplicates
        for key, count in all_prompt_keys.items():
            if count > 1:
                raise ValueError(f"Duplicate prompt key {key} found.")

    def get_prompts(self) -> Dict[str, BasePromptTemplate]:
        """Get a prompt."""
        all_prompts, _ = self.get_all_prompts_and_modules()
        return all_prompts

    def get_all_prompts_and_modules(self) -> Tuple[PromptDictType, PromptMixinType]:
        """Get all prompts and prompt modules.

        Returns all prompts (including nested) and prompt modules.
        
        """
        prompts_dict, module_dict = self._get_prompts_and_modules()
        self._validate_prompts(prompts_dict, module_dict)
            
        all_prompts = prompts_dict
        for prompt_module in module_dict.values():
            all_prompts.update(prompt_module.get_prompts())
        return all_prompts, module_dict
    
    def _get_prompts_and_modules(self) -> Tuple[PromptDictType, PromptMixinType]:
        """Get prompts and prompt modules.

        Also allows the ability to return the prompt modules.
        
        """
        # TODO: implement in subclasses
        return {}, {}

    def update_prompts(self, **prompts: BasePromptTemplate) -> None:
        """Update prompts.

        Other prompts will remain in place.
        
        """
        prompt_dict, prompt_modules = self._get_prompts_and_modules()

        # Update ones in prompt_dict first. Leftover keys should go to prompt_modules.
        base_prompt_keys = set(prompt_dict.keys())
        leftover_prompt_keys = set(prompts.keys()) - base_prompt_keys

        # update prompts for current module
        self._update_prompts(**{key: prompts[key] for key in base_prompt_keys})

        # update modules with leftover keys
        # first, make sure all keys are at least in one module
        all_prompt_module_keys = {
            key
            for module in prompt_modules.values()
            for key in module.get_prompts().keys()
        }
        if len(leftover_prompt_keys - all_prompt_module_keys) > 0:
            raise ValueError(
                "Prompt keys not found in prompt modules: "
                f"{leftover_prompt_keys - all_prompt_module_keys}"
            )

        for key in leftover_prompt_keys:
            for module in prompt_modules.values():
                module_prompts = module.get_prompts()
                if key in module_prompts:
                    module.update_prompts(**{key: prompts[key]})
                    break

    def _update_prompts(self, **prompts: BasePromptTemplate) -> None:
        """Update prompts."""
        # TODO: make abstractmethod, implement in all subclasses