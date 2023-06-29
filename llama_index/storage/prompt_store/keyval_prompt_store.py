from typing import List, Optional

from llama_index.prompts.base import Prompt
from llama_index.storage.prompt_store.types import BasePromptStore
from llama_index.storage.kvstore.types import BaseKVStore


DEFAULT_NAMESPACE = "prompt_store"


class KVPromptStore(BasePromptStore):
    """Key-Value Prompt store.

    Args:
        kvstore (BaseKVStore): key-value store
        namespace (str): namespace for the prompt store

    """

    def __init__(self, kvstore: BaseKVStore, namespace: Optional[str] = None) -> None:
        """Init a KVIndexStore."""
        self._kvstore = kvstore
        self._namespace = namespace or DEFAULT_NAMESPACE
        self._prompt_collection = f"{self._namespace}/prompts"
        self._prompt_registry = f"{self._namespace}/registry"

    def register_prompt(self, prompt_name: str, object_cls: str, prompt: Prompt) -> None:
        """Add a prompt to the registry."""
        self._kvstore.put(prompt_name, {"prompt_str": prompt.original_template}, collection=self._prompt_collection)

        existing_registry = self._kvstore.get(object_cls, collection=self._prompt_registry)
        if existing_registry is None:
            self._kvstore.put(object_cls, {'prompts': [prompt_name]}, collection=self._prompt_registry)
        else:
            existing_registry['prompts'].append(prompt_name)
            self._kvstore.put(object_cls, existing_registry, collection=self._prompt_registry)
      
    def get_prompts(self, object_cls: str) -> List[Prompt]:
        """Get all prompts for a given object class."""
        prompt_registry = self._kvstore.get(object_cls, collection=self._prompt_registry)
        if prompt_registry is None:
            return []
        prompts = []
        for prompt_name in prompt_registry['prompts']:
            prompt_dict = self._kvstore.get(prompt_name, collection=self._prompt_collection)
            prompts.append(Prompt(prompt_dict['prompt_str']))
        return prompts
    
    def get_prompt(self, prompt_name: str) -> Prompt:
        """Get a single prompt by name."""
        prompt_dict = self._kvstore.get(prompt_name, collection=self._prompt_collection)
        return Prompt(prompt_dict['prompt_str'])
