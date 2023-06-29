from abc import ABC, abstractmethod
from typing import List, Optional

from llama_index.prompts.base import Prompt
import os
import fsspec


DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_FNAME = "prompt_store.json"
DEFAULT_PERSIST_PATH = os.path.join(DEFAULT_PERSIST_DIR, DEFAULT_PERSIST_FNAME)


class BasePromptStore(ABC):
    
    @abstractmethod
    def register_prompt(self, prompt_name: str, object_cls: str, prompt: Prompt) -> None:
        """Add a prompt to the registry."""

    @abstractmethod
    def get_prompts(self, object_cls: str) -> List[Prompt]:
        """Get all prompts for a given object class."""

    @abstractmethod    
    def get_prompt(self, prompt_name: str) -> Prompt:
        """Get a single prompt by name."""

    def persist(
        self,
        persist_path: str = DEFAULT_PERSIST_PATH,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Persist the index store to disk."""
        pass

