"""Llava Completion Pack."""


from typing import Any, Dict

from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.llms.replicate import Replicate


class LlavaCompletionPack(BaseLlamaPack):
    """Llava Completion pack."""

    def __init__(
        self,
        image_url: str,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        import os

        if not os.environ.get("REPLICATE_API_TOKEN", None):
            raise ValueError("Replicate API Token is missing or blank.")

        self.image_url = image_url

        self.llm = Replicate(
            model="yorickvp/llava-13b:2facb4a474a0462c15041b78b1ad70952ea46b5ec6ad29583c0b29dbd4249591",
            image=self.image_url,
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "llm": self.llm,
            "image_url": self.image_url,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.llm.complete(*args, **kwargs)
