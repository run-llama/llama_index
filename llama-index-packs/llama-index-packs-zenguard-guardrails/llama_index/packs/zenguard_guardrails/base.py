from typing import Any, Dict

from llama_index.core.llama_pack import BaseLlamaPack


class ZenguardGuardrailsPack(BaseLlamaPack):
    def get_modules(self) -> Dict[str, Any]:
        raise NotImplementedError("")

    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("")
