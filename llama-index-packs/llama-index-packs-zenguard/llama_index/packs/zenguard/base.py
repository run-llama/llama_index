from typing import Any, Dict, List

from llama_index.core.llama_pack import BaseLlamaPack
from zenguard import ZenGuardConfig, Detector, ZenGuard


class ZenGuardPack(BaseLlamaPack):
    def __init__(self, config: ZenGuardConfig):
        self._zenguard = ZenGuard(config)

    def get_modules(self) -> Dict[str, Any]:
        return {"zenguard": self._zenguard}

    def run(self, prompt: str, detectors: List[Detector]) -> Dict[str, Any]:
        return self._zenguard.detect(detectors, prompt)
