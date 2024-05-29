from typing import Any, Dict, List

from llama_index.core.llama_pack import BaseLlamaPack
from zenguard import ZenGuardConfig, Detector, ZenGuard


class ZenguardGuardrailsPack(BaseLlamaPack):
    def __init__(self, config: ZenGuardConfig):
        self._guardrails = ZenGuard(config)

    def get_modules(self) -> Dict[str, Any]:
        return {"guardrails": self._guardrails}

    def run(self, prompt: str, detectors: List[Detector]) -> Dict[str, Any]:
        return self._guardrails.detect(detectors, prompt)
