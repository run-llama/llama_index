import sys
from types import ModuleType
from unittest.mock import MagicMock

mock_azure = ModuleType("mistralai_azure")
mock_azure.MistralAzure = MagicMock()  # type: ignore[attr-defined]
mock_azure_models = ModuleType("mistralai_azure.models")
sys.modules["mistralai_azure"] = mock_azure
sys.modules["mistralai_azure.models"] = mock_azure_models

mock_mistralai = ModuleType("mistralai")
mock_mistralai.Mistral = MagicMock()  # type: ignore[attr-defined]
mock_mistralai_models = ModuleType("mistralai.models")
sys.modules.setdefault("mistralai", mock_mistralai)
sys.modules.setdefault("mistralai.models", mock_mistralai_models)
