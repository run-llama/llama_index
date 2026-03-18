import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import pytest

MODULE_PATH = Path(
    "/Users/lalakiya/Desktop/Opensource/llama-Index/llama_index/"
    "llama-index-finetuning/llama_index/finetuning/rerankers/cohere_reranker.py"
)


def _load_module():
    cohere_module = ModuleType("cohere")
    cohere_module.Client = MagicMock()
    cohere_finetuning = ModuleType("cohere.finetuning")

    class BaseModel:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Settings:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FinetunedModel:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    cohere_finetuning.BaseModel = BaseModel
    cohere_finetuning.FinetunedModel = FinetunedModel
    cohere_finetuning.Settings = Settings
    cohere_module.finetuning = cohere_finetuning

    types_module = ModuleType("llama_index.finetuning.types")
    types_module.BaseCohereRerankerFinetuningEngine = object
    postprocessor_module = ModuleType("llama_index.postprocessor.cohere_rerank")

    class CohereRerank:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    postprocessor_module.CohereRerank = CohereRerank

    sys.modules["cohere"] = cohere_module
    sys.modules["cohere.finetuning"] = cohere_finetuning
    sys.modules["llama_index.finetuning.types"] = types_module
    sys.modules["llama_index.postprocessor.cohere_rerank"] = postprocessor_module

    spec = importlib.util.spec_from_file_location("cohere_reranker_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.importlib.util.find_spec = lambda _name: object()
    return module


def _mock_dataset(dataset_id: str, validation_status: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=SimpleNamespace(id=dataset_id, validation_status=validation_status)
    )


def test_finetune_uses_cohere_v5_flow() -> None:
    module = _load_module()
    engine = module.CohereRerankerFinetuneEngine(api_key="test_key")

    train_dataset = _mock_dataset("dataset-123", "pending")
    validated_dataset = _mock_dataset("dataset-123", "validated")
    finetuned_model = SimpleNamespace(id="finetuned-456")

    engine._model.datasets.create.return_value = train_dataset
    engine._model.datasets.get.side_effect = [train_dataset, validated_dataset]
    engine._model.finetuning.create_finetuned_model.return_value = finetuned_model

    with patch("builtins.open", mock_open()), patch.object(module.time, "sleep"):
        engine.finetune()

    engine._model.datasets.create.assert_called_once()
    create_kwargs = engine._model.datasets.create.call_args.kwargs
    assert create_kwargs["name"] == "exp_finetune"
    assert create_kwargs["type"] == "rerank-finetune-input"
    assert "data" in create_kwargs

    engine._model.datasets.get.assert_any_call("dataset-123")
    engine._model.finetuning.create_finetuned_model.assert_called_once()
    request = engine._model.finetuning.create_finetuned_model.call_args.kwargs[
        "request"
    ]
    assert request.name == "exp_finetune"
    assert request.settings.dataset_id == "dataset-123"
    assert request.settings.base_model.base_type == "BASE_TYPE_RERANK"
    assert request.settings.base_model.name == "english"
    assert engine._finetune_model == finetuned_model


def test_finetune_raises_on_dataset_validation_failure() -> None:
    module = _load_module()
    engine = module.CohereRerankerFinetuneEngine(api_key="test_key")

    failed_dataset = _mock_dataset("dataset-123", "failed")
    engine._model.datasets.create.return_value = failed_dataset
    engine._model.datasets.get.return_value = failed_dataset

    with patch("builtins.open", mock_open()), patch.object(module.time, "sleep"):
        with pytest.raises(ValueError, match="Cohere dataset validation failed"):
            engine.finetune()

    engine._model.finetuning.create_finetuned_model.assert_not_called()
