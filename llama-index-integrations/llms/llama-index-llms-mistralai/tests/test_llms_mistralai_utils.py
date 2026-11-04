import pytest

from llama_index.llms.mistralai.utils import (
    MISTRALAI_CODE_MODELS,
    MISTRALAI_MODELS,
    is_mistralai_code_model,
    is_mistralai_function_calling_model,
    mistralai_modelname_to_contextsize,
)


def test_code_models_are_a_collection_of_names():
    # A bare string here would make `is_mistralai_code_model` a substring test.
    assert isinstance(MISTRALAI_CODE_MODELS, tuple)
    assert "codestral-latest" in MISTRALAI_CODE_MODELS


@pytest.mark.parametrize("model", MISTRALAI_CODE_MODELS)
def test_code_models_are_recognized(model: str):
    assert is_mistralai_code_model(model)


@pytest.mark.parametrize("model", MISTRALAI_CODE_MODELS)
def test_code_models_have_a_known_context_size(model: str):
    assert model in MISTRALAI_MODELS


@pytest.mark.parametrize(
    "model",
    ["mistral-small-latest", "mistral-large-latest", "open-mistral-nemo", "unknown"],
)
def test_non_code_models_are_rejected(model: str):
    assert not is_mistralai_code_model(model)


@pytest.mark.parametrize("model", ["", "l", "code", "codestral", "-latest"])
def test_substrings_of_a_code_model_name_are_rejected(model: str):
    # Regression: these all matched while the model list was a plain string.
    assert not is_mistralai_code_model(model)


def test_function_calling_models_are_recognized():
    assert is_mistralai_function_calling_model("mistral-large-latest")
    assert not is_mistralai_function_calling_model("open-mistral-7b")


def test_modelname_to_contextsize_returns_the_mapped_size():
    assert mistralai_modelname_to_contextsize("mistral-small-2506") == 128_000


def test_modelname_to_contextsize_resolves_latest_tags():
    assert mistralai_modelname_to_contextsize(
        "codestral-latest"
    ) == mistralai_modelname_to_contextsize("codestral-2508")


def test_modelname_to_contextsize_unwraps_finetuned_models():
    assert (
        mistralai_modelname_to_contextsize("ft:mistral-small-2506:abc123")
        == MISTRALAI_MODELS["mistral-small-2506"]
    )


def test_modelname_to_contextsize_rejects_unknown_models():
    with pytest.raises(ValueError, match="Unknown model"):
        mistralai_modelname_to_contextsize("not-a-mistral-model")
