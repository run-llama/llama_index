"""
Offline configuration tests for SentenceTransformerRerank.

CrossEncoder is stubbed, so no models are downloaded and
`sentence-transformers` is not required.
"""

import inspect
import json
import sys
from types import ModuleType
from typing import Any, Dict, Optional
from unittest.mock import Mock, call

import pytest

from llama_index.core.postprocessor.sbert_rerank import SentenceTransformerRerank


def test_trust_remote_code_constructor_default_is_false() -> None:
    """
    The constructor must not enable remote code execution by default.

    `trust_remote_code` is forwarded straight to `CrossEncoder`, so a `True`
    default lets HuggingFace import and execute whatever `modeling_*.py` the
    model repository ships, for any caller that does not pass the argument.
    """
    signature = inspect.signature(SentenceTransformerRerank.__init__)
    assert signature.parameters["trust_remote_code"].default is False


def test_trust_remote_code_constructor_default_matches_field() -> None:
    """The constructor default must agree with the documented field default."""
    signature = inspect.signature(SentenceTransformerRerank.__init__)
    field = SentenceTransformerRerank.model_fields["trust_remote_code"]
    assert signature.parameters["trust_remote_code"].default is field.default


@pytest.fixture()
def cross_encoder(monkeypatch: pytest.MonkeyPatch) -> Mock:
    encoder = Mock()
    module = ModuleType("sentence_transformers")
    monkeypatch.setattr(module, "CrossEncoder", encoder, raising=False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return encoder


@pytest.mark.parametrize(
    "trust_remote_code", [None, False, True], ids=["omitted", "false", "true"]
)
def test_trust_remote_code_matches_loader_and_serialized_config(
    cross_encoder: Mock, trust_remote_code: Optional[bool]
) -> None:
    kwargs: Dict[str, Any] = (
        {} if trust_remote_code is None else {"trust_remote_code": trust_remote_code}
    )
    reranker = SentenceTransformerRerank(
        model="offline-fixture", device="cpu", **kwargs
    )
    expected = False if trust_remote_code is None else trust_remote_code

    cross_encoder.assert_called_once_with(
        "offline-fixture", max_length=512, device="cpu", trust_remote_code=expected
    )
    assert reranker.trust_remote_code is expected
    assert reranker.to_dict()["trust_remote_code"] is expected
    assert json.loads(reranker.to_json())["trust_remote_code"] is expected


@pytest.mark.parametrize(
    "trust_remote_code", [None, False, True], ids=["omitted", "false", "true"]
)
@pytest.mark.parametrize("serialization_format", ["dict", "json"])
def test_trust_remote_code_survives_round_trip(
    cross_encoder: Mock,
    trust_remote_code: Optional[bool],
    serialization_format: str,
) -> None:
    kwargs: Dict[str, Any] = (
        {} if trust_remote_code is None else {"trust_remote_code": trust_remote_code}
    )
    original = SentenceTransformerRerank(
        model="offline-fixture",
        device="cpu",
        top_n=3,
        keep_retrieval_score=True,
        **kwargs,
    )
    if serialization_format == "dict":
        restored = SentenceTransformerRerank.from_dict(original.to_dict())
    else:
        restored = SentenceTransformerRerank.from_json(original.to_json())
    expected = False if trust_remote_code is None else trust_remote_code

    assert cross_encoder.call_args_list == [
        call(
            "offline-fixture", max_length=512, device="cpu", trust_remote_code=expected
        ),
        call(
            "offline-fixture", max_length=512, device="cpu", trust_remote_code=expected
        ),
    ]
    assert restored.trust_remote_code is expected
    assert restored.to_dict()["trust_remote_code"] is expected
    assert json.loads(restored.to_json())["trust_remote_code"] is expected
    assert restored.model == original.model
    assert restored.device == original.device
    assert restored.top_n == 3
    assert restored.keep_retrieval_score is True
