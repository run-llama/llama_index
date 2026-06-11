from http import HTTPStatus
from types import SimpleNamespace
import sys

from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.embeddings.dashscope.base import get_multimodal_embedding


def test_dashscope_embedding_class():
    names_of_base_classes = [b.__name__ for b in DashScopeEmbedding.__mro__]
    assert MultiModalEmbedding.__name__ in names_of_base_classes


def test_get_multimodal_embedding_accepts_current_dashscope_shape(monkeypatch):
    class FakeMultiModalEmbedding:
        @staticmethod
        def call(**_kwargs):
            return SimpleNamespace(
                status_code=HTTPStatus.OK,
                output={
                    "embeddings": [
                        {
                            "embedding": [0.1, 0.2, 0.3],
                            "index": 0,
                            "type": "text",
                        }
                    ]
                },
            )

    monkeypatch.setitem(
        sys.modules,
        "dashscope",
        SimpleNamespace(MultiModalEmbedding=FakeMultiModalEmbedding),
    )

    assert get_multimodal_embedding("multimodal-embedding-v1", [{"text": "hello"}]) == [
        0.1,
        0.2,
        0.3,
    ]


def test_get_multimodal_embedding_preserves_legacy_dashscope_shape(monkeypatch):
    class FakeMultiModalEmbedding:
        @staticmethod
        def call(**_kwargs):
            return SimpleNamespace(
                status_code=HTTPStatus.OK,
                output={"embedding": [0.4, 0.5, 0.6]},
            )

    monkeypatch.setitem(
        sys.modules,
        "dashscope",
        SimpleNamespace(MultiModalEmbedding=FakeMultiModalEmbedding),
    )

    assert get_multimodal_embedding("multimodal-embedding-v1", [{"text": "hello"}]) == [
        0.4,
        0.5,
        0.6,
    ]
