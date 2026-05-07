import sys
import types

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface_optimum_intel import IntelEmbedding
import llama_index.embeddings.huggingface_optimum_intel.base as optimum_intel_base


def test_optimum_intel_embedding_class():
    names_of_base_classes = [b.__name__ for b in IntelEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


def test_optimum_intel_load_kwargs(monkeypatch):
    model_calls = []
    tokenizer_calls = []

    class MockConfig:
        max_position_embeddings = 512

    class MockModel:
        config = MockConfig()

        def to(self, device):
            self.device = device
            return self

    class MockIPEXModel:
        @classmethod
        def from_pretrained(cls, folder_name, **kwargs):
            model_calls.append((folder_name, kwargs))
            return MockModel()

    class MockTokenizer:
        model_max_length = 256

        @classmethod
        def from_pretrained(cls, folder_name, **kwargs):
            tokenizer_calls.append((folder_name, kwargs))
            return cls()

    optimum = types.ModuleType("optimum")
    optimum_intel = types.ModuleType("optimum.intel")
    optimum_intel.IPEXModel = MockIPEXModel
    monkeypatch.setitem(sys.modules, "optimum", optimum)
    monkeypatch.setitem(sys.modules, "optimum.intel", optimum_intel)
    monkeypatch.setattr(optimum_intel_base, "AutoTokenizer", MockTokenizer)

    embed_model = IntelEmbedding(
        "Intel/bge-small-en-v1.5-rag-int8-static",
        cache_folder="/tmp/hf-cache",
        model_kwargs={"revision": "main"},
        tokenizer_kwargs={"use_fast": True},
        device="cpu",
    )

    assert embed_model.cache_folder == "/tmp/hf-cache"
    assert model_calls == [
        (
            "Intel/bge-small-en-v1.5-rag-int8-static",
            {
                "revision": "main",
                "cache_dir": "/tmp/hf-cache",
                "weights_only": False,
            },
        )
    ]
    assert tokenizer_calls == [
        (
            "Intel/bge-small-en-v1.5-rag-int8-static",
            {"use_fast": True, "cache_dir": "/tmp/hf-cache"},
        )
    ]
