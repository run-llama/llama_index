import llama_index.embeddings.dashscope as module


def test_all_contains_importable_names() -> None:
    assert "DashScopeEmbedding" in module.__all__
    for name in module.__all__:
        assert isinstance(name, str)
        assert hasattr(module, name)


def test_star_import() -> None:
    namespace: dict = {}
    exec("from llama_index.embeddings.dashscope import *", namespace)
    assert "DashScopeEmbedding" in namespace
