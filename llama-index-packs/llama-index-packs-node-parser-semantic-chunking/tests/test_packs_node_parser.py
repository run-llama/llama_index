from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.packs.node_parser_semantic_chunking import (
    SemanticChunkingQueryEnginePack,
)


def test_class():
    names_of_base_classes = [
        b.__name__ for b in SemanticChunkingQueryEnginePack.__mro__
    ]
    assert BaseLlamaPack.__name__ in names_of_base_classes
