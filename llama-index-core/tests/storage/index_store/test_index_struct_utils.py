import json
from dataclasses import dataclass

import pytest
from llama_index.core.constants import DATA_KEY, TYPE_KEY
from llama_index.core.data_structs.data_structs import (
    IndexDict,
    IndexGraph,
    IndexList,
    IndexStruct,
    KeywordTable,
)
from llama_index.core.data_structs.struct_type import IndexStructType
from llama_index.core.schema import TextNode
from llama_index.core.storage.index_store.utils import (
    index_struct_to_json,
    json_to_index_struct,
)


def _index_dict() -> IndexDict:
    index_dict = IndexDict(summary="summary")
    for i in range(10):
        index_dict.add_node(TextNode(text="text", id_=f"node_{i}"))
    index_dict.embeddings_dict = {"node_0": [0.1, 0.2]}
    return index_dict


@pytest.mark.parametrize(
    "index_struct",
    [
        _index_dict(),
        IndexList(nodes=["a", "b", "c"]),
        IndexGraph(
            all_nodes={0: "a", 1: "b"},
            root_nodes={1: "b"},
            node_id_to_children_ids={"b": ["a"]},
        ),
        KeywordTable(table={"keyword": {"a", "b"}}),
    ],
)
def test_index_struct_to_json_matches_dataclass_json(
    index_struct: IndexStruct,
) -> None:
    result = index_struct_to_json(index_struct)

    assert result[TYPE_KEY] == index_struct.get_type()
    assert json.loads(result[DATA_KEY]) == json.loads(index_struct.to_json())
    assert json_to_index_struct(result) == index_struct


def test_index_struct_to_json_falls_back_for_custom_serialization() -> None:
    @dataclass
    class CustomIndexStruct(IndexStruct):
        values: list = None  # type: ignore[assignment]

        @classmethod
        def get_type(cls) -> IndexStructType:
            return IndexStructType.LIST

        def to_json(self, *args, **kwargs) -> str:  # type: ignore[no-untyped-def]
            return '{"custom": true}'

    result = index_struct_to_json(CustomIndexStruct(values=[1]))
    assert result[DATA_KEY] == '{"custom": true}'
