import jwt  # noqa

from llama_index.core import Document
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilter,
    MetadataFilters,
    FilterCondition,
    FilterOperator,
)
from llama_index.vector_stores.deeplake import DeepLakeVectorStore


def test_class():
    names_of_base_classes = [b.__name__ for b in DeepLakeVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_e2e():
    vs = DeepLakeVectorStore(dataset_path="mem://test", overwrite=True)
    ids = vs.add(
        nodes=[
            Document(text="Doc 1", embedding=[1, 2, 1], metadata={"a": "1", "b": 10}),
            Document(text="Doc 2", embedding=[1, 2, 2], metadata={"a": "2", "b": 11}),
            Document(text="Doc 3", embedding=[1, 2, 3], metadata={"a": "3", "b": 12}),
        ]
    )

    nodes = vs.get_nodes(node_ids=[ids[0], ids[2]])
    assert [x.text for x in nodes] == ["Doc 1", "Doc 3"]

    nodes = vs.get_nodes(node_ids=["a"])
    assert len(nodes) == 0

    assert [
        x.text
        for x in vs.get_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="a", value="2"),
                ]
            )
        )
    ] == ["Doc 2"]

    assert [
        x.text
        for x in vs.get_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="a", value="2"),
                    MetadataFilter(key="a", value="3"),
                ]
            )
        )
    ] == []

    assert [
        x.text
        for x in vs.get_nodes(
            filters=MetadataFilters(
                condition=FilterCondition.OR,
                filters=[
                    MetadataFilter(key="a", value="2"),
                    MetadataFilter(key="a", value="3"),
                ],
            )
        )
    ] == ["Doc 2", "Doc 3"]

    assert [
        x.text
        for x in vs.get_nodes(
            filters=MetadataFilters(
                condition=FilterCondition.OR,
                filters=[
                    MetadataFilter(key="a", value="2"),
                    MetadataFilter(key="a", value="3"),
                ],
            )
        )
    ] == ["Doc 2", "Doc 3"]

    assert [
        x.text
        for x in vs.get_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="b", value=10, operator=FilterOperator.GT),
                ]
            )
        )
    ] == ["Doc 2", "Doc 3"]

    assert [
        x.text
        for x in vs.get_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="b", value=11, operator=FilterOperator.LTE),
                ]
            )
        )
    ] == ["Doc 1", "Doc 2"]

    vs.delete_nodes(node_ids=[ids[0], ids[2]])
    assert [x.text for x in vs.get_nodes()] == ["Doc 2"]

    vs.add(
        nodes=[
            Document(text="Doc 4", embedding=[1, 2, 1], metadata={"a": "4", "b": 14}),
            Document(text="Doc 5", embedding=[1, 2, 2], metadata={"a": "5", "b": 15}),
            Document(text="Doc 6", embedding=[1, 2, 3], metadata={"a": "6", "b": 16}),
        ]
    )

    vs.delete_nodes(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="b", value=14, operator=FilterOperator.GT),
            ]
        )
    )
    assert [x.text for x in vs.get_nodes()] == ["Doc 2", "Doc 4"]

    vs.clear()
