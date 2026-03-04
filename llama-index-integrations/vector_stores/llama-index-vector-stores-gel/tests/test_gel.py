import subprocess
import pytest
import pytest_asyncio
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    FilterOperator,
    FilterCondition,
    MetadataFilters,
    MetadataFilter,
)

from llama_index.vector_stores.gel import GelVectorStore, get_filter_clause

try:
    subprocess.run(["gel", "project", "init", "--non-interactive"], check=True)
except subprocess.CalledProcessError as e:
    print(e)


NODES = [
    TextNode(
        id_="1",
        text="there are cats in the pond",
        metadata={"location": "pond", "topic": "animals"},
        embedding=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ),
    TextNode(
        id_="2",
        text="ducks are also found in the pond",
        metadata={"location": "pond", "topic": "animals"},
        embedding=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ),
    TextNode(
        id_="3",
        text="fresh apples are available at the market",
        metadata={"location": "market", "topic": "food"},
        embedding=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    ),
    TextNode(
        id_="4",
        text="the market also sells fresh oranges",
        metadata={"location": "market", "topic": "food"},
        embedding=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    ),
    TextNode(
        id_="5",
        text="the new art exhibit is fascinating",
        metadata={"location": "museum", "topic": "art"},
        embedding=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ),
    TextNode(
        id_="6",
        text="a sculpture exhibit is also at the museum",
        metadata={"location": "museum", "topic": "art"},
        embedding=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ),
    TextNode(
        id_="7",
        text="a new coffee shop opened on Main Street",
        metadata={"location": "Main Street", "topic": "food"},
        embedding=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    ),
    TextNode(
        id_="8",
        text="the book club meets at the library",
        metadata={"location": "library", "topic": "reading"},
        embedding=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    ),
    TextNode(
        id_="9",
        text="the library hosts a weekly story time for kids",
        metadata={"location": "library", "topic": "reading"},
        embedding=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    ),
    TextNode(
        id_="10",
        text="a cooking class for beginners is offered at the community center",
        metadata={"location": "community center", "topic": "classes"},
        embedding=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ),
]


@pytest.fixture()
def vectorstore() -> GelVectorStore:
    vectorstore = GelVectorStore()
    vectorstore.clear()
    return vectorstore


@pytest_asyncio.fixture()
async def vectorstore_async() -> GelVectorStore:
    vectorstore = GelVectorStore()
    await vectorstore.aclear()
    return vectorstore


def test_get_filter_clause():
    test_cases = [
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="field", value="value", operator=FilterOperator.EQ.value
                    )
                ]
            ),
            '<str>json_get(.metadata, "field") = "value"',
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="field", value=1, operator=FilterOperator.EQ.value
                    )
                ]
            ),
            '<str>json_get(.metadata, "field") = 1',
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="field", value="value", operator=FilterOperator.NE.value
                    )
                ]
            ),
            '<str>json_get(.metadata, "field") != "value"',
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="field", value="value", operator=FilterOperator.LT.value
                    )
                ]
            ),
            '<str>json_get(.metadata, "field") < "value"',
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="field",
                        value="value",
                        operator=FilterOperator.LTE.value,
                    )
                ]
            ),
            '<str>json_get(.metadata, "field") <= "value"',
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="field", value="value", operator=FilterOperator.GT.value
                    )
                ]
            ),
            '<str>json_get(.metadata, "field") > "value"',
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="field",
                        value="value",
                        operator=FilterOperator.GTE.value,
                    )
                ]
            ),
            '<str>json_get(.metadata, "field") >= "value"',
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="field",
                        value=[1, 2, 3],
                        operator=FilterOperator.IN.value,
                    )
                ]
            ),
            '<str>json_get(.metadata, "field") in array_unpack([1, 2, 3])',
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="field",
                        value=[1, 2, 3],
                        operator=FilterOperator.NIN.value,
                    )
                ]
            ),
            '<str>json_get(.metadata, "field") not in array_unpack([1, 2, 3])',
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="field",
                        value="pattern",
                        operator=FilterOperator.TEXT_MATCH.value,
                    )
                ]
            ),
            '<str>json_get(.metadata, "field") like "pattern"',
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="field1",
                        value="value1",
                        operator=FilterOperator.EQ.value,
                    ),
                    MetadataFilter(
                        key="field2",
                        value="value2",
                        operator=FilterOperator.EQ.value,
                    ),
                ],
                condition=FilterCondition.AND,
            ),
            '(<str>json_get(.metadata, "field1") = "value1" and <str>json_get(.metadata, "field2") = "value2")',
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="field1",
                        value="value1",
                        operator=FilterOperator.EQ.value,
                    ),
                    MetadataFilter(
                        key="field2",
                        value="value2",
                        operator=FilterOperator.EQ.value,
                    ),
                ],
                condition=FilterCondition.OR,
            ),
            '(<str>json_get(.metadata, "field1") = "value1" or <str>json_get(.metadata, "field2") = "value2")',
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilters(
                        filters=[
                            MetadataFilter(
                                key="field1",
                                value=[1, 2, 3],
                                operator=FilterOperator.IN.value,
                            ),
                            MetadataFilter(
                                key="field2",
                                value=100,
                                operator=FilterOperator.GT.value,
                            ),
                        ],
                        condition=FilterCondition.OR,
                    ),
                    MetadataFilter(
                        key="field3",
                        value="%pattern%",
                        operator=FilterOperator.TEXT_MATCH.value,
                    ),
                ],
                condition=FilterCondition.AND,
            ),
            '((<str>json_get(.metadata, "field1") in array_unpack([1, 2, 3]) or <str>json_get(.metadata, "field2") > 100) and <str>json_get(.metadata, "field3") like "%pattern%")',
        ),
    ]

    for filter_dict, expected in test_cases:
        assert get_filter_clause(filter_dict) == expected


def test_add(vectorstore: GelVectorStore):
    inserted_ids = vectorstore.add(NODES)

    assert len(inserted_ids) == len(NODES)

    for node in NODES:
        assert node.id_ in inserted_ids

    vectorstore.clear()


def test_query(vectorstore: GelVectorStore):
    inserted_ids = vectorstore.add(NODES)

    query = VectorStoreQuery(
        query_embedding=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        similarity_top_k=1,
    )
    results = vectorstore.query(query)

    assert len(results.nodes) == 1
    assert results.nodes[0].id_ == "1"

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="location", value="market", operator=FilterOperator.EQ),
            MetadataFilter(
                key="topic", value=["food", "art"], operator=FilterOperator.IN
            ),
        ],
        condition=FilterCondition.AND,
    )

    results = vectorstore.query(
        VectorStoreQuery(
            query_embedding=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            similarity_top_k=1,
            filters=filters,
        )
    )

    assert "1" not in [r.id_ for r in results.nodes]
    vectorstore.clear()


def test_get_nodes(vectorstore: GelVectorStore):
    inserted_ids = vectorstore.add(NODES)
    results = vectorstore.get_nodes(node_ids=["1", "2"])

    assert len(results) == 2
    assert results[0].id_ == "1"
    assert results[1].id_ == "2"

    assert len(vectorstore.get_nodes()) == 0


def test_delete(vectorstore: GelVectorStore):
    inserted_ids = vectorstore.add(NODES)
    vectorstore.delete(ref_doc_id="1")
    results = vectorstore.get_nodes(node_ids=["1"])
    assert len(results) == 0


def test_clear(vectorstore: GelVectorStore):
    inserted_ids = vectorstore.add(NODES)
    vectorstore.clear()
    assert len(vectorstore.get_nodes()) == 0


async def test_async_add(vectorstore_async: GelVectorStore):
    inserted_ids = await vectorstore_async.async_add(NODES)

    assert len(inserted_ids) == len(NODES)

    for node in NODES:
        assert node.id_ in inserted_ids

    await vectorstore_async.aclear()


async def test_aquery(vectorstore_async: GelVectorStore):
    inserted_ids = await vectorstore_async.async_add(NODES)

    query = VectorStoreQuery(
        query_embedding=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        similarity_top_k=1,
    )
    results = await vectorstore_async.aquery(query)

    assert len(results.nodes) == 1
    assert results.nodes[0].id_ == "1"

    await vectorstore_async.aclear()


async def test_aget_nodes(vectorstore_async: GelVectorStore):
    inserted_ids = await vectorstore_async.async_add(NODES)
    results = await vectorstore_async.aget_nodes(node_ids=["1", "2"])

    assert len(results) == 2
    assert results[0].id_ == "1"
    assert results[1].id_ == "2"

    assert len(await vectorstore_async.aget_nodes()) == 0
    await vectorstore_async.aclear()


async def test_adelete(vectorstore_async: GelVectorStore):
    inserted_ids = await vectorstore_async.async_add(NODES)
    await vectorstore_async.adelete(ref_doc_id="1")
    results = await vectorstore_async.aget_nodes(node_ids=["1"])
    assert len(results) == 0

    await vectorstore_async.aclear()
