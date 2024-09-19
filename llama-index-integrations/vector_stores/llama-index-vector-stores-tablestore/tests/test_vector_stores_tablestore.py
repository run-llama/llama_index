import os

import pytest
import tablestore

from llama_index.core import MockEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    FilterCondition,
)
from llama_index.vector_stores.tablestore import TablestoreVectorStore


def test_class():
    names_of_base_classes = [b.__name__ for b in TablestoreVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_tablestore() -> None:
    """Test end to end construction and search."""
    end_point = os.getenv("end_point")
    instance_name = os.getenv("instance_name")
    access_key_id = os.getenv("access_key_id")
    access_key_secret = os.getenv("access_key_secret")
    if (
        end_point is None
        or instance_name is None
        or access_key_id is None
        or access_key_secret is None
    ):
        pytest.skip(
            "end_point is None or instance_name is None or "
            "access_key_id is None or access_key_secret is None"
        )

    # 1. create tablestore vector store
    test_dimension_size = 4
    ref_doc_id_field = "ref_doc_id"
    store = TablestoreVectorStore(
        endpoint=os.getenv("end_point"),
        instance_name=os.getenv("instance_name"),
        access_key_id=os.getenv("access_key_id"),
        access_key_secret=os.getenv("access_key_secret"),
        vector_dimension=test_dimension_size,
        vector_metric_type=tablestore.VectorMetricType.VM_COSINE,
        ref_doc_id_field=ref_doc_id_field,
        # metadata mapping is used to filter non-vector fields.
        metadata_mappings=[
            tablestore.FieldSchema(
                "type",
                tablestore.FieldType.KEYWORD,
                index=True,
                enable_sort_and_agg=True,
            ),
            tablestore.FieldSchema(
                "time", tablestore.FieldType.LONG, index=True, enable_sort_and_agg=True
            ),
        ],
    )

    # 2. create table and index
    store.create_table_if_not_exist()
    store.create_search_index_if_not_exist()

    # 3. new a mock embedding for test
    embedder = MockEmbedding(test_dimension_size)

    # 4. prepare some docs
    movies = [
        TextNode(
            id_="1",
            text="hello world",
            metadata={"type": "a", "time": 1995, ref_doc_id_field: "1"},
        ),
        TextNode(
            id_="2",
            text="a b c",
            metadata={"type": "a", "time": 1990, ref_doc_id_field: "1"},
        ),
        TextNode(
            id_="3",
            text="sky cloud table",
            metadata={"type": "a", "time": 2009, ref_doc_id_field: "2"},
        ),
        TextNode(
            id_="4",
            text="dog cat",
            metadata={"type": "a", "time": 2023, ref_doc_id_field: "3"},
        ),
        TextNode(
            id_="5",
            text="computer python java",
            metadata={"type": "b", "time": 2018, ref_doc_id_field: "4"},
        ),
        TextNode(
            id_="6",
            text="java python js nodejs",
            metadata={"type": "c", "time": 2010, ref_doc_id_field: "5"},
        ),
        TextNode(
            id_="7",
            text="sdk golang python",
            metadata={"type": "a", "time": 2023, ref_doc_id_field: "6"},
        ),
    ]
    for movie in movies:
        movie.embedding = embedder.get_text_embedding(movie.text)

    # 5. write some docs
    ids = store.add(movies)
    assert len(ids) == 7

    nodes = store.get_nodes(["0", "1", "7", "8"])
    assert len(nodes) == 4
    assert nodes[0] is None
    assert nodes[3] is None

    nodes = store.get_nodes(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="time", value=2000, operator=FilterOperator.GTE),
            ],
            condition=FilterCondition.AND,
        )
    )
    assert len(nodes) == 5

    # 6. delete docs
    store.delete_nodes(["0"])
    store.delete_nodes(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="time", value=1990, operator=FilterOperator.GTE),
                MetadataFilter(key="time", value=1995, operator=FilterOperator.LT),
            ],
            condition=FilterCondition.AND,
        )
    )
    store.delete(ref_doc_id="1")

    # 7. query with filters
    query_embedding = embedder.get_text_embedding("nature fight physical")
    # modify it for test
    query_embedding[0] = 0.1
    query_result = store.query(
        query=VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=5,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="type", value="a", operator=FilterOperator.EQ),
                    MetadataFilter(key="time", value=2020, operator=FilterOperator.LTE),
                ],
                condition=FilterCondition.AND,
            ),
        ),
    )
    print(query_result)
    assert query_result is not None
    assert query_result.ids is not None
    assert query_result.similarities is not None
    assert query_result.similarities is not None
