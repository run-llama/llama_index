from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilters,
    MetadataFilter,
    VectorStoreQuery,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.base import _to_milvus_filter
from llama_index.vector_stores.milvus.utils import (
    ScalarMetadataFilter,
    ScalarMetadataFilters,
    FilterOperatorFunction,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in MilvusVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_to_milvus_filter_with_scalar_filters():
    filters = None
    scalar_filters = ScalarMetadataFilters(
        filters=[ScalarMetadataFilter(key="a", value=1)]
    )
    expr = _to_milvus_filter(filters, scalar_filters.to_dict())
    assert expr == "ARRAY_CONTAINS(a, 1)"

    scalar_filters = ScalarMetadataFilters(
        filters=[
            ScalarMetadataFilter(
                key="a", value=1, operator=FilterOperatorFunction.NARRAY_CONTAINS
            )
        ]
    )
    expr = _to_milvus_filter(filters, scalar_filters.to_dict())
    assert expr == "not ARRAY_CONTAINS(a, 1)"

    scalar_filters = ScalarMetadataFilters(
        filters=[
            ScalarMetadataFilter(
                key="a", value="b", operator=FilterOperatorFunction.NARRAY_CONTAINS
            ),
            ScalarMetadataFilter(
                key="c", value=2, operator=FilterOperatorFunction.ARRAY_LENGTH
            ),
        ]
    )
    expr = _to_milvus_filter(filters, scalar_filters.to_dict())
    assert expr == "(not ARRAY_CONTAINS(a, 'b') and ARRAY_LENGTH(c) == 2)"


def test_to_milvus_filter_with_various_operators():
    filters = MetadataFilters(filters=[MetadataFilter(key="a", value=1)])
    expr = _to_milvus_filter(filters)
    assert expr == "a == 1"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.NE)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "a != 1"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.GT)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "a > 1"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.GTE)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "a >= 1"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.LT)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "a < 1"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.LTE)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "a <= 1"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=[1, 2, 3], operator=FilterOperator.IN)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "a in [1, 2, 3]"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=[1, 2, 3], operator=FilterOperator.NIN)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "a not in [1, 2, 3]"

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="a", value="substring", operator=FilterOperator.TEXT_MATCH
            )
        ]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "a like 'substring%'"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=1, operator=FilterOperator.CONTAINS)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "array_contains(a, 1)"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=[1, 2], operator=FilterOperator.ANY)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "array_contains_any(a, [1, 2])"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=[1, 2], operator=FilterOperator.ALL)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "array_contains_all(a, [1, 2])"

    filters = MetadataFilters(
        filters=[MetadataFilter(key="a", value=None, operator=FilterOperator.IS_EMPTY)]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "array_length(a) == 0"


def test_to_milvus_filter_with_string_value():
    filters = MetadataFilters(filters=[MetadataFilter(key="a", value="hello")])
    expr = _to_milvus_filter(filters)
    assert expr == "a == 'hello'"

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value="hello", operator=FilterOperator.CONTAINS)
        ]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "array_contains(a, 'hello')"

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=["you", "me"], operator=FilterOperator.ANY)
        ]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "array_contains_any(a, ['you', 'me'])"

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=["you", "me"], operator=FilterOperator.ALL)
        ]
    )
    expr = _to_milvus_filter(filters)
    assert expr == "array_contains_all(a, ['you', 'me'])"


def test_to_milvus_filter_with_multiple_filters():
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=1, operator=FilterOperator.GTE),
            MetadataFilter(key="a", value=10, operator=FilterOperator.LTE),
        ],
        condition=FilterCondition.AND,
    )
    expr = _to_milvus_filter(filters)
    assert expr == "(a >= 1 and a <= 10)"

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=1, operator=FilterOperator.LT),
            MetadataFilter(key="a", value=10, operator=FilterOperator.GT),
        ],
        condition=FilterCondition.OR,
    )
    expr = _to_milvus_filter(filters)
    assert expr == "(a < 1 or a > 10)"


def test_milvus_filter_with_nested_filters():
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=1, operator=FilterOperator.EQ),
            MetadataFilters(
                filters=[
                    MetadataFilter(key="b", value=2, operator=FilterOperator.EQ),
                    MetadataFilter(key="c", value=3, operator=FilterOperator.EQ),
                ],
                condition=FilterCondition.OR,
            ),
        ],
        condition=FilterCondition.AND,
    )
    expr = _to_milvus_filter(filters)
    assert expr == "(a == 1 and (b == 2 or c == 3))"


def test_milvus_vector_store():
    vector_store = MilvusVectorStore(
        dim=1536,
        collection_name="test_collection",
        embedding_field="embedding",
        id_field="id",
        similarity_metric="COSINE",
        overwrite=True,
    )

    node = TextNode(text="Hello world", embedding=[0.5] * 1536)

    vector_store.add([node])

    result = vector_store.query(
        VectorStoreQuery(query_embedding=[0.5] * 1536, similarity_top_k=1)
    )
    assert len(result.nodes) == 1
    assert result.nodes[0].text == "Hello world"
