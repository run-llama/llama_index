from abc import ABC, abstractmethod
from typing import List, Dict, Any
import uuid
import sys
import logging
from enum import Enum
from llama_index.core.bridge.pydantic import (
    BaseModel,
    StrictFloat,
    StrictInt,
    StrictStr,
)
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

from llama_index.core.vector_stores.types import (
    FilterCondition,
    MetadataFilters,
    FilterOperator,
)
from llama_index.core.vector_stores.utils import DEFAULT_TEXT_KEY


logger = logging.getLogger(__name__)

try:
    from pymilvus import Function as BaseMilvusBuiltInFunction
    from pymilvus import FunctionType
except ImportError:
    error_info = (
        "Cannot import built-in function from PyMilvus. "
        "To use Milvus built-in function, "
        "you may need to upgrade pymilvus to the latest version:\n"
        "pip install pymilvus -U\n"
    )
    logger.warning(error_info)


DEFAULT_SPARSE_EMBEDDING_KEY = "sparse_embedding"


class FilterOperatorFunction(str, Enum):
    ARRAY_CONTAINS = "ARRAY_CONTAINS({key}, {value})"  # array contains single value
    NARRAY_CONTAINS = (
        "not ARRAY_CONTAINS({key}, {value})"  # array does not contain single value
    )
    ARRAY_CONTAINS_ANY = (
        "ARRAY_CONTAINS_ANY({key}, {value})"  # array contains any value in the list
    )
    NARRAY_CONTAINS_ANY = "not ARRAY_CONTAINS_ANY({key}, {value})"  # array does not contain any value in the list
    ARRAY_CONTAINS_ALL = (
        "ARRAY_CONTAINS_ALL({key}, {value})"  # array contains all values in the list
    )
    NARRAY_CONTAINS_ALL = "not ARRAY_CONTAINS_ALL({key}, {value})"  # array does not contain all values in the list
    # GT, GTE, LT, LTE not yet supported on ARRAY_LENGTH functions
    ARRAY_LENGTH = "ARRAY_LENGTH({key}) == {value}"  # array length equals value
    NARRAY_LENGTH = "ARRAY_LENGTH({key}) != {value}"  # array length not equals value


class ScalarMetadataFilter(BaseModel):
    key: str
    value: Union[
        StrictInt,
        StrictFloat,
        StrictStr,
        List[StrictStr],
        List[StrictFloat],
        List[StrictInt],
    ]
    operator: FilterOperatorFunction = FilterOperatorFunction.ARRAY_CONTAINS

    def to_dict(self):
        return {"key": self.key, "value": self.value, "operator": self.operator.value}

    @classmethod
    def from_dict(
        cls,
        filter_dict: Dict,
    ) -> "ScalarMetadataFilter":
        """Create ScalarMetadataFilter from dictionary.

        Args:
            filter_dict: Dict with key, value and FilterOperatorFunction.

        """
        return ScalarMetadataFilter.parse_obj(filter_dict)


class ScalarMetadataFilters(BaseModel):
    # scalar metadata filters for advanced vector filtering
    # https://docs.zilliz.com/docs/use-array-fields#advanced-scalar-filtering
    filters: List[ScalarMetadataFilter]
    # and/or such conditions for combining different filters
    condition: Optional[FilterCondition] = FilterCondition.AND

    def to_dict(self):
        return [filter.to_dict() for filter in self.filters]

    @classmethod
    def from_dict(cls, data):
        filters = [ScalarMetadataFilter.from_dict(item) for item in data]
        return cls(filters=filters)


def parse_filter_value(filter_value: any, is_text_match: bool = False):
    if filter_value is None:
        return filter_value

    if is_text_match:
        # Per Milvus, "only prefix pattern match like ab% and equal match like ab(no wildcards) are supported"
        return f"'{filter_value!s}%'"

    if isinstance(filter_value, str):
        # Escape single quotes in strings
        filter_value = filter_value.replace("'", "''")
        return f"'{filter_value!s}'"

    return str(filter_value)


def parse_standard_filters(standard_filters: MetadataFilters = None):
    filters = []
    if standard_filters is None or standard_filters.filters is None:
        return filters, ""

    for filter in standard_filters.filters:
        if isinstance(filter, MetadataFilters):
            filters.append(f"({parse_standard_filters(filter)[1]})")
            continue
        filter_value = parse_filter_value(filter.value)
        if filter_value is None and filter.operator != FilterOperator.IS_EMPTY:
            continue

        if filter.operator == FilterOperator.NIN:
            filters.append(f"{filter.key!s} not in {filter_value}")
        elif filter.operator == FilterOperator.CONTAINS:
            filters.append(f"array_contains({filter.key!s}, {filter_value})")
        elif filter.operator == FilterOperator.ANY:
            filters.append(f"array_contains_any({filter.key!s}, {filter_value})")
        elif filter.operator == FilterOperator.ALL:
            filters.append(f"array_contains_all({filter.key!s}, {filter_value})")
        elif filter.operator == FilterOperator.TEXT_MATCH:
            filters.append(
                f"{filter.key!s} like {parse_filter_value(filter.value, True)}"
            )
        elif filter.operator == FilterOperator.IS_EMPTY:
            # in Milvus, array_length(field_name) returns 0 if the field does not exist or is not an array
            filters.append(f"array_length({filter.key!s}) == 0")
        elif filter.operator in [
            FilterOperator.EQ,
            FilterOperator.GT,
            FilterOperator.LT,
            FilterOperator.NE,
            FilterOperator.GTE,
            FilterOperator.LTE,
            FilterOperator.IN,
        ]:
            filters.append(f"{filter.key!s} {filter.operator.value} {filter_value}")
        else:
            raise ValueError(
                f'Operator {filter.operator} ("{filter.operator.value}") is not supported by Milvus.'
            )

    return filters, f" {standard_filters.condition.value} ".join(filters)


def parse_scalar_filters(scalar_filters: ScalarMetadataFilters = None):
    filters = []
    if scalar_filters is None:
        return filters, ""

    scalar_filters = ScalarMetadataFilters.from_dict(scalar_filters)
    for filter in scalar_filters.filters:
        filter_value = parse_filter_value(filter.value)
        if filter_value is None:
            continue

        operator = filter.operator.value.format(key=filter.key, value=filter_value)
        filters.append(operator)

    return filters, f" {scalar_filters.condition.value} ".join(filters)


class BaseSparseEmbeddingFunction(ABC):
    @abstractmethod
    def encode_queries(self, queries: List[str]) -> List[Dict[int, float]]:
        pass

    async def async_encode_queries(self, queries: List[str]) -> List[Dict[int, float]]:
        """
        Encode queries asynchronously. Use sync method if not implemented.
        """
        return self.encode_queries(queries)

    @abstractmethod
    def encode_documents(self, documents: List[str]) -> List[Dict[int, float]]:
        pass

    async def async_encode_documents(
        self, documents: List[str]
    ) -> List[Dict[int, float]]:
        """
        Encode documents asynchronously. Use sync method if not implemented.
        """
        return self.encode_documents(documents)


class BGEM3SparseEmbeddingFunction(BaseSparseEmbeddingFunction):
    def __init__(self) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel

            self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
        except Exception as ImportError:
            error_info = (
                "Cannot import BGEM3FlagModel from FlagEmbedding. It seems it is not installed. "
                "Please install it using:\n"
                "pip install FlagEmbedding\n"
            )
            logger.fatal(error_info)
            sys.exit(1)

    def encode_queries(self, queries: List[str]):
        outputs = self.model.encode(
            queries, return_dense=False, return_sparse=True, return_colbert_vecs=False
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def encode_documents(self, documents: List[str]):
        outputs = self.model.encode(
            documents, return_dense=False, return_sparse=True, return_colbert_vecs=False
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def _to_standard_dict(self, raw_output):
        result = {}
        for k in raw_output:
            result[int(k)] = raw_output[k]
        return result


def get_default_sparse_embedding_function() -> BGEM3SparseEmbeddingFunction:
    return BGEM3SparseEmbeddingFunction()


class BM25BuiltInFunction(BaseMilvusBuiltInFunction):
    """BM25 function built in Milvus."""

    def __init__(
        self,
        input_field_names: Union[str, List[str]] = DEFAULT_TEXT_KEY,
        output_field_names: Union[str, List[str]] = DEFAULT_SPARSE_EMBEDDING_KEY,
        function_name: str = None,
        function_description: str = "",
        function_params: Optional[Dict] = None,
        analyzer_params: Optional[Dict[Any, Any]] = None,
        enable_match: bool = False,
    ):
        if not function_name:
            function_name = f"bm25_function_{str(uuid.uuid4())[:8]}"
        super().__init__(
            function_name,
            FunctionType.BM25,
            input_field_names,
            output_field_names,
            function_description,
            function_params,
        )
        self.analyzer_params = analyzer_params
        self.enable_match = enable_match

    def get_field_kwargs(self) -> dict:
        field_schema_kwargs: Dict[Any, Any] = {
            "enable_analyzer": True,
            "enable_match": self.enable_match,
        }
        if self.analyzer_params is not None:
            field_schema_kwargs["analyzer_params"] = self.analyzer_params
        return field_schema_kwargs
