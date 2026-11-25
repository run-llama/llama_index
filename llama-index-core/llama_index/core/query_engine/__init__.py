import importlib
from typing import Any

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.query_engine.citation_query_engine import CitationQueryEngine
from llama_index.core.query_engine.cogniswitch_query_engine import (
    CogniswitchQueryEngine,
)
from llama_index.core.query_engine.custom import CustomQueryEngine
from llama_index.core.query_engine.flare.base import FLAREInstructQueryEngine
from llama_index.core.query_engine.graph_query_engine import (
    ComposableGraphQueryEngine,
)
from llama_index.core.query_engine.jsonalyze import (
    JSONalyzeQueryEngine,
)
from llama_index.core.query_engine.knowledge_graph_query_engine import (
    KnowledgeGraphQueryEngine,
)
from llama_index.core.query_engine.multi_modal import SimpleMultiModalQueryEngine
from llama_index.core.query_engine.multistep_query_engine import (
    MultiStepQueryEngine,
)
from llama_index.core.query_engine.pandas.pandas_query_engine import (
    PandasQueryEngine,
)
from llama_index.core.query_engine.retriever_query_engine import (
    RetrieverQueryEngine,
)
from llama_index.core.query_engine.retry_query_engine import (
    RetryGuidelineQueryEngine,
    RetryQueryEngine,
)
from llama_index.core.query_engine.retry_source_query_engine import (
    RetrySourceQueryEngine,
)
from llama_index.core.query_engine.router_query_engine import (
    RetrieverRouterQueryEngine,
    RouterQueryEngine,
    ToolRetrieverRouterQueryEngine,
)
from llama_index.core.query_engine.sub_question_query_engine import (
    SubQuestionAnswerPair,
    SubQuestionQueryEngine,
)
from llama_index.core.query_engine.transform_query_engine import (
    TransformQueryEngine,
)

__all__ = [
    "CitationQueryEngine",
    "CogniswitchQueryEngine",
    "ComposableGraphQueryEngine",
    "RetrieverQueryEngine",
    "TransformQueryEngine",
    "MultiStepQueryEngine",
    "RouterQueryEngine",
    "RetrieverRouterQueryEngine",
    "ToolRetrieverRouterQueryEngine",
    "SubQuestionQueryEngine",
    "SubQuestionAnswerPair",
    "SQLJoinQueryEngine",
    "SQLAutoVectorQueryEngine",
    "RetryQueryEngine",
    "RetrySourceQueryEngine",
    "RetryGuidelineQueryEngine",
    "FLAREInstructQueryEngine",
    "PandasQueryEngine",
    "JSONalyzeQueryEngine",
    "KnowledgeGraphQueryEngine",
    "BaseQueryEngine",
    "CustomQueryEngine",
    # multimodal
    "SimpleMultiModalQueryEngine",
    # SQL
    "SQLTableRetrieverQueryEngine",
    "NLSQLTableQueryEngine",
    "PGVectorSQLQueryEngine",
]


# SQL support imports SQLAlchemy, which is slow to load, so these names are
# resolved on first access (PEP 562) instead of at package import time.
_LAZY_IMPORTS = {
    "NLSQLTableQueryEngine": "llama_index.core.indices.struct_store.sql_query",
    "PGVectorSQLQueryEngine": "llama_index.core.indices.struct_store.sql_query",
    "SQLTableRetrieverQueryEngine": "llama_index.core.indices.struct_store.sql_query",
    "SQLJoinQueryEngine": "llama_index.core.query_engine.sql_join_query_engine",
    "SQLAutoVectorQueryEngine": "llama_index.core.query_engine.sql_vector_query_engine",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        value = getattr(importlib.import_module(_LAZY_IMPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
