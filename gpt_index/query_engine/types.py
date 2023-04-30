from dataclasses import dataclass
from typing import Optional
from gpt_index.indices.query.base import BaseQueryEngine


@dataclass
class Metadata:
    description: str
    name: Optional[str] = None


@dataclass
class QueryEngineWithMetadata:
    query_engine: BaseQueryEngine
    metadata: Metadata
