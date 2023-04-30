from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence, Union

from gpt_index.indices.query.schema import QueryBundle
from gpt_index.query_engine.types import Metadata



@dataclass
class SelectorResult:
    selection_inds: List[int]

    @property
    def selection_ind(self) -> int:
        if len(self.selection_inds) != 1:
            raise ValueError(
                f"There are {len(self.selection_inds)} selections, "
                "please use .selection_inds."
            )
        return self.selection_inds[0]


MetadataType = Union[str, Metadata]
QueryType = Union[str, QueryBundle]


def _wrap_choice(choice: MetadataType) -> Metadata:
    if isinstance(choice, Metadata):
        return choice
    elif isinstance(choice, str):
        return Metadata(description=choice)
    else:
        raise ValueError(f"Unexpected type: {type(choice)}")


def _wrap_query(query: QueryType) -> QueryBundle:
    if isinstance(query, QueryBundle):
        return query
    elif isinstance(query, str):
        return QueryBundle(query_str=query)
    else:
        raise ValueError(f"Unexpected type: {type(query)}")


class BaseSelector(ABC):
    def select(
        self, choices: Sequence[MetadataType], query: QueryType
    ) -> SelectorResult:
        metadatas = [_wrap_choice(choice) for choice in choices]
        query_bundle = _wrap_query(query)
        return self._select(choices=metadatas, query=query_bundle)

    async def aselect(
        self, choices: Sequence[MetadataType], query: QueryType
    ) -> SelectorResult:
        metadatas = [_wrap_choice(choice) for choice in choices]
        query_bundle = _wrap_query(query)
        return await self._aselect(choices=metadatas, query=query_bundle)

    @abstractmethod
    def _select(
        self, choices_metadata: Sequence[MetadataType], query: QueryType
    ) -> SelectorResult:
        pass

    @abstractmethod
    async def _aselect(
        self, choices_metadata: Sequence[MetadataType], query: QueryType
    ) -> SelectorResult:
        pass
