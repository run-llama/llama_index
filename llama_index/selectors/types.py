from abc import abstractmethod
from typing import List, Sequence, Union

from llama_index.bridge.pydantic import BaseModel
from llama_index.indices.query.schema import QueryBundle, QueryType
from llama_index.prompts.mixin import PromptMixin, PromptMixinType
from llama_index.tools.types import ToolMetadata

MetadataType = Union[str, ToolMetadata]


class SingleSelection(BaseModel):
    """A single selection of a choice."""

    index: int
    reason: str


class MultiSelection(BaseModel):
    """A multi-selection of choices."""

    selections: List[SingleSelection]

    @property
    def ind(self) -> int:
        if len(self.selections) != 1:
            raise ValueError(
                f"There are {len(self.selections)} selections, " "please use .inds."
            )
        return self.selections[0].index

    @property
    def reason(self) -> str:
        if len(self.reasons) != 1:
            raise ValueError(
                f"There are {len(self.reasons)} selections, " "please use .reasons."
            )
        return self.selections[0].reason

    @property
    def inds(self) -> List[int]:
        return [x.index for x in self.selections]

    @property
    def reasons(self) -> List[str]:
        return [x.reason for x in self.selections]


# separate name for clarity and to not confuse function calling model
SelectorResult = MultiSelection


def _wrap_choice(choice: MetadataType) -> ToolMetadata:
    if isinstance(choice, ToolMetadata):
        return choice
    elif isinstance(choice, str):
        return ToolMetadata(description=choice)
    else:
        raise ValueError(f"Unexpected type: {type(choice)}")


def _wrap_query(query: QueryType) -> QueryBundle:
    if isinstance(query, QueryBundle):
        return query
    elif isinstance(query, str):
        return QueryBundle(query_str=query)
    else:
        raise ValueError(f"Unexpected type: {type(query)}")


class BaseSelector(PromptMixin):
    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {}

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
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        pass

    @abstractmethod
    async def _aselect(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        pass
