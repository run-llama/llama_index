from typing import Any, Callable, ClassVar, List, Sequence, Type, cast

import pytest

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import Document
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.ondemand_loader_tool import OnDemandLoaderTool


class RecordingIndex:
    instances: ClassVar[List["RecordingIndex"]] = []

    def __init__(self, documents: Sequence[Document]) -> None:
        self.documents = list(documents)

    @classmethod
    def from_documents(
        cls, documents: Sequence[Document], **kwargs: Any
    ) -> "RecordingIndex":
        index = cls(documents)
        cls.instances.append(index)
        return index

    def as_query_engine(self) -> "RecordingIndex":
        return self

    def query(self, query: str) -> str:
        return f"{query}: {self.documents[0].get_content()}"

    async def aquery(self, query: str) -> str:
        return self.query(query)


@pytest.fixture()
def recording_index() -> Type[BaseIndex]:
    RecordingIndex.instances.clear()
    return cast(Type[BaseIndex], RecordingIndex)


class BoundLoaderSchema(BaseModel):
    path: str
    query_str: str


class UnboundLoaderSchema(BoundLoaderSchema):
    tenant: str


@pytest.mark.asyncio
@pytest.mark.parametrize("call_kind", ["call", "acall"])
@pytest.mark.parametrize("bound", [True, False])
@pytest.mark.parametrize("source_kind", ["sync", "async"])
async def test_from_tool_keeps_partial_params(
    recording_index: Type[BaseIndex],
    call_kind: str,
    bound: bool,
    source_kind: str,
) -> None:
    loaded = []

    def load_sync(tenant: str, path: str) -> List[Document]:
        """Load documents for a tenant."""
        loaded.append((tenant, path))
        return [Document(text=f"{tenant}/{path}")]

    async def load_async(tenant: str, path: str) -> List[Document]:
        """Load documents for a tenant."""
        return load_sync(tenant, path)

    loader: Callable[..., Any] = load_sync if source_kind == "sync" else load_async
    original = FunctionTool.from_defaults(
        loader, partial_params={"tenant": "fixed"} if bound else None
    )
    kwargs = {"path": "sample"}
    if not bound:
        kwargs["tenant"] = "fixed"

    control = (
        original.call(**kwargs)
        if call_kind == "call"
        else await original.acall(**kwargs)
    )
    assert control.raw_output[0].get_content() == "fixed/sample"
    assert loaded == [("fixed", "sample")]
    loaded.clear()

    # Async FunctionTool retains its existing sync bridge; its explicit schema
    # supplies the loader's signature to this adapter.
    explicit_schema = (
        (BoundLoaderSchema if bound else UnboundLoaderSchema)
        if source_kind == "async"
        else None
    )
    wrapped = OnDemandLoaderTool.from_tool(
        original,
        name="loader",
        index_cls=recording_index,
        fn_schema=explicit_schema,
    )
    result = (
        wrapped.call(query_str="question", **kwargs)
        if call_kind == "call"
        else await wrapped.acall(query_str="question", **kwargs)
    )

    assert loaded == [("fixed", "sample")]
    assert len(RecordingIndex.instances) == 1
    assert RecordingIndex.instances[0].documents[0].get_content() == "fixed/sample"
    assert result.content == "question: fixed/sample"
    assert wrapped.metadata.fn_schema is not None
    assert set(wrapped.metadata.fn_schema.model_fields) == (
        {"path", "query_str"} if bound else {"tenant", "path", "query_str"}
    )
    if explicit_schema is not None:
        assert wrapped.metadata.fn_schema is explicit_schema


@pytest.mark.asyncio
@pytest.mark.parametrize("call_kind", ["call", "acall"])
async def test_from_tool_allows_overriding_bound_values(
    recording_index: Type[BaseIndex], call_kind: str
) -> None:
    def load_docs(tenant: str, path: str) -> List[Document]:
        """Load documents for a tenant."""
        return [Document(text=f"{tenant}/{path}")]

    original = FunctionTool.from_defaults(load_docs, partial_params={"tenant": "fixed"})
    wrapped = OnDemandLoaderTool.from_tool(
        original, name="loader", index_cls=recording_index
    )
    kwargs = {"tenant": "override", "path": "sample", "query_str": "question"}
    result = (
        wrapped.call(**kwargs) if call_kind == "call" else await wrapped.acall(**kwargs)
    )

    assert RecordingIndex.instances[0].documents[0].get_content() == "override/sample"
    assert result.content == "question: override/sample"
    assert original.partial_params == {"tenant": "fixed"}


@pytest.mark.asyncio
@pytest.mark.parametrize("call_kind", ["call", "acall"])
async def test_from_tool_forwards_custom_query_key(
    recording_index: Type[BaseIndex], call_kind: str
) -> None:
    def load_docs(tenant: str, path: str, question: str) -> List[Document]:
        """Load documents using the query."""
        return [Document(text=f"{tenant}/{path}/{question}")]

    original = FunctionTool.from_defaults(load_docs, partial_params={"tenant": "fixed"})
    wrapped = OnDemandLoaderTool.from_tool(
        original,
        name="loader",
        index_cls=recording_index,
        query_str_kwargs_key="question",
        use_query_str_in_loader=True,
    )
    result = (
        wrapped.call(path="sample", question="find")
        if call_kind == "call"
        else await wrapped.acall(path="sample", question="find")
    )

    assert RecordingIndex.instances[0].documents[0].get_content() == "fixed/sample/find"
    assert result.content == "find: fixed/sample/find"
    assert wrapped.metadata.fn_schema is not None
    assert set(wrapped.metadata.fn_schema.model_fields) == {"path", "question"}


def test_from_tool_still_requires_query(recording_index: Type[BaseIndex]) -> None:
    def load_docs(tenant: str) -> List[Document]:
        """Load documents for a tenant."""
        return [Document(text=tenant)]

    wrapped = OnDemandLoaderTool.from_tool(
        FunctionTool.from_defaults(load_docs, partial_params={"tenant": "fixed"}),
        name="loader",
        index_cls=recording_index,
    )
    with pytest.raises(ValueError, match="Missing query_str"):
        wrapped.call()
    assert RecordingIndex.instances == []
