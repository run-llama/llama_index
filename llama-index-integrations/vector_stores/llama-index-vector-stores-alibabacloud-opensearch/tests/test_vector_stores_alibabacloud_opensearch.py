import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.alibabacloud_opensearch import (
    AlibabaCloudOpenSearchConfig,
    AlibabaCloudOpenSearchStore,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in AlibabaCloudOpenSearchStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


@pytest.mark.asyncio
async def test_adelete_escapes_ref_doc_id_filter():
    config = AlibabaCloudOpenSearchConfig(
        endpoint="endpoint",
        instance_id="instance",
        username="user",
        password="password",
        table_name="table",
    )
    vector_store = AlibabaCloudOpenSearchStore.model_construct()
    vector_store._config = config
    vector_store._client = MagicMock()
    vector_store._client.fetch.return_value = SimpleNamespace(
        body=json.dumps({"result": [{"id": "node-id"}]})
    )
    vector_store._client.push_documents_async = AsyncMock()

    await vector_store.adelete("x' OR 1=1 OR '1'='1")

    request = vector_store._client.fetch.call_args.args[0]
    assert request.filter == "doc_id='x'' OR 1=1 OR ''1''=''1'"
