import asyncio
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.core.storage.docstore.types import DEFAULT_BATCH_SIZE, RefDocInfo
from llama_index.storage.kvstore.azure import AzureKVStore
from llama_index.storage.kvstore.azure.base import ServiceMode

# https://learn.microsoft.com/en-us/azure/cosmos-db/concepts-limits#per-item-limits
# Azure Table Storage has a size limit of 64KiB for a "Property" and 1MiB for an "Item".
# Typically, nodes (__data__) would be constrained by the 64KiB property limit
# as they're stored in a single property. However, we can increase this limit to 1MiB
# by splitting the node (__data__) across multiple properties, up to the 1MiB "Item" limit.
# AzureKVStore in `STORAGE` ServiceMode handles this for us (Cosmos has different limits).
# Similarly, `ref_doc_info` (nodes associated with a document) can easily exceed
# both the 64KiB property limit and the 1MiB "Item" limit, as a large document could
# be split into more than 1MiB of nodes.
# Therefore, the `node_ids` property of `ref_doc_info` is not inserted into the store or used.
# Instead, the normalized version `metadata` is queried, which has each node as a separate "Item"
# along with it's corresponding `ref_doc_id`. This approach provides the best balance of
# normalization, supporting larger data ingestion while offering similar flexibility
# for future data schema changes.


class AzureDocumentStore(KVDocumentStore):
    """
    Azure Document (Node) store.
    An Azure Table store for Document and Node objects.
    """

    _kvstore: AzureKVStore

    def __init__(
        self,
        azure_kvstore: AzureKVStore,
        namespace: Optional[str] = None,
        node_collection_suffix: Optional[str] = None,
        ref_doc_collection_suffix: Optional[str] = None,
        metadata_collection_suffix: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Initialize an AzureDocumentStore."""
        super().__init__(
            azure_kvstore,
            namespace,
            batch_size,
            node_collection_suffix,
            ref_doc_collection_suffix,
            metadata_collection_suffix,
        )

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        namespace: Optional[str] = None,
        node_collection_suffix: Optional[str] = None,
        ref_doc_collection_suffix: Optional[str] = None,
        metadata_collection_suffix: Optional[str] = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        partition_key: Optional[str] = None,
        **kwargs,
    ) -> "AzureDocumentStore":
        """Initialize an AzureDocumentStore from an Azure connection string."""
        azure_kvstore = AzureKVStore.from_connection_string(
            connection_string,
            service_mode=service_mode,
            partition_key=partition_key,
        )
        return cls(
            azure_kvstore,
            namespace,
            node_collection_suffix,
            ref_doc_collection_suffix,
            metadata_collection_suffix,
            **kwargs,
        )

    @classmethod
    def from_account_and_key(
        cls,
        account_name: str,
        account_key: str,
        namespace: Optional[str] = None,
        node_collection_suffix: Optional[str] = None,
        ref_doc_collection_suffix: Optional[str] = None,
        metadata_collection_suffix: Optional[str] = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        partition_key: Optional[str] = None,
        **kwargs,
    ) -> "AzureDocumentStore":
        """Initialize an AzureDocumentStore from an account name and key."""
        azure_kvstore = AzureKVStore.from_account_and_key(
            account_name,
            account_key,
            service_mode=service_mode,
            partition_key=partition_key,
        )
        return cls(
            azure_kvstore,
            namespace,
            node_collection_suffix,
            ref_doc_collection_suffix,
            metadata_collection_suffix,
            **kwargs,
        )

    @classmethod
    def from_account_and_id(
        cls,
        account_name: str,
        namespace: Optional[str] = None,
        node_collection_suffix: Optional[str] = None,
        ref_doc_collection_suffix: Optional[str] = None,
        metadata_collection_suffix: Optional[str] = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        partition_key: Optional[str] = None,
        **kwargs,
    ) -> "AzureDocumentStore":
        """Initialize an AzureDocumentStore from an account name and managed ID."""
        azure_kvstore = AzureKVStore.from_account_and_id(
            account_name,
            service_mode=service_mode,
            partition_key=partition_key,
        )
        return cls(
            azure_kvstore,
            namespace,
            node_collection_suffix,
            ref_doc_collection_suffix,
            metadata_collection_suffix,
            **kwargs,
        )

    @classmethod
    def from_sas_token(
        cls,
        endpoint: str,
        sas_token: str,
        namespace: Optional[str] = None,
        node_collection_suffix: Optional[str] = None,
        ref_doc_collection_suffix: Optional[str] = None,
        metadata_collection_suffix: Optional[str] = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        partition_key: Optional[str] = None,
        **kwargs,
    ) -> "AzureDocumentStore":
        """Initialize an AzureDocumentStore from a SAS token."""
        azure_kvstore = AzureKVStore.from_sas_token(
            endpoint,
            sas_token,
            service_mode=service_mode,
            partition_key=partition_key,
        )
        return cls(
            azure_kvstore,
            namespace,
            node_collection_suffix,
            ref_doc_collection_suffix,
            metadata_collection_suffix,
            **kwargs,
        )

    @classmethod
    def from_aad_token(
        cls,
        endpoint: str,
        namespace: Optional[str] = None,
        node_collection_suffix: Optional[str] = None,
        ref_doc_collection_suffix: Optional[str] = None,
        metadata_collection_suffix: Optional[str] = None,
        service_mode: ServiceMode = ServiceMode.STORAGE,
        partition_key: Optional[str] = None,
        **kwargs,
    ) -> "AzureDocumentStore":
        """Initialize an AzureDocumentStore from an AAD token."""
        azure_kvstore = AzureKVStore.from_aad_token(
            endpoint,
            service_mode=service_mode,
            partition_key=partition_key,
        )
        return cls(
            azure_kvstore,
            namespace,
            node_collection_suffix,
            ref_doc_collection_suffix,
            metadata_collection_suffix,
            **kwargs,
        )

    def _extract_doc_metadatas(
        self, ref_doc_kv_pairs: List[Tuple[str, dict]]
    ) -> List[Tuple[str, Optional[dict]]]:
        """Prepare reference document key-value pairs."""
        doc_metadatas: List[Tuple[str, dict]] = [
            (doc_id, {"metadata": doc_dict.get("metadata")})
            for doc_id, doc_dict in ref_doc_kv_pairs
        ]
        return doc_metadatas

    def add_documents(
        self,
        docs: Sequence[BaseNode],
        allow_update: bool = True,
        batch_size: Optional[int] = None,
        store_text: bool = True,
    ) -> None:
        """Add documents to the store."""
        batch_size = batch_size or self._batch_size

        node_kv_pairs, metadata_kv_pairs, ref_doc_kv_pairs = super()._prepare_kv_pairs(
            docs, allow_update, store_text
        )

        # Change ref_doc_kv_pairs
        ref_doc_kv_pairs = self._extract_doc_metadatas(ref_doc_kv_pairs)

        self._kvstore.put_all(
            node_kv_pairs,
            collection=self._node_collection,
            batch_size=batch_size,
        )

        self._kvstore.put_all(
            metadata_kv_pairs,
            collection=self._metadata_collection,
            batch_size=batch_size,
        )

        self._kvstore.put_all(
            ref_doc_kv_pairs,
            collection=self._ref_doc_collection,
            batch_size=batch_size,
        )

    async def async_add_documents(
        self,
        docs: Sequence[BaseNode],
        allow_update: bool = True,
        batch_size: Optional[int] = None,
        store_text: bool = True,
    ) -> None:
        """Add documents to the store."""
        batch_size = batch_size or self._batch_size

        (
            node_kv_pairs,
            metadata_kv_pairs,
            ref_doc_kv_pairs,
        ) = await super()._async_prepare_kv_pairs(docs, allow_update, store_text)

        # Change ref_doc_kv_pairs
        ref_doc_kv_pairs = self._extract_doc_metadatas(ref_doc_kv_pairs)

        await asyncio.gather(
            self._kvstore.aput_all(
                node_kv_pairs,
                collection=self._node_collection,
                batch_size=batch_size,
            ),
            self._kvstore.aput_all(
                metadata_kv_pairs,
                collection=self._metadata_collection,
                batch_size=batch_size,
            ),
            self._kvstore.aput_all(
                ref_doc_kv_pairs,
                collection=self._ref_doc_collection,
                batch_size=batch_size,
            ),
        )

    def get_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
        """Get the RefDocInfo for a given ref_doc_id."""
        ref_doc_infos = self._kvstore.query(
            f"PartitionKey eq '{self._kvstore.partition_key}' and ref_doc_id eq '{ref_doc_id}'",
            self._metadata_collection,
            select="RowKey",
        )

        node_ids = [doc["RowKey"] for doc in ref_doc_infos]
        if not node_ids:
            return None

        doc_metadata = self._kvstore.get(
            ref_doc_id, collection=self._ref_doc_collection, select="metadata"
        )

        ref_doc_info_dict = {
            "node_ids": node_ids,
            "metadata": doc_metadata.get("metadata"),
        }

        # TODO: deprecated legacy support
        return self._remove_legacy_info(ref_doc_info_dict)

    async def aget_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
        """Get the RefDocInfo for a given ref_doc_id."""
        metadatas = await self._kvstore.aquery(
            f"PartitionKey eq '{self._kvstore.partition_key}' and RowKey eq '{ref_doc_id}'",
            self._metadata_collection,
            select="RowKey",
        )

        node_ids = [metadata["RowKey"] async for metadata in metadatas]

        if not node_ids:
            return None

        doc_metadata = await self._kvstore.aget(
            ref_doc_id, collection=self._ref_doc_collection, select="metadata"
        )

        ref_doc_info_dict = {
            "node_ids": node_ids,
            "metadata": doc_metadata.get("metadata") if doc_metadata else None,
        }

        # TODO: deprecated legacy support
        return self._remove_legacy_info(ref_doc_info_dict)

    def get_all_ref_doc_info(self) -> Optional[Dict[str, RefDocInfo]]:
        """
        Get a mapping of ref_doc_id -> RefDocInfo for all ingested documents.
        """
        ref_doc_infos = self._kvstore.query(
            f"PartitionKey eq '{self._kvstore.partition_key}'",
            self._metadata_collection,
            select=["RowKey", "ref_doc_id"],
        )

        # TODO: deprecated legacy support
        all_ref_doc_infos = defaultdict(lambda: {"node_ids": [], "metadata": None})
        for ref_doc_info in ref_doc_infos:
            ref_doc_id = ref_doc_info["ref_doc_id"]
            ref_doc_info_dict = all_ref_doc_infos[ref_doc_id]
            ref_doc_info_dict["node_ids"].append(ref_doc_info["RowKey"])

            if ref_doc_info_dict["metadata"] is None:
                ref_doc = self._kvstore.get(
                    ref_doc_id, collection=self._ref_doc_collection, select="metadata"
                )
                ref_doc_info_dict["metadata"] = ref_doc.get("metadata")

        for ref_doc_id, ref_doc_info_dict in all_ref_doc_infos.items():
            all_ref_doc_infos[ref_doc_id] = self._remove_legacy_info(ref_doc_info_dict)

        return all_ref_doc_infos

    async def aget_all_ref_doc_info(self) -> Optional[Dict[str, RefDocInfo]]:
        """
        Get a mapping of ref_doc_id -> RefDocInfo for all ingested documents.
        """
        ref_doc_infos = await self._kvstore.aquery(
            f"PartitionKey eq '{self._kvstore.partition_key}'",
            self._metadata_collection,
            select=["RowKey", "ref_doc_id"],
        )

        # TODO: deprecated legacy support
        all_ref_doc_infos = defaultdict(lambda: {"node_ids": [], "metadata": None})
        async for ref_doc_info in ref_doc_infos:
            ref_doc_id = ref_doc_info["ref_doc_id"]
            ref_doc_info_dict = all_ref_doc_infos[ref_doc_id]
            ref_doc_info_dict["node_ids"].append(ref_doc_info["RowKey"])

            if ref_doc_info_dict["metadata"] is None:
                ref_doc = await self._kvstore.aget(
                    ref_doc_id, collection=self._ref_doc_collection, select="metadata"
                )
                ref_doc_info_dict["metadata"] = ref_doc.get("metadata")

        for ref_doc_id, ref_doc_info_dict in all_ref_doc_infos.items():
            all_ref_doc_infos[ref_doc_id] = self._remove_legacy_info(ref_doc_info_dict)

        return all_ref_doc_infos

    def _remove_from_ref_doc_node(self, doc_id: str) -> None:
        """
        Helper function to remove node doc_id from ref_doc_collection.
        If ref_doc has no more doc_ids, delete it from the collection.
        """
        self._kvstore.delete(doc_id, collection=self._metadata_collection)

    async def _aremove_from_ref_doc_node(self, doc_id: str) -> None:
        """
        Helper function to remove node doc_id from ref_doc_collection.
        If ref_doc has no more doc_ids, delete it from the collection.
        """
        await self._kvstore.adelete(doc_id, collection=self._metadata_collection)
