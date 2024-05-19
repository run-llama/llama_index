import asyncio
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.core.storage.docstore.types import DEFAULT_BATCH_SIZE, RefDocInfo
from llama_index.storage.kvstore.azure import AzureKVStore
from llama_index.storage.kvstore.azure.base import DEFAULT_PARTITION_KEY, ServiceMode

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
    """Azure Document (Node) store.

    An Azure Table store for Document and Node objects.

    Args:
        azure_kvstore (AzureKVStore): Azure key-value store
        namespace (Optional[str]): namespace for the AzureDocumentStore
        node_collection_suffix (Optional[str]): suffix for node collection
        ref_doc_collection_suffix (Optional[str]): suffix for ref doc collection
        metadata_collection_suffix (Optional[str]): suffix for metadata collection
        batch_size (int): batch size for batch operations
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
        """Init an AzureDocumentStore."""
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
    ) -> "AzureDocumentStore":
        """Load an AzureDocumentStore from an Azure connection string.

        Args:
            connection_string (str): Azure connection string
            namespace (Optional[str]): namespace for the AzureDocumentStore
            node_collection_suffix (Optional[str]): suffix for node collection
            ref_doc_collection_suffix (Optional[str]): suffix for ref doc collection
            metadata_collection_suffix (Optional[str]): suffix for metadata collection
            service_mode (ServiceMode): CosmosDB or Azure Table service mode
        """
        azure_kvstore = AzureKVStore.from_connection_string(
            connection_string, service_mode=service_mode
        )
        return cls(
            azure_kvstore,
            namespace,
            node_collection_suffix,
            ref_doc_collection_suffix,
            metadata_collection_suffix,
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
    ) -> "AzureDocumentStore":
        """Load an AzureDocumentStore from an account name and key.

        Args:
            account_name (str): Azure Storage Account Name
            account_key (str): Azure Storage Account Key
            namespace (Optional[str]): namespace for the AzureDocumentStore
            node_collection_suffix (Optional[str]): suffix for node collection
            ref_doc_collection_suffix (Optional[str]): suffix for ref doc collection
            metadata_collection_suffix (Optional[str]): suffix for metadata collection
            service_mode (ServiceMode): CosmosDB or Azure Table service mode
        """
        azure_kvstore = AzureKVStore.from_account_and_key(
            account_name, account_key, service_mode=service_mode
        )
        return cls(
            azure_kvstore,
            namespace,
            node_collection_suffix,
            ref_doc_collection_suffix,
            metadata_collection_suffix,
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
    ) -> "AzureDocumentStore":
        """Load an AzureDocumentStore from a SAS token.

        Args:
            endpoint (str): Azure Table service endpoint
            sas_token (str): Shared Access Signature token
            namespace (Optional[str]): namespace for the AzureDocumentStore
            node_collection_suffix (Optional[str]): suffix for node collection
            ref_doc_collection_suffix (Optional[str]): suffix for ref doc collection
            metadata_collection_suffix (Optional[str]): suffix for metadata collection
            service_mode (ServiceMode): CosmosDB or Azure Table service mode
        """
        azure_kvstore = AzureKVStore.from_sas_token(
            endpoint, sas_token, service_mode=service_mode
        )
        return cls(
            azure_kvstore,
            namespace,
            node_collection_suffix,
            ref_doc_collection_suffix,
            metadata_collection_suffix,
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
    ) -> "AzureDocumentStore":
        """Load an AzureDocumentStore from an AAD token.

        Args:
            endpoint (str): Azure Table service endpoint
            namespace (Optional[str]): namespace for the AzureDocumentStore
            node_collection_suffix (Optional[str]): suffix for node collection
            ref_doc_collection_suffix (Optional[str]): suffix for ref doc collection
            metadata_collection_suffix (Optional[str]): suffix for metadata collection
            service_mode (ServiceMode): CosmosDB or Azure Table service mode
        """
        azure_kvstore = AzureKVStore.from_aad_token(endpoint, service_mode=service_mode)
        return cls(
            azure_kvstore,
            namespace,
            node_collection_suffix,
            ref_doc_collection_suffix,
            metadata_collection_suffix,
        )

    def _extract_doc_metadatas(
        self, ref_doc_kv_pairs: List[Tuple[str, dict]]
    ) -> List[Tuple[str, Optional[dict]]]:
        """
        Prepare reference document key-value pairs.

        This method processes the ref_doc_kv_pairs, separates the metadata, and flattens
        the list of node IDs. Each node_id becomes a separate entity with the ref_doc_id as
        the PartitionKey and the node_id as the RowKey. The metadata is prepared separately
        with the ref_doc_id.

        Args:
            ref_doc_kv_pairs (Dict[str, List[Tuple[str, dict]]]): Dictionary of reference document key-value pairs.

        Returns:
            List[Tuple[str, Optional[dict]]]
                - Key-value pairs for metadata
        """
        doc_metadatas: List[Tuple[str, dict]] = [
            (doc_id, {"metadata": doc_dict.get("metadata")})
            for doc_id, doc_dict in ref_doc_kv_pairs
        ]
        return doc_metadatas

    def add_documents(
        self,
        nodes: Sequence[BaseNode],
        allow_update: bool = True,
        batch_size: Optional[int] = None,
        store_text: bool = True,
    ) -> None:
        """Add a document to the store.

        Args:
            docs (List[BaseDocument]): documents
            allow_update (bool): allow update of docstore from document

        """
        batch_size = batch_size or self._batch_size

        node_kv_pairs, metadata_kv_pairs, ref_doc_kv_pairs = super()._prepare_kv_pairs(
            nodes, allow_update, store_text
        )

        # Change ref_doc_kv_pairs
        doc_metadata = self._extract_doc_metadatas(ref_doc_kv_pairs)

        self._kvstore.put_all(
            node_kv_pairs,
            collection=self._node_collection,
            batch_size=batch_size,
        )

        self._kvstore.put_all(
            doc_metadata,
            collection=self._ref_doc_collection,
            batch_size=batch_size,
        )

        self._kvstore.put_all(
            metadata_kv_pairs,
            collection=self._metadata_collection,
            batch_size=batch_size,
        )

    async def async_add_documents(
        self,
        nodes: Sequence[BaseNode],
        allow_update: bool = True,
        batch_size: int | None = None,
        store_text: bool = True,
    ) -> None:
        batch_size = batch_size or self._batch_size

        (
            node_kv_pairs,
            metadata_kv_pairs,
            ref_doc_kv_pairs,
        ) = await super()._async_prepare_kv_pairs(nodes, allow_update, store_text)

        # Change ref_doc_kv_pairs
        doc_metadatas = self._extract_doc_metadatas(ref_doc_kv_pairs)

        coroutines = []

        coroutines.append(
            self._kvstore.aput_all(
                node_kv_pairs,
                collection=self._node_collection,
                batch_size=batch_size,
            )
        )

        coroutines.append(
            self._kvstore.aput_all(
                metadata_kv_pairs,
                collection=self._metadata_collection,
                batch_size=batch_size,
            )
        )

        coroutines.append(
            self._kvstore.aput_all(
                doc_metadatas,
                collection=self._doc_metadata_collection,
                batch_size=batch_size,
            )
        )

        coroutines.append(
            self._kvstore.aput_all(
                doc_metadatas,
                collection=self._doc_metadata_collection,
                batch_size=batch_size,
            )
        )

        await asyncio.gather(*coroutines)

    def get_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
        """Get the RefDocInfo for a given ref_doc_id."""
        ref_doc_infos = self._kvstore.query(
            f"PartitionKey eq '{DEFAULT_PARTITION_KEY}' and ref_doc_id eq '{ref_doc_id}'",
            self._metadata_collection,
        )

        node_ids = [doc["RowKey"] for doc in ref_doc_infos]
        if not node_ids:
            return None

        doc_metadata = self._kvstore.get(
            ref_doc_id, collection=self._ref_doc_collection
        )

        ref_doc_info_dict = {
            "node_ids": node_ids,
            "metadata": doc_metadata.get("metadata"),
        }

        # TODO: deprecated legacy support
        return self._remove_legacy_info(ref_doc_info_dict)

    async def aget_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
        """Get the RefDocInfo for a given ref_doc_id."""
        ref_doc_infos_task = self._kvstore.aquery(
            f"PartitionKey eq '{DEFAULT_PARTITION_KEY}' and ref_doc_id eq '{ref_doc_id}'",
            self._metadata_collection,
        )

        doc_metadata_task = self._kvstore.aget(
            ref_doc_id, collection=self._doc_metadata_collection
        )

        ref_doc_infos, doc_metadata = await asyncio.gather(
            *[ref_doc_infos_task, doc_metadata_task]
        )

        node_ids = [doc["RowKey"] async for doc in ref_doc_infos]
        if not node_ids:
            return None

        ref_doc_info_dict = {
            "node_ids": node_ids,
            "metadata": doc_metadata.get("metadata"),
        }

        # TODO: deprecated legacy support
        return self._remove_legacy_info(ref_doc_info_dict)

    def get_all_ref_doc_info(self) -> Optional[Dict[str, RefDocInfo]]:
        """Get a mapping of ref_doc_id -> RefDocInfo for all ingested documents."""
        ref_doc_infos = self._kvstore.query(
            f"PartitionKey eq '{DEFAULT_PARTITION_KEY}'",
            self._metadata_collection,
        )

        # TODO: deprecated legacy support
        all_ref_doc_infos = defaultdict(lambda: {"node_ids": [], "metadata": None})
        for ref_doc_info in ref_doc_infos:
            ref_doc_id = ref_doc_info["ref_doc_id"]
            ref_doc_info_dict = all_ref_doc_infos[ref_doc_id]
            ref_doc_info_dict["node_ids"].append(ref_doc_info["RowKey"])

            if ref_doc_info_dict["metadata"] is None:
                ref_doc = self._kvstore.get(
                    ref_doc_id, collection=self._ref_doc_collection
                )
                ref_doc_info_dict["metadata"] = ref_doc.get("metadata")

        for ref_doc_id, ref_doc_info_dict in all_ref_doc_infos.items():
            all_ref_doc_infos[ref_doc_id] = self._remove_legacy_info(ref_doc_info_dict)

        return all_ref_doc_infos

    async def aget_all_ref_doc_info(self) -> Optional[Dict[str, RefDocInfo]]:
        """Get a mapping of ref_doc_id -> RefDocInfo for all ingested documents."""
        ref_doc_infos = await self._kvstore.aquery(
            f"PartitionKey eq '{DEFAULT_PARTITION_KEY}'",
            self._metadata_collection,
        )

        # TODO: deprecated legacy support
        all_ref_doc_infos = defaultdict(lambda: {"node_ids": [], "metadata": None})
        async for ref_doc_info in ref_doc_infos:
            ref_doc_id = ref_doc_info["ref_doc_id"]
            ref_doc_info_dict = all_ref_doc_infos[ref_doc_id]
            ref_doc_info_dict["node_ids"].append(ref_doc_info["RowKey"])

            if ref_doc_info_dict["metadata"] is None:
                ref_doc = await self._kvstore.aget(
                    ref_doc_id, collection=self._doc_metadata_collection
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
