"""Document store."""

from typing import Dict, Optional, Sequence

from llama_index.data_structs.node import Node
from llama_index.storage.docstore.types import BaseDocumentStore, RefDocInfo
from llama_index.storage.docstore.utils import doc_to_json, json_to_doc
from llama_index.schema import BaseDocument
from llama_index.storage.kvstore.types import BaseKVStore

DEFAULT_NAMESPACE = "docstore"


class KVDocumentStore(BaseDocumentStore):
    """Document (Node) store.

    NOTE: at the moment, this store is primarily used to store Node objects.
    Each node will be assigned an ID.

    The same docstore can be reused across index structures. This
    allows you to reuse the same storage for multiple index structures;
    otherwise, each index would create a docstore under the hood.

    .. code-block:: python
        nodes = SimpleNodeParser.get_nodes_from_documents()
        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)
        storage_context = StorageContext.from_defaults(docstore=docstore)

        list_index = GPTListIndex(nodes, storage_context=storage_context)
        vector_index = GPTVectorStoreIndex(nodes, storage_context=storage_context)
        keyword_table_index = GPTSimpleKeywordTableIndex(
            nodes,
            storage_context=storage_context
        )

    This will use the same docstore for multiple index structures.

    Args:
        kvstore (BaseKVStore): key-value store
        namespace (str): namespace for the docstore

    """

    def __init__(
        self,
        kvstore: BaseKVStore,
        namespace: Optional[str] = None,
    ) -> None:
        """Init a KVDocumentStore."""
        self._kvstore = kvstore
        namespace = namespace or DEFAULT_NAMESPACE
        self._node_collection = f"{namespace}/data"
        self._ref_doc_collection = f"{namespace}/ref_doc_info"
        self._metadata_collection = f"{namespace}/metadata"

    @property
    def docs(self) -> Dict[str, BaseDocument]:
        """Get all documents.

        Returns:
            Dict[str, BaseDocument]: documents

        """
        json_dict = self._kvstore.get_all(collection=self._node_collection)
        return {key: json_to_doc(json) for key, json in json_dict.items()}

    def add_documents(
        self, docs: Sequence[BaseDocument], allow_update: bool = True
    ) -> None:
        """Add a document to the store.

        Args:
            docs (List[BaseDocument]): documents
            allow_update (bool): allow update of docstore from document

        """
        for doc in docs:
            if doc.is_doc_id_none:
                raise ValueError("doc_id not set")

            # NOTE: doc could already exist in the store, but we overwrite it
            if not allow_update and self.document_exists(doc.get_doc_id()):
                raise ValueError(
                    f"doc_id {doc.get_doc_id()} already exists. "
                    "Set allow_update to True to overwrite."
                )
            node_key = doc.get_doc_id()
            data = doc_to_json(doc)
            self._kvstore.put(node_key, data, collection=self._node_collection)

            # update doc_collection if needed
            metadata = {"doc_hash": doc.get_doc_hash()}
            if isinstance(doc, Node) and doc.ref_doc_id is not None:
                ref_doc_info = self.get_ref_doc_info(doc.ref_doc_id) or RefDocInfo()
                ref_doc_info.doc_ids.append(doc.get_doc_id())
                if not ref_doc_info.extra_info:
                    ref_doc_info.extra_info = doc.extra_info or {}
                self._kvstore.put(
                    doc.ref_doc_id,
                    ref_doc_info.to_dict(),
                    collection=self._ref_doc_collection,
                )

                # update metadata with map
                metadata["ref_doc_id"] = doc.ref_doc_id
                self._kvstore.put(
                    node_key, metadata, collection=self._metadata_collection
                )
            else:
                self._kvstore.put(
                    node_key, metadata, collection=self._metadata_collection
                )

    def get_document(
        self, doc_id: str, raise_error: bool = True
    ) -> Optional[BaseDocument]:
        """Get a document from the store.

        Args:
            doc_id (str): document id
            raise_error (bool): raise error if doc_id not found

        """
        json = self._kvstore.get(doc_id, collection=self._node_collection)
        if json is None:
            if raise_error:
                raise ValueError(f"doc_id {doc_id} not found.")
            else:
                return None
        return json_to_doc(json)

    def get_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
        """Get the RefDocInfo for a given ref_doc_id."""
        ref_doc_info = self._kvstore.get(
            ref_doc_id, collection=self._ref_doc_collection
        )
        if not ref_doc_info:
            return None
        return RefDocInfo(**ref_doc_info)

    def get_all_ref_doc_info(self) -> Optional[Dict[str, RefDocInfo]]:
        """Get a mapping of ref_doc_id -> RefDocInfo for all ingested documents."""
        ref_doc_infos = self._kvstore.get_all(collection=self._ref_doc_collection)
        if ref_doc_infos is None:
            return None
        return {key: RefDocInfo(**val) for key, val in ref_doc_infos.items()}

    def ref_doc_exists(self, ref_doc_id: str) -> bool:
        """Check if a ref_doc_id has been ingested."""
        return self.get_ref_doc_info(ref_doc_id) is not None

    def document_exists(self, doc_id: str) -> bool:
        """Check if document exists."""
        return self._kvstore.get(doc_id, self._node_collection) is not None

    def _remove_ref_doc_node(self, doc_id: str) -> None:
        """Helper function to remove node doc_id from ref_doc_collection."""
        metadata = self._kvstore.get(doc_id, collection=self._metadata_collection)
        if metadata is None:
            return

        ref_doc_id = metadata.get("ref_doc_id", None)
        if ref_doc_id is None:
            return

        ref_doc_info = self._kvstore.get(
            ref_doc_id, collection=self._ref_doc_collection
        )
        if ref_doc_info is not None:
            ref_doc_obj = RefDocInfo(**ref_doc_info)
            ref_doc_obj.doc_ids.remove(doc_id)

            # delete ref_doc from collection if it has no more doc_ids
            if len(ref_doc_obj.doc_ids) > 0:
                self._kvstore.put(
                    ref_doc_id,
                    ref_doc_obj.to_dict(),
                    collection=self._ref_doc_collection,
                )
            else:
                self._kvstore.delete(ref_doc_id, collection=self._ref_doc_collection)

    def delete_document(
        self, doc_id: str, raise_error: bool = True, remove_ref_doc_node: bool = True
    ) -> None:
        """Delete a document from the store."""
        if remove_ref_doc_node:
            self._remove_ref_doc_node(doc_id)

        delete_success = self._kvstore.delete(doc_id, collection=self._node_collection)
        _ = self._kvstore.delete(doc_id, collection=self._metadata_collection)

        if not delete_success and raise_error:
            raise ValueError(f"doc_id {doc_id} not found.")

    def delete_ref_doc(self, ref_doc_id: str, raise_error: bool = True) -> None:
        """Delete a ref_doc and all it's associated nodes."""
        ref_doc_info = self.get_ref_doc_info(ref_doc_id)
        if ref_doc_info is None:
            if raise_error:
                raise ValueError(f"ref_doc_id {ref_doc_id} not found.")
            else:
                return

        for doc_id in ref_doc_info.doc_ids:
            self.delete_document(doc_id, raise_error=False, remove_ref_doc_node=False)
        self._kvstore.delete(ref_doc_id, collection=self._ref_doc_collection)

    def set_document_hash(self, doc_id: str, doc_hash: str) -> None:
        """Set the hash for a given doc_id."""
        metadata = {"doc_hash": doc_hash}
        self._kvstore.put(doc_id, metadata, collection=self._metadata_collection)

    def get_document_hash(self, doc_id: str) -> Optional[str]:
        """Get the stored hash for a document, if it exists."""
        metadata = self._kvstore.get(doc_id, collection=self._metadata_collection)
        if metadata is not None:
            return metadata.get("doc_hash", None)
        else:
            return None
