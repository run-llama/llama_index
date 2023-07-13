"""Milvus vector store index.

An index that is built within Milvus.

"""
import logging
from typing import Any, List, Optional
from uuid import uuid4

from llama_index.schema import MetadataMode, NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.types import (
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
    DEFAULT_DOC_ID_KEY,
    DEFAULT_EMBEDDING_KEY,
    DEFAULT_TEXT_KEY,
)

logger = logging.getLogger(__name__)


class MilvusVectorStore(VectorStore):
    """The Milvus Vector Store.

    In this vector store we store the text, its embedding and
    a few pieces of its metadata in a Milvus collection. This implemnetation
    allows the use of an already existing collection.
    It also supports creating a new one if the collection doesnt
    exist or if `overwrite` is set to True.

    Args:
        collection_name (str, optional): The name of the collection where data will be
            stored. Defaults to "llamalection".
        index_params (dict, optional): The index parameters for Milvus, if none are
            provided an HNSW index will be used. Defaults to None.
        search_params (dict, optional): The search parameters for a Milvus query.
            If none are provided, default params will be generated. Defaults to None.
        dim (int, optional): The dimension of the embeddings. If it is not provided,
            collection creation will be done on first insert. Defaults to None.
        host (str, optional): The host address of Milvus. Defaults to "localhost".
        port (int, optional): The port of Milvus. Defaults to 19530.
        user (str, optional): The username for RBAC. Defaults to "".
        password (str, optional): The password for RBAC. Defaults to "".
        use_secure (bool, optional): Use https. Required for Zilliz Cloud.
            Defaults to False.
        overwrite (bool, optional): Whether to overwrite existing collection with same
            name. Defaults to False.

    Raises:
        ImportError: Unable to import `pymilvus`.
        MilvusException: Error communicating with Milvus, more can be found in logging
            under Debug.

    Returns:
        MilvusVectorstore: Vectorstore that supports add, delete, and query.
    """

    stores_text: bool = True
    stores_node: bool = True

    def __init__(
        self,
        collection_name: str = "llamalection",
        index_params: Optional[dict] = None,
        search_params: Optional[dict] = None,
        dim: Optional[int] = None,
        embedding_field: str = DEFAULT_EMBEDDING_KEY,
        text_field: str = DEFAULT_TEXT_KEY,
        doc_id_field: str = DEFAULT_DOC_ID_KEY,
        consistency_level: Optional[str] = "Strong",
        host: str = "localhost",
        port: int = 19530,
        user: str = "",
        password: str = "",
        use_secure: bool = False,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        import_err_msg = (
            "`pymilvus` package not found, please run `pip install pymilvus`"
        )
        try:
            import pymilvus  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        from pymilvus import Collection, MilvusException, utility

        self.collection_name = collection_name
        self.search_params = search_params
        self.index_params = index_params
        self.dim = dim
        self.embedding_field = embedding_field
        self.text_field = text_field
        self.doc_id_field = doc_id_field
        self.consistency_level = consistency_level
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.use_secure = use_secure
        self.overwrite = overwrite

        self.default_search_params = {
            "IVF_FLAT": {"metric_type": "IP", "params": {"nprobe": 10}},
            "IVF_SQ8": {"metric_type": "IP", "params": {"nprobe": 10}},
            "IVF_PQ": {"metric_type": "IP", "params": {"nprobe": 10}},
            "HNSW": {"metric_type": "IP", "params": {"ef": 10}},
            "RHNSW_FLAT": {"metric_type": "IP", "params": {"ef": 10}},
            "RHNSW_SQ": {"metric_type": "IP", "params": {"ef": 10}},
            "RHNSW_PQ": {"metric_type": "IP", "params": {"ef": 10}},
            "IVF_HNSW": {"metric_type": "IP", "params": {"nprobe": 10, "ef": 10}},
            "ANNOY": {"metric_type": "IP", "params": {"search_k": 10}},
            "AUTOINDEX": {"metric_type": "IP", "params": {}},
        }

        # Generate a connection alias
        self._create_connection_alias()

        # Figure out if there is already a created collection
        if utility.has_collection(self.collection_name, using=self.alias):
            self.collection = Collection(self.collection_name, using=self.alias)
        else:
            self.collection = None

        # If a collection already exists and we are overwriting, delete it
        if self.collection is not None and self.overwrite is True:
            try:
                utility.drop_collection(self.collection_name, using=self.alias)
                self.collection = None
                logger.debug(
                    f"Successfully removed old collection: {self.collection_name}"
                )
            except MilvusException as e:
                logger.debug(f"Failed to remove old collection: {self.collection_name}")
                raise e

        # If there is no collection and a dim is provided, we can create a collection
        if self.collection is None and self.dim is not None:
            self._create_collection()

        # If there is a collection and no index exists on it, create an index
        if self.collection is not None and len(self.collection.indexes) == 0:
            self._create_index()
        # If using an existing index and no search params were provided,
        #   generate the correct params
        elif self.collection is not None and self.search_params is None:
            self._create_search_params()

        # If there is a collection with an index, make sure its loaded
        if self.collection is not None and len(self.collection.indexes) != 0:
            self.collection.load()

    def _create_connection_alias(self) -> None:
        from pymilvus import connections

        self.alias = None

        # Attempt to reuse an open connection
        for x in connections.list_connections():
            addr = connections.get_connection_addr(x[0])
            tmp_user = "" if self.user is None else self.user
            if (
                x[1]
                and ("address" in addr)
                and (addr["address"] == "{}:{}".format(self.host, self.port))
                and ("user" in addr)
                and (addr["user"] == tmp_user)
            ):
                self.alias = x[0]
                logger.debug(f"Using previous connection: {self.alias}")
                break

        # Connect to the Milvus instance using the passed in Environment variables
        if self.alias is None:
            self.alias = uuid4().hex
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=self.port,
                user=self.user,  # type: ignore
                password=self.password,  # type: ignore
                secure=self.use_secure,
            )
            logger.debug(f"Creating new connection: {self.alias}")

    def _create_collection(self) -> None:
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            MilvusException,
        )

        try:
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    description="Unique ID",
                    is_primary=True,
                    auto_id=False,
                    max_length=65535,
                ),
                FieldSchema(
                    name=self.doc_id_field,
                    dtype=DataType.VARCHAR,
                    description="Source document ID",
                    max_length=65535,
                ),
                FieldSchema(
                    name=self.text_field,
                    dtype=DataType.VARCHAR,
                    description="The embedding vector",
                    max_length=65535,
                ),
                FieldSchema(
                    name=self.embedding_field,
                    dtype=DataType.FLOAT_VECTOR,
                    description="The embedding vector",
                    dim=self.dim,
                ),
                FieldSchema(
                    name="node",
                    dtype=DataType.VARCHAR,
                    description="The node content",
                    max_length=65535,
                ),
            ]

            col_schema = CollectionSchema(fields=fields)
            self.collection = Collection(
                self.collection_name,
                col_schema,
                using=self.alias,
                consistency_level=self.consistency_level,
            )
            logger.debug(
                f"Successfully created a new collection: {self.collection_name}"
            )

        except MilvusException as e:
            logger.debug(f"Failure to create a new collection: {self.collection_name}")
            raise e

    def _create_index(self) -> None:
        from pymilvus import MilvusException

        try:
            # If no index params, use a default HNSW based one
            if self.index_params is None:
                self.index_params = {
                    "metric_type": "IP",
                    "index_type": "HNSW",
                    "params": {"M": 8, "efConstruction": 64},
                }
            assert self.index_params is not None

            try:
                self.collection.create_index(
                    self.embedding_field,
                    index_params=self.index_params,
                    using=self.alias,
                )
            # If default did not work, most likely on Zilliz Cloud
            except MilvusException:
                # Attempt creating autoindex
                self.index_params = {
                    "metric_type": "IP",
                    "index_type": "AUTOINDEX",
                    "params": {},
                }
                self.collection.create_index(
                    self.embedding_field,
                    index_params=self.index_params,
                    using=self.alias,
                )

            # If search params dont exist already, grab the default
            if self.search_params is None:
                self.search_params = self.default_search_params[
                    self.index_params["index_type"]
                ]
            logger.debug(
                f"Successfully created an index on collection: {self.collection_name}"
            )

        except MilvusException as e:
            logger.debug(
                f"Failed to create an index on collection: {self.collection_name}"
            )
            raise e

    def _create_search_params(self) -> None:
        index = self.collection.indexes[0]._index_params
        self.search_params = self.default_search_params[index["index_type"]]
        self.search_params["metric_type"] = index["metric_type"]

    @property
    def client(self) -> Any:
        """Get client."""
        return self.collection

    def add(self, embedding_results: List[NodeWithEmbedding]) -> List[str]:
        """Add the embeddings and their nodes into Milvus.

        Args:
            embedding_results (List[NodeWithEmbedding]): The embeddings and their data
                to insert.

        Raises:
            MilvusException: Failed to insert data.

        Returns:
            List[str]: List of ids inserted.
        """
        from pymilvus import MilvusException

        # If the collection doesnt exist yet, create the collection, index, and load it
        if self.collection is None and len(embedding_results) != 0:
            self.dim = len(embedding_results[0].embedding)
            self._create_collection()
            self._create_index()
            assert self.collection is not None
            self.collection.load()

        elif len(embedding_results) == 0:
            return []

        ids = []
        doc_ids = []
        texts = []
        embeddings = []
        nodes = []

        # Process that data we are going to insert
        for result in embedding_results:
            ids.append(result.id)
            doc_ids.append(result.ref_doc_id)
            texts.append(result.node.get_content(metadata_mode=MetadataMode.NONE))
            embeddings.append(result.embedding)

            # Store node without text
            metadata = node_to_metadata_dict(result.node, remove_text=True)
            nodes.append(metadata["_node_content"])

        try:
            # Insert the data into milvus
            self.collection.insert([ids, doc_ids, texts, embeddings, nodes])
            logger.debug(
                f"Successfully inserted embeddings into: {self.collection_name} "
                f"Num Inserted: {len(ids)}"
            )
        except MilvusException as e:
            logger.debug(f"Failed to insert embeddings into: {self.collection_name}")
            raise e
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        Raises:
            MilvusException: Failed to delete the doc.
        """
        from pymilvus import MilvusException

        if self.collection is None:
            return

        # Adds ability for multiple doc delete in future.
        doc_ids: List[str]
        if type(ref_doc_id) != list:
            doc_ids = [ref_doc_id]
        else:
            doc_ids = ref_doc_id  # type: ignore

        try:
            # Begin by querying for the primary keys to delete
            doc_ids = ['"' + entry + '"' for entry in doc_ids]
            entries = self.collection.query(
                f"{self.doc_id_field} in [{','.join(doc_ids)}]"
            )
            ids = [entry["id"] for entry in entries]
            ids = ['"' + entry + '"' for entry in ids]
            self.collection.delete(f"id in [{','.join(ids)}]")
            logger.debug(f"Successfully deleted embedding with doc_id: {doc_ids}")
        except MilvusException as e:
            logger.debug(f"Unsuccessfully deleted embedding with doc_id: {doc_ids}")
            raise e

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes
            doc_ids (Optional[List[str]]): list of doc_ids to filter by
            output_fields (Optional[List[str]]): list of fields to return
            embedding_field (Optional[str]): name of embedding field
        """
        from pymilvus import MilvusException

        if self.collection is None:
            raise ValueError("Milvus instance not initialized.")

        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise ValueError(f"Milvus does not support {query.mode} yet.")

        if query.filters is not None:
            raise ValueError("Metadata filters not implemented for Milvus yet.")

        expr: Optional[str] = None
        if query.doc_ids is not None and len(query.doc_ids) != 0:
            expr_list = ['"' + entry + '"' for entry in query.doc_ids]
            expr = f"{self.doc_id_field} in [{','.join(expr_list)}]"

        if query.output_fields is None:
            output_fields = [self.doc_id_field, self.text_field]

        if query.embedding_field is None:
            embedding_field = self.embedding_field

        try:
            res = self.collection.search(
                [query.query_embedding],
                embedding_field,
                self.search_params,
                limit=query.similarity_top_k,
                output_fields=output_fields + ["node"],
                expr=expr,
            )
            logger.debug(
                f"Successfully searched embedding in collection: {self.collection_name}"
                f" Num Results: {len(res[0])}"
            )
        except MilvusException:
            # TODO: Legacy support for old dbs
            res = self.collection.search(
                [query.query_embedding],
                "embedding",
                self.search_params,
                limit=query.similarity_top_k,
                output_fields=output_fields,
                expr=expr,
            )
            logger.debug(
                f"Successfully searched embedding in collection: {self.collection_name}"
                f" Num Results: {len(res[0])}"
            )

        nodes = []
        similarities = []
        ids = []

        for hit in res[0]:
            try:
                node = metadata_dict_to_node({"_node_content": hit.entity.get("node")})
                node.text = hit.entity.get(self.text_field)
            except Exception:
                # TODO: Legacy support for old nodes
                node = TextNode(
                    text=hit.entity.get(self.text_field),
                    id_=hit.id,
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(
                            node_id=hit.entity.get(self.doc_id_field)
                        ),
                    },
                )
            nodes.append(node)
            similarities.append(hit.score)
            ids.append(hit.id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
