"""Tencent Vector store index.

An index that that is built with Tencent Vector Database.

"""
import json
from typing import Any, Dict, List, Optional

from llama_index.schema import BaseNode, NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import DEFAULT_DOC_ID_KEY, DEFAULT_TEXT_KEY

DEFAULT_USERNAME = "root"
DEFAULT_DATABASE_NAME = "llama_default_database"
DEFAULT_COLLECTION_NAME = "llama_default_collection"
DEFAULT_COLLECTION_DESC = "Collection for llama index"
DEFAULT_TIMEOUT: int = 30

DEFAULT_SHARD = 1
DEFAULT_REPLICAS = 2
DEFAULT_INDEX_TYPE = "HNSW"
DEFAULT_METRIC_TYPE = "COSINE"

DEFAULT_HNSW_M = 16
DEFAULT_HNSW_EF = 200
DEFAULT_IVF_NLIST = 128
DEFAULT_IVF_PQ_M = 16

FIELD_ID: str = "id"
FIELD_VECTOR: str = "vector"
FIELD_METADATA: str = "metadata"

READ_CONSISTENCY = "read_consistency"
READ_STRONG_CONSISTENCY = "strongConsistency"
READ_EVENTUAL_CONSISTENCY = "eventualConsistency"
READ_CONSISTENCY_VALUES = "['strongConsistency', 'eventualConsistency']"

VALUE_NONE_ERROR = "Parameter `{}` can not be None."
VALUE_RANGE_ERROR = "The value of parameter `{}` must be within {}."
NOT_SUPPORT_INDEX_TYPE_ERROR = (
    "Unsupported index type: `{}`, supported index types are {}"
)
NOT_SUPPORT_METRIC_TYPE_ERROR = (
    "Unsupported metric type: `{}`, supported metric types are {}"
)


def _try_import() -> None:
    try:
        import tcvectordb  # noqa
    except ImportError:
        raise ImportError(
            "`tcvectordb` package not found, please run `pip install tcvectordb`"
        )


class FilterField:
    name: str
    data_type: str = "string"

    def __init__(self, name: str, data_type: str = "string"):
        self.name = name
        self.data_type = "string" if data_type is None else data_type

    def match_value(self, value: Any) -> bool:
        if self.data_type == "uint64":
            return isinstance(value, int)
        else:
            return isinstance(value, str)

    def to_vdb_filter(self) -> Any:
        from tcvectordb.model.enum import FieldType, IndexType
        from tcvectordb.model.index import FilterIndex

        return FilterIndex(
            name=self.name,
            field_type=FieldType(self.data_type),
            index_type=IndexType.FILTER,
        )


class CollectionParams:
    r"""Tencent vector DB Collection params.
    See the following documentation for details:
    https://cloud.tencent.com/document/product/1709/95826.

    Args:
        dimension int: The dimension of vector.
        shard int: The number of shards in the collection.
        replicas int: The number of replicas in the collection.
        index_type (Optional[str]): HNSW, IVF_FLAT, IVF_PQ, IVF_SQ8... Default value is "HNSW"
        metric_type (Optional[str]): L2, COSINE, IP. Default value is "COSINE"
        drop_exists (Optional[bool]): Delete the existing Collection. Default value is False.
        vector_params (Optional[Dict]):
          if HNSW set parameters: `M` and `efConstruction`, for example `{'M': 16, efConstruction: 200}`
          if IVF_FLAT or IVF_SQ8 set parameter: `nlist`
          if IVF_PQ set parameters: `M` and `nlist`
          default is HNSW
        filter_fields: Optional[List[FilterField]]: Set the fields for filtering
          for example: [FilterField(name='author'), FilterField(name='age', data_type=uint64)]
          This can be used when calling the query method：
             store.add([
                TextNode(..., metadata={'age'=23, 'name'='name1'})
            ])
             ...
             query = VectorStoreQuery(...)
             store.query(query, filter="age > 20 and age < 40 and name in (\"name1\", \"name2\")")
    """

    def __init__(
        self,
        dimension: int,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        collection_description: str = DEFAULT_COLLECTION_DESC,
        shard: int = DEFAULT_SHARD,
        replicas: int = DEFAULT_REPLICAS,
        index_type: str = DEFAULT_INDEX_TYPE,
        metric_type: str = DEFAULT_METRIC_TYPE,
        drop_exists: Optional[bool] = False,
        vector_params: Optional[Dict] = None,
        filter_fields: Optional[List[FilterField]] = [],
    ):
        self.collection_name = collection_name
        self.collection_description = collection_description
        self.dimension = dimension
        self.shard = shard
        self.replicas = replicas
        self.index_type = index_type
        self.metric_type = metric_type
        self.vector_params = vector_params
        self.drop_exists = drop_exists
        self.filter_fields = filter_fields or []


class TencentVectorDB(VectorStore):
    """Tencent Vector Store.

    In this vector store, embeddings and docs are stored within a Collection.
    If the Collection does not exist, it will be automatically created.

    In order to use this you need to have a database instance.
    See the following documentation for details:
    https://cloud.tencent.com/document/product/1709/94951

    Args:
        url (Optional[str]): url of Tencent vector database
        username (Optional[str]): The username for Tencent vector database. Default value is "root"
        key (Optional[str]): The Api-Key for Tencent vector database
        collection_params (Optional[CollectionParams]): The collection parameters for vector database

    """

    stores_text: bool = True
    filter_fields: List[FilterField] = []

    def __init__(
        self,
        url: str,
        key: str,
        username: str = DEFAULT_USERNAME,
        database_name: str = DEFAULT_DATABASE_NAME,
        read_consistency: str = READ_EVENTUAL_CONSISTENCY,
        collection_params: CollectionParams = CollectionParams(dimension=1536),
        batch_size: int = 512,
        **kwargs: Any,
    ):
        """Init params."""
        self._init_client(url, username, key, read_consistency)
        self._create_database_if_not_exists(database_name)
        self._create_collection(database_name, collection_params)
        self._init_filter_fields()
        self.batch_size = batch_size

    def _init_filter_fields(self) -> None:
        fields = vars(self.collection).get("indexes", [])
        for field in fields:
            if field["fieldName"] not in [FIELD_ID, DEFAULT_DOC_ID_KEY, FIELD_VECTOR]:
                self.filter_fields.append(
                    FilterField(name=field["fieldName"], data_type=field["fieldType"])
                )

    @classmethod
    def class_name(cls) -> str:
        return "TencentVectorDB"

    @classmethod
    def from_params(
        cls,
        url: str,
        key: str,
        username: str = DEFAULT_USERNAME,
        database_name: str = DEFAULT_DATABASE_NAME,
        read_consistency: str = READ_EVENTUAL_CONSISTENCY,
        collection_params: CollectionParams = CollectionParams(dimension=1536),
        batch_size: int = 512,
        **kwargs: Any,
    ) -> "TencentVectorDB":
        _try_import()
        return cls(
            url=url,
            username=username,
            key=key,
            database_name=database_name,
            read_consistency=read_consistency,
            collection_params=collection_params,
            batch_size=batch_size,
            **kwargs,
        )

    def _init_client(
        self, url: str, username: str, key: str, read_consistency: str
    ) -> None:
        import tcvectordb
        from tcvectordb.model.enum import ReadConsistency

        if read_consistency is None:
            raise ValueError(VALUE_RANGE_ERROR.format(read_consistency))

        try:
            v_read_consistency = ReadConsistency(read_consistency)
        except ValueError:
            raise ValueError(
                VALUE_RANGE_ERROR.format(READ_CONSISTENCY, READ_CONSISTENCY_VALUES)
            )

        self.tencent_client = tcvectordb.VectorDBClient(
            url=url,
            username=username,
            key=key,
            read_consistency=v_read_consistency,
            timeout=DEFAULT_TIMEOUT,
        )

    def _create_database_if_not_exists(self, database_name: str) -> None:
        db_list = self.tencent_client.list_databases()

        if database_name in [db.database_name for db in db_list]:
            self.database = self.tencent_client.database(database_name)
        else:
            self.database = self.tencent_client.create_database(database_name)

    def _create_collection(
        self, database_name: str, collection_params: CollectionParams
    ) -> None:
        import tcvectordb

        collection_name: str = self._compute_collection_name(
            database_name, collection_params
        )
        collection_description = collection_params.collection_description

        if collection_params is None:
            raise ValueError(VALUE_NONE_ERROR.format("collection_params"))

        try:
            self.collection = self.database.describe_collection(collection_name)
            if collection_params.drop_exists:
                self.database.drop_collection(collection_name)
                self._create_collection_in_db(
                    collection_name, collection_description, collection_params
                )
        except tcvectordb.exceptions.VectorDBException:
            self._create_collection_in_db(
                collection_name, collection_description, collection_params
            )

    @staticmethod
    def _compute_collection_name(
        database_name: str, collection_params: CollectionParams
    ) -> str:
        if database_name == DEFAULT_DATABASE_NAME:
            return collection_params.collection_name
        if collection_params.collection_name != DEFAULT_COLLECTION_NAME:
            return collection_params.collection_name
        else:
            return database_name + "_" + DEFAULT_COLLECTION_NAME

    def _create_collection_in_db(
        self,
        collection_name: str,
        collection_description: str,
        collection_params: CollectionParams,
    ) -> None:
        from tcvectordb.model.enum import FieldType, IndexType
        from tcvectordb.model.index import FilterIndex, Index, VectorIndex

        index_type = self._get_index_type(collection_params.index_type)
        metric_type = self._get_metric_type(collection_params.metric_type)
        index_param = self._get_index_params(index_type, collection_params)
        index = Index(
            FilterIndex(
                name=FIELD_ID,
                field_type=FieldType.String,
                index_type=IndexType.PRIMARY_KEY,
            ),
            FilterIndex(
                name=DEFAULT_DOC_ID_KEY,
                field_type=FieldType.String,
                index_type=IndexType.FILTER,
            ),
            VectorIndex(
                name=FIELD_VECTOR,
                dimension=collection_params.dimension,
                index_type=index_type,
                metric_type=metric_type,
                params=index_param,
            ),
        )
        for field in collection_params.filter_fields:
            index.add(field.to_vdb_filter())

        self.collection = self.database.create_collection(
            name=collection_name,
            shard=collection_params.shard,
            replicas=collection_params.replicas,
            description=collection_description,
            index=index,
        )

    @staticmethod
    def _get_index_params(index_type: Any, collection_params: CollectionParams) -> None:
        from tcvectordb.model.enum import IndexType
        from tcvectordb.model.index import (
            HNSWParams,
            IVFFLATParams,
            IVFPQParams,
            IVFSQ4Params,
            IVFSQ8Params,
            IVFSQ16Params,
        )

        vector_params = (
            {}
            if collection_params.vector_params is None
            else collection_params.vector_params
        )

        if index_type == IndexType.HNSW:
            return HNSWParams(
                m=vector_params.get("M", DEFAULT_HNSW_M),
                efconstruction=vector_params.get("efConstruction", DEFAULT_HNSW_EF),
            )
        elif index_type == IndexType.IVF_FLAT:
            return IVFFLATParams(nlist=vector_params.get("nlist", DEFAULT_IVF_NLIST))
        elif index_type == IndexType.IVF_PQ:
            return IVFPQParams(
                m=vector_params.get("M", DEFAULT_IVF_PQ_M),
                nlist=vector_params.get("nlist", DEFAULT_IVF_NLIST),
            )
        elif index_type == IndexType.IVF_SQ4:
            return IVFSQ4Params(nlist=vector_params.get("nlist", DEFAULT_IVF_NLIST))
        elif index_type == IndexType.IVF_SQ8:
            return IVFSQ8Params(nlist=vector_params.get("nlist", DEFAULT_IVF_NLIST))
        elif index_type == IndexType.IVF_SQ16:
            return IVFSQ16Params(nlist=vector_params.get("nlist", DEFAULT_IVF_NLIST))
        return None

    @staticmethod
    def _get_index_type(index_type_value: str) -> Any:
        from tcvectordb.model.enum import IndexType

        index_type_value = index_type_value or IndexType.HNSW
        try:
            return IndexType(index_type_value)
        except ValueError:
            support_index_types = [d.value for d in IndexType.__members__.values()]
            raise ValueError(
                NOT_SUPPORT_INDEX_TYPE_ERROR.format(
                    index_type_value, support_index_types
                )
            )

    @staticmethod
    def _get_metric_type(metric_type_value: str) -> Any:
        from tcvectordb.model.enum import MetricType

        metric_type_value = metric_type_value or MetricType.COSINE
        try:
            return MetricType(metric_type_value.upper())
        except ValueError:
            support_metric_types = [d.value for d in MetricType.__members__.values()]
            raise ValueError(
                NOT_SUPPORT_METRIC_TYPE_ERROR.format(
                    metric_type_value, support_metric_types
                )
            )

    @property
    def client(self) -> Any:
        """Get client."""
        return self.tencent_client

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        from tcvectordb.model.document import Document

        ids = []
        entries = []
        for node in nodes:
            document = Document(id=node.node_id, vector=node.get_embedding())
            if node.ref_doc_id is not None:
                document.__dict__[DEFAULT_DOC_ID_KEY] = node.ref_doc_id
            if node.metadata is not None:
                document.__dict__[FIELD_METADATA] = json.dumps(node.metadata)
                for field in self.filter_fields:
                    v = node.metadata.get(field.name)
                    if field.match_value(v):
                        document.__dict__[field.name] = v
            if isinstance(node, TextNode) and node.text is not None:
                document.__dict__[DEFAULT_TEXT_KEY] = node.text

            entries.append(document)
            ids.append(node.node_id)

            if len(entries) >= self.batch_size:
                self.collection.upsert(
                    documents=entries, build_index=True, timeout=DEFAULT_TIMEOUT
                )
                entries = []

        if len(entries) > 0:
            self.collection.upsert(
                documents=entries, build_index=True, timeout=DEFAULT_TIMEOUT
            )

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id or ids.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        if ref_doc_id is None or len(ref_doc_id) == 0:
            return

        from tcvectordb.model.document import Filter

        delete_ids = ref_doc_id if isinstance(ref_doc_id, list) else [ref_doc_id]
        self.collection.delete(filter=Filter(Filter.In(DEFAULT_DOC_ID_KEY, delete_ids)))

    def query_by_ids(self, ids: List[str]) -> List[Dict]:
        return self.collection.query(document_ids=ids, limit=len(ids))

    def truncate(self) -> None:
        self.database.truncate_collection(self.collection.collection_name)

    def describe_collection(self) -> Any:
        return self.database.describe_collection(self.collection.collection_name)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): contains
                query_embedding (List[float]): query embedding
                similarity_top_k (int): top k most similar nodes
                doc_ids (Optional[List[str]]): filter by doc_id
                filters (Optional[MetadataFilters]): filter result
            kwargs.filter (Optional[str|Filter]):

            if `kwargs` in kwargs:
               using filter: `age > 20 and author in (...) and ...`
            elif query.filters:
               using filter: " and ".join([f'{f.key} = "{f.value}"' for f in query.filters.filters])
            elif query.doc_ids:
               using filter: `doc_id in (query.doc_ids)`
        """
        search_filter = self._to_vdb_filter(query, **kwargs)
        results = self.collection.search(
            vectors=[query.query_embedding],
            limit=query.similarity_top_k,
            retrieve_vector=True,
            output_fields=query.output_fields,
            filter=search_filter,
        )
        if len(results) == 0:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        nodes = []
        similarities = []
        ids = []
        for doc in results[0]:
            ids.append(doc.get(FIELD_ID))
            similarities.append(doc.get("score"))

            meta_str = doc.get(FIELD_METADATA)
            meta = {} if meta_str is None else json.loads(meta_str)
            doc_id = doc.get(DEFAULT_DOC_ID_KEY)

            node = TextNode(
                id_=doc.get(FIELD_ID),
                text=doc.get(DEFAULT_TEXT_KEY),
                embedding=doc.get(FIELD_VECTOR),
                metadata=meta,
            )
            if doc_id is not None:
                node.relationships = {
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc_id)
                }

            nodes.append(node)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    @staticmethod
    def _to_vdb_filter(query: VectorStoreQuery, **kwargs: Any) -> Any:
        from tcvectordb.model.document import Filter

        search_filter = None
        if "filter" in kwargs:
            search_filter = kwargs.pop("filter")
            search_filter = (
                search_filter
                if type(search_filter) is Filter
                else Filter(search_filter)
            )
        elif query.filters is not None:
            search_filter = " and ".join(
                [f'{f.key} = "{f.value}"' for f in query.filters.filters]
            )
            search_filter = Filter(search_filter)
        elif query.doc_ids is not None:
            search_filter = Filter(Filter.In(DEFAULT_DOC_ID_KEY, query.doc_ids))

        return search_filter
