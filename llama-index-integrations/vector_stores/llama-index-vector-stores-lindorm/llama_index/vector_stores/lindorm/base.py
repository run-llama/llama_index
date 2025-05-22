"""Lindorm Vector Store."""

import asyncio
import uuid
from typing import Any, Dict, Iterable, List, Optional, Union, cast

from llama_index.core.bridge.pydantic import PrivateAttr

from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
    FilterOperator,
    FilterCondition,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from opensearchpy.client import Client as OSClient

IMPORT_OPENSEARCH_PY_ERROR = "Could not import OpenSearch Python SDK. Please install it with `pip install opensearch-py`."
INVALID_HYBRID_QUERY_ERROR = "Please specify the lexical_query for hybrid search."
MATCH_ALL_QUERY = {"match_all": {}}  # type: Dict


class LindormVectorClient:
    """
    Object encapsulating an Lindorm index that has vector search enabled.

    If the index does not yet exist, it is created during init.
    Therefore, the underlying index is assumed to either:
    1) not exist yet or 2) be created due to previous usage of this class.

    Two index types are available: IVFPQ & HNSW. Default: IVFPQ.

    Detailed info for these arguments can be found here:
    https://help.aliyun.com/document_detail/2773371.html

    Args:
        host (str): Elasticsearch compatible host of the lindorm search engine.
        port (int): Port of you lindorm instance.
        username (str): Username of your lindorm instance.
        password (str): Password of your lindorm instance.
        index (str): Name of the index.
        dimension (int): Dimension of the vector.

    how to obtain an lindorm instance:
    https://alibabacloud.com/help/en/lindorm/latest/create-an-instance

    how to access your lindorm instance:
    https://www.alibabacloud.com/help/en/lindorm/latest/view-endpoints

    run curl commands to connect to and use LindormSearch:
    https://www.alibabacloud.com/help/en/lindorm/latest/connect-and-use-the-search-engine-with-the-curl-command

    Optional Args:
        text_field(str): Document field the text of the document is stored in. Defaults to "content".
        max_chunk_bytes(int): Maximum size of a chunk in bytes; default : 1 * 1024 * 1024.
        os_client(OSClient): opensearch_client; default : None.

    Optional Keyword Args to construct method of mapping:
        method_name(str): "ivfpq","hnsw"; default: "ivfpq".
        engine(str): "lvector"; default: "lvector".
        space_type(str): "l2", "cosinesimil", "innerproduct"; default: "l2"
        vector_field(str): Document field embeddings are stored in. default: "vector_field".

    Optional Keyword Args for lindorm search extension setting:
        filter_type (str): filter type for lindorm search, pre_filter or post_filter; default: post_filter.
        nprobe (str): number of cluster units to query; between 1 and method.parameters.nlist.
            No default value.
        reorder_factor (str): reorder_factor for lindorm search; between 1 and 200; default: 10.

    Optional Keyword Args for IVFPQ:
        m(int): Number of subspaces. Between 2 and 32768; default: 16.
        nlist(int): Number of cluster centersdefault. Between 2 and 1000000; default: 10000.
        centroids_use_hnsw(bool): Whether to use the HNSW algorithm when searching for cluster centers; default: True.
        centroids_hnsw_m: Between 1 and 100; default: 16.
        centroids_hnsw_ef_search(int): Size of the dynamic list used during k-NN searches. Higher values.
            lead to more accurate but slower searches; default: 100.
        centroids_hnsw_ef_construct(int): Size of the dynamic list used during k-NN graph creation.
            Higher values lead to more accurate graph but slower indexing speed; default: 100.

    Optional Keyword Args for HNSW:
        m(int): maximum number of outgoing edges in each layer of the graph. Between 1 and 100; default: 16.
        ef_construction(int): Length of the dynamic list when the index is built. Between 1 and 1000; default: 100.

    """

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        index: str,
        dimension: int,
        text_field: str = "content",
        max_chunk_bytes: int = 1 * 1024 * 1024,
        os_client: Optional[OSClient] = None,
        **kwargs: Any,
    ):
        """Init params."""
        method_name = kwargs.get("method_name", "ivfpq")
        engine = kwargs.get("engine", "lvector")
        space_type = kwargs.get("space_type", "l2")
        vector_field = kwargs.get("vector_field", "vector_field")
        filter_type = kwargs.get("filter_type", "post_filter")
        nprobe = kwargs.get("nprobe", "1")
        reorder_factor = kwargs.get("reorder_factor", "10")

        if filter_type not in ["post_filter", "pre_filter"]:
            raise ValueError(
                f"Unsupported filter type: {filter_type}, only post_filter and pre_filter are suopported now."
            )

        # initialize parameters
        if method_name == "ivfpq":
            m = kwargs.get("m", dimension)
            nlist = kwargs.get("nlist", 10000)
            centroids_use_hnsw = kwargs.get("centroids_use_hnsw", True)
            centroids_hnsw_m = kwargs.get("centroids_hnsw_m", 16)
            centroids_hnsw_ef_construct = kwargs.get("centroids_hnsw_ef_construct", 100)
            centroids_hnsw_ef_search = kwargs.get("centroids_hnsw_ef_search", 100)
            parameters = {
                "m": m,
                "nlist": nlist,
                "centroids_use_hnsw": centroids_use_hnsw,
                "centroids_hnsw_m": centroids_hnsw_m,
                "centroids_hnsw_ef_construct": centroids_hnsw_ef_construct,
                "centroids_hnsw_ef_search": centroids_hnsw_ef_search,
            }
        elif method_name == "hnsw":
            m = kwargs.get("m", 16)
            ef_construction = kwargs.get("ef_construction", 100)
            parameters = {"m": m, "ef_construction": ef_construction}
        else:
            raise RuntimeError(f"unexpected method_name: {method_name}")

        self._vector_field = vector_field
        self._filter_type = filter_type
        self._nprobe = nprobe
        self._reorder_factor = reorder_factor

        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._dimension = dimension
        self._index = index
        self._text_field = text_field
        self._max_chunk_bytes = max_chunk_bytes

        # initialize mapping
        mapping = {
            "settings": {"index": {"number_of_shards": 4, "knn": True}},
            "mappings": {
                "_source": {"excludes": [vector_field]},
                "properties": {
                    vector_field: {
                        "type": "knn_vector",
                        "dimension": dimension,
                        "data_type": "float",
                        "method": {
                            "engine": engine,
                            "name": method_name,
                            "space_type": space_type,
                            "parameters": parameters,
                        },
                    },
                },
            },
        }

        self._os_client = os_client or self._get_async_lindorm_search_client(
            self._host, self._port, self._username, self._password, **kwargs
        )
        not_found_error = self._import_not_found_error()

        event_loop = asyncio.get_event_loop()

        try:
            event_loop.run_until_complete(
                self._os_client.indices.get(index=self._index)
            )
        except not_found_error:
            event_loop.run_until_complete(
                self._os_client.indices.create(index=self._index, body=mapping)
            )
            event_loop.run_until_complete(
                self._os_client.indices.refresh(index=self._index)
            )

    def _import_async_opensearch(self) -> Any:
        """Import OpenSearch Python SDK if available, otherwise raise error."""
        try:
            from opensearchpy import AsyncOpenSearch
        except ImportError:
            raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)
        return AsyncOpenSearch

    def _import_async_bulk(self) -> Any:
        """Import bulk if available, otherwise raise error."""
        try:
            from opensearchpy.helpers import async_bulk
        except ImportError:
            raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)
        return async_bulk

    def _import_not_found_error(self) -> Any:
        """Import not found error if available, otherwise raise error."""
        try:
            from opensearchpy.exceptions import NotFoundError
        except ImportError:
            raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)
        return NotFoundError

    def _get_async_lindorm_search_client(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        time_out: Optional[int] = 100,
        **kwargs: Any,
    ) -> Any:
        """Get lindorm search client through `opensearchpy` base on the lindorm_search_instance, otherwise raise error."""
        try:
            opensearch = self._import_async_opensearch()
            auth = (username, password)
            client = opensearch(
                hosts=[{"host": host, "port": port}],
                http_auth=auth,
                time_out=time_out,
                **kwargs,
            )
        except ValueError as e:
            raise ValueError(
                f"Async Lindorm Search Client string provided is not in proper format. "
                f"Got error: {e} "
            )
        return client

    def _flatten_request(self, request) -> Dict:
        """Flatten metadata in request."""
        if "metadata" in request:
            for key, value in request["metadata"].items():
                request[key] = value
            del request["metadata"]
        return request

    async def _bulk_ingest_embeddings(
        self,
        client: Any,
        index_name: str,
        embeddings: List[List[float]],
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        vector_field: str = "vector_field",
        text_field: str = "content",
        mapping: Optional[Dict] = None,
        max_chunk_bytes: Optional[int] = 1 * 1024 * 1024,
    ) -> List[str]:
        """Async Bulk Ingest Embeddings into given index."""
        if not mapping:
            mapping = {}

        async_bulk = self._import_async_bulk()
        not_found_error = self._import_not_found_error()
        requests = []
        return_ids = []
        mapping = mapping

        try:
            await client.indices.get(index=index_name)
        except not_found_error:
            await client.indices.create(index=index_name, body=mapping)

        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            _id = ids[i] if ids else str(uuid.uuid4())
            request = {
                "_op_type": "index",
                "_index": index_name,
                vector_field: embeddings[i],
                text_field: text,
                "metadata": metadata,
                "_id": _id,
            }
            # Flatten metadata in request
            request = self._flatten_request(request)
            requests.append(request)
            return_ids.append(_id)
        await async_bulk(client, requests, max_chunk_bytes=max_chunk_bytes)
        await client.indices.refresh(index=index_name)
        return return_ids

    def _default_approximate_search_query(
        self,
        query_vector: List[float],
        k: int,
        nprobe: str,
        reorder_factor: str,
        vector_field: str = "vector_field",
    ) -> Dict:
        """
        For Approximate k-NN Search, this is the default query.

        Args:
            query_vector(List[float]): Vector embedding to query.
            k(int): Maximum number of results. default: 4.
            nprobe (str): number of cluster units to query; between 1 and method.parameters.nlist.
                No default value.
            reorder_factor (str): reorder_factor for lindorm search; between 1 and 200; default: 10.

        Optional Args:
            vector_field(str): Document field embeddings are stored in. default: "vector_field".

        Return:
            A dictionary representing the query.

        """
        return {
            "size": k,
            "query": {"knn": {vector_field: {"vector": query_vector, "k": k}}},
            "ext": {"lvector": {"nprobe": nprobe, "reorder_factor": reorder_factor}},
        }

    def _search_query_with_filter(
        self,
        query_vector: List[float],
        k: int,
        filter_type: str,
        nprobe: str,
        reorder_factor: str,
        vector_field: str = "vector_field",
        filter: Union[Dict, List, None] = None,
    ) -> Dict:
        """
        Construct search query with pre-filter or post-filter.

        Args:
            query_vector(List[float]): Vector embedding to query.
            k(int): Maximum number of results. default: 4.
            filter_type(str): filter_type for lindorm search, pre_filter and post_filter are supported;
                default: "post_filter".
            nprobe (str): number of cluster units to query; between 1 and method.parameters.nlist.
                No default value.
            reorder_factor (str): reorder_factor for lindorm search; between 1 and 200; default: 10.
            vector_field(str): Document field embeddings are stored in. default: "vector_field".
            filter(Union[Dict, List, None]): filter for lindorm search. default: None.

        Returns:
            A dictionary representing the query.

        """
        if not filter:
            filter = MATCH_ALL_QUERY
        return {
            "size": k,
            "query": {
                "knn": {
                    vector_field: {"vector": query_vector, "filter": filter, "k": k}
                }
            },
            "ext": {
                "lvector": {
                    "filter_type": filter_type,
                    "nprobe": nprobe,
                    "reorder_factor": reorder_factor,
                }
            },
        }

    def _metadatafilter_to_dict(self, filter: MetadataFilter) -> Dict:
        """
        Parse MetadataFilter into a dictionary.

        Args:
            filter (MetadataFilter): A MetadataFilter object.

        Returns:
            dict: A dictionary representing the filter.

        """
        operator = filter.operator

        range_operators = {
            FilterOperator.GTE: "gte",
            FilterOperator.LTE: "lte",
            FilterOperator.GT: "gt",
            FilterOperator.LT: "lt",
        }

        if operator in range_operators:
            filter_dict = {
                "range": {filter.key: {range_operators[operator]: filter.value}}
            }
        elif operator == FilterOperator.EQ:
            filter_dict = {"term": {filter.key: filter.value}}
        else:
            raise ValueError(f"Unsupported filter operator: {operator}")

        return filter_dict

    def _parse_filters(self, filters: Optional[MetadataFilters]) -> Any:
        """
        Parse MetadataFilters into a list of dictionaries.

        Args:
            filters (Optional[MetadataFilters]): An optional MetadataFilters object.

        Returns:
            list: A list of dictionaries. If no filters are provided, an empty list is returned.

        """
        filter_list = []
        if filters is not None:
            for filter in filters.filters:
                filter_list.append(self._metadatafilter_to_dict(filter))
        return filter_list

    def _knn_search_query(
        self,
        vector_field: str,
        query_embedding: List[float],
        k: int,
        filter_type: str,
        nprobe: str,
        reorder_factor: str,
        filters: Optional[MetadataFilters] = None,
    ) -> Dict:
        """
        Do knn search.

        If there are no filters do approx-knn search.
        If there are filters, do an exhaustive exact knn search using filters.

        Note that approximate knn search does not support metadata filting.

        Args:
            query_embedding(List[float]): Vector embedding to query.
            k(int): Maximum number of results.
            filter_type(str): filter_type for lindorm search, pre_filter and post_filter are supported;
                default: "post_filter".
            nprobe (str): number of cluster units to query; between 1 and method.parameters.nlist.
                No default value.
            reorder_factor (str): reorder_factor for lindorm search; between 1 and 200; default: 10.

        Optional Args:
            filters(Optional[MetadataFilters]): Optional filters to apply before the search.
                Supports filter-context queries documented at
                https://opensearch.org/docs/latest/query-dsl/query-filter-context/

        Returns:
            Up to k targets closest to query_embedding.

        """
        filter_list = self._parse_filters(filters)
        if not filters:
            search_query = self._default_approximate_search_query(
                query_vector=query_embedding,
                k=k,
                vector_field=vector_field,
                nprobe=nprobe,
                reorder_factor=reorder_factor,
            )
        else:
            if filters.condition == FilterCondition.AND:
                filter = {"bool": {"must": filter_list}}
            elif filters.condition == FilterCondition.OR:
                filter = {"bool": {"should": filter_list}}
            else:
                # TODO: FilterCondition can also be 'NOT', but llama_index does not support it yet.
                # https://opensearch.org/docs/latest/query-dsl/compound/bool/
                # post_filter = {"bool": {"must_not": filter_list}}
                raise ValueError(f"Unsupported filter condition: {filters.condition}")

            search_query = self._search_query_with_filter(
                query_vector=query_embedding,
                vector_field=vector_field,
                k=k,
                filter=filter,
                nprobe=nprobe,
                reorder_factor=reorder_factor,
                filter_type=filter_type,
            )

        return search_query

    def _hybrid_search_query(
        self,
        text_field: str,
        query_str: str,
        vector_field: str,
        query_embedding: List[float],
        k: int,
        filter_type: str,
        nprobe: str,
        reorder_factor: str,
        filters: Optional[MetadataFilters] = None,
    ) -> Dict:
        """
        Do hybrid search.

        Args:
            text_field(str): Document field to query.
            query_str(str): Query string.
            vector_field(str): Document field embeddings are stored in.
            query_embedding(List[float]): Vector embedding to query.
            k(int): Maximum number of results.
            filter_type(str): filter_type for lindorm search, pre_filter and post_filter are supported;
                default: "post_filter".
            nprobe (str): number of cluster units to query; between 1 and method.parameters.nlist.
                No default value.
            reorder_factor (str): reorder_factor for lindorm search; between 1 and 200; default: 10.

        Optional Args:
            filters(Optional[MetadataFilters]): Optional filters to apply before the search.
                Supports filter-context queries documented at
                https://opensearch.org/docs/latest/query-dsl/query-filter-context/

        Returns:
            Up to k targets closest to query_embedding

        """
        knn_query = self._knn_search_query(
            vector_field=vector_field,
            filter_type=filter_type,
            nprobe=nprobe,
            reorder_factor=reorder_factor,
            query_embedding=query_embedding,
            k=k,
            filters=filters,
        )
        lexical_query = self._lexical_search_query(text_field, query_str, k, filters)

        # Combine knn and lexical search query
        knn_field_query = knn_query["query"]["knn"][vector_field]
        if "filter" not in knn_field_query:
            knn_field_query["filter"] = {"bool": {"must": []}}
        elif "bool" not in knn_field_query["filter"]:
            knn_field_query["filter"]["bool"] = {"must": []}
        elif "must" not in knn_field_query["filter"]["bool"]:
            knn_field_query["filter"]["bool"]["must"] = []

        knn_query["query"]["knn"][vector_field]["filter"]["bool"]["must"].append(
            lexical_query["query"]["bool"]["must"]
        )

        return {
            "size": k,
            "query": knn_query["query"],
            "ext": {
                "lvector": {
                    "filter_type": filter_type,
                    "nprobe": nprobe,
                    "reorder_factor": reorder_factor,
                }
            },
        }

    def _lexical_search_query(
        self,
        text_field: str,
        query_str: str,
        k: int,
        filters: Optional[MetadataFilters] = None,
    ) -> Dict:
        """
        Do lexical search.

        Args:
            text_field(str): Document field to query.
            query_str(str): Query string.
            k(int): Maximum number of results.

        Optional Args:
            filters(Optional[MetadataFilters]): Optional filters to apply before the search.
                Supports filter-context queries documented at
                https://opensearch.org/docs/latest/query-dsl/query-filter-context/

        Returns:
            Up to k targets closest to query_embedding.

        """
        lexical_query = {
            "bool": {"must": {"match": {text_field: {"query": query_str}}}}
        }

        parsed_filters = self._parse_filters(filters)
        if len(parsed_filters) > 0:
            lexical_query["bool"]["filter"] = parsed_filters

        return {
            "size": k,
            "query": lexical_query,
        }

    async def index_results(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        """
        Store results in the index.

        Args:
            nodes (List[BaseNode]): A list of BaseNode objects.

        Returns:
            List[str]: A list of node_ids

        """
        embeddings: List[List[float]] = []
        texts: List[str] = []
        metadatas: List[dict] = []
        ids: List[str] = []
        for node in nodes:
            ids.append(node.node_id)
            embeddings.append(node.get_embedding())
            texts.append(node.get_content(metadata_mode=MetadataMode.NONE))
            metadatas.append(node_to_metadata_dict(node, remove_text=True))

        return await self._bulk_ingest_embeddings(
            self._os_client,
            self._index,
            embeddings,
            texts,
            metadatas=metadatas,
            ids=ids,
            vector_field=self._vector_field,
            text_field=self._text_field,
            mapping=None,
            max_chunk_bytes=self._max_chunk_bytes,
        )

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """
        Deletes nodes corresponding to the given LlamaIndex `Document` ID.

        Args:
            doc_id (str): a LlamaIndex `Document` id.

        """
        search_query = {"query": {"term": {"doc_id.keyword": {"value": doc_id}}}}
        await self._os_client.delete_by_query(index=self._index, body=search_query)

    async def aquery(
        self,
        query_mode: VectorStoreQueryMode,
        query_str: Optional[str],
        query_embedding: List[float],
        k: int,
        filters: Optional[MetadataFilters] = None,
    ) -> VectorStoreQueryResult:
        """
        Do vector search.

        Args:
            query_mode (VectorStoreQueryMode): Query mode.
            query_str (Optional[str]): Query string.
            query_embedding (List[float]): Query embedding.
            k (int): Maximum number of results.

        Optional Args:
            filters(Optional[MetadataFilters]): Optional filters to apply before the search.
                Supports filter-context queries documented at
                https://opensearch.org/docs/latest/query-dsl/query-filter-context/

        Returns:
            VectorStoreQueryResult.

        """
        if query_mode == VectorStoreQueryMode.HYBRID:
            if query_str is None:
                raise ValueError(INVALID_HYBRID_QUERY_ERROR)
            search_query = self._hybrid_search_query(
                text_field=self._text_field,
                query_str=query_str,
                vector_field=self._vector_field,
                query_embedding=query_embedding,
                k=k,
                filters=filters,
                filter_type=self._filter_type,
                nprobe=self._nprobe,
                reorder_factor=self._reorder_factor,
            )
            params = None
        elif query_mode == VectorStoreQueryMode.TEXT_SEARCH:
            search_query = self._lexical_search_query(
                self._text_field, query_str, k, filters=filters
            )
            params = None
        else:
            search_query = self._knn_search_query(
                vector_field=self._vector_field,
                query_embedding=query_embedding,
                k=k,
                filters=filters,
                filter_type=self._filter_type,
                nprobe=self._nprobe,
                reorder_factor=self._reorder_factor,
            )
            params = None

        res = await self._os_client.search(
            index=self._index, body=search_query, _source=True, params=params
        )

        return self._to_query_result(res)

    def _to_query_result(self, res) -> VectorStoreQueryResult:
        """
        Convert Lindorm search result to VectorStoreQueryResult.

        Args:
            res(Dict): Lindorm search result.

        Returns:
            VectorStoreQueryResult.

        """
        nodes = []
        ids = []
        scores = []
        for hit in res["hits"]["hits"]:
            source = hit["_source"]
            node_id = hit["_id"]
            text = source[self._text_field]
            metadata = source.get("metadata", None)

            try:
                node = metadata_dict_to_node(metadata)
                node.text = text
            except Exception:
                # Legacy support for old nodes
                node_info = source.get("node_info")
                relationships = source.get("relationships") or {}
                start_char_idx = None
                end_char_idx = None
                if isinstance(node_info, dict):
                    start_char_idx = node_info.get("start", None)
                    end_char_idx = node_info.get("end", None)

                node = TextNode(
                    text=text,
                    metadata=metadata,
                    id_=node_id,
                    start_char_idx=start_char_idx,
                    end_char_idx=end_char_idx,
                    relationships=relationships,
                )
            ids.append(node_id)
            nodes.append(node)
            scores.append(hit["_score"])

        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=scores)


class LindormVectorStore(BasePydanticVectorStore):
    """
    Lindorm vector store.

    Args:
        client (LindormVectorClient): Vector index client to use.
            for data insertion/querying.

    Examples:
        `pip install llama-index`
        `pip install opensearch-py`
        `pip install llama-index-vector-stores-lindorm`


        ```python
        from llama_index.vector_stores.lindorm import (
            LindormVectorStore,
            LindormVectorClient,
        )

        # lindorm instance info
        # how to obtain an lindorm search instance:
        # https://alibabacloud.com/help/en/lindorm/latest/create-an-instance

        # how to access your lindorm search instance:
        # https://www.alibabacloud.com/help/en/lindorm/latest/view-endpoints

        # run curl commands to connect to and use LindormSearch:
        # https://www.alibabacloud.com/help/en/lindorm/latest/connect-and-use-the-search-engine-with-the-curl-command
        host = "ld-bp******jm*******-proxy-search-pub.lindorm.aliyuncs.com"
        port = 30070
        username = 'your_username'
        password = 'your_password'

        # index to demonstrate the VectorStore impl
        index_name = "lindorm_test_index"

        # extension param of lindorm search, number of cluster units to query; between 1 and method.parameters.nlist.
        nprobe = "a number(string type)"

        # extension param of lindorm search, usually used to improve recall accuracy, but it increases performance overhead;
        #   between 1 and 200; default: 10.
        reorder_factor = "a number(string type)"

        # LindormVectorClient encapsulates logic for a single index with vector search enabled
        client = LindormVectorClient(
            host=host,
            port=port,
            username=username,
            password=password,
            index=index_name,
            dimension=1536, # match with your embedding model
            nprobe=nprobe,
            reorder_factor=reorder_factor,
            # filter_type="pre_filter/post_filter(default)"
        )

        # initialize vector store
        vector_store = LindormVectorStore(client)
        ```

    """

    stores_text: bool = True
    _client: LindormVectorClient = PrivateAttr(default=None)

    def __init__(
        self,
        client: LindormVectorClient,
    ) -> None:
        """Initialize params."""
        super().__init__()
        self._client = client

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to index.
        Synchronous wrapper,using asynchronous logic of async_add function in synchronous way.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings.

        Returns:
            List[str]: List of node_ids

        """
        return asyncio.get_event_loop().run_until_complete(
            self.async_add(nodes, **add_kwargs)
        )

    async def async_add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Async add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings.

        Returns:
            List[str]: List of node_ids

        """
        await self._client.index_results(nodes)
        return [result.node_id for result in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using a ref_doc_id.
        Synchronous wrapper,using asynchronous logic of async_add function in synchronous way.

        Args:
            ref_doc_id (str): The doc_id of the document whose nodes should be deleted.

        """
        asyncio.get_event_loop().run_until_complete(
            self.adelete(ref_doc_id, **delete_kwargs)
        )

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Async delete nodes using a ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document whose nodes should be deleted.

        """
        await self._client.delete_by_doc_id(ref_doc_id)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.
        Synchronous wrapper,using asynchronous logic of async_add function in synchronous way.

        Args:
            query (VectorStoreQuery): Store query object.

        """
        return asyncio.get_event_loop().run_until_complete(self.aquery(query, **kwargs))

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """
        Async query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery): Store query object.

        """
        query_embedding = cast(List[float], query.query_embedding)
        return await self._client.aquery(
            query.mode,
            query.query_str,
            query_embedding,
            query.similarity_top_k,
            filters=query.filters,
        )
