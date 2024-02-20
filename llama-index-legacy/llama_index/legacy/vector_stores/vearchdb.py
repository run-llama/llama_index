import logging
import os
import time
import typing
import uuid
from typing import TYPE_CHECKING, Any, Iterable, List, Optional

import numpy as np

from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

if TYPE_CHECKING:
    import vearch
logger = logging.getLogger(__name__)


class VearchVectorStore(VectorStore):
    """
    Vearch vector store:
        embeddings are stored within a Vearch table.
        when query, the index uses Vearch to query for the top
        k most similar nodes.

    Args:
        chroma_collection (chromadb.api.models.Collection.Collection):
            ChromaDB collection instance
    """

    flat_metadata: bool = True
    stores_text: bool = True
    _DEFAULT_TABLE_NAME = "liama_index_vearch"
    _DEFAULT_CLUSTER_DB_NAME = "liama_index_vearch_client_db"
    _DEFAULT_VERSION = 1

    def __init__(
        self,
        path_or_url: Optional[str] = None,
        table_name: str = _DEFAULT_TABLE_NAME,
        db_name: str = _DEFAULT_CLUSTER_DB_NAME,
        flag: int = _DEFAULT_VERSION,
        **kwargs: Any,
    ) -> None:
        """
        Initialize vearch vector store
        flag 1 for cluster,0 for standalone.
        """
        try:
            if flag:
                import vearch_cluster
            else:
                import vearch
        except ImportError:
            raise ValueError(
                "Could not import suitable python package."
                "Please install it with `pip install vearch or vearch_cluster."
            )

        if flag:
            if path_or_url is None:
                raise ValueError("Please input url of cluster")
            if not db_name:
                db_name = self._DEFAULT_CLUSTER_DB_NAME
                db_name += "_"
                db_name += str(uuid.uuid4()).split("-")[-1]
            self.using_db_name = db_name
            self.url = path_or_url
            self.vearch = vearch_cluster.VearchCluster(path_or_url)
        else:
            if path_or_url is None:
                metadata_path = os.getcwd().replace("\\", "/")
            else:
                metadata_path = path_or_url
            if not os.path.isdir(metadata_path):
                os.makedirs(metadata_path)
            log_path = os.path.join(metadata_path, "log")
            if not os.path.isdir(log_path):
                os.makedirs(log_path)
            self.vearch = vearch.Engine(metadata_path, log_path)
            self.using_metapath = metadata_path
        if not table_name:
            table_name = self._DEFAULT_TABLE_NAME
            table_name += "_"
            table_name += str(uuid.uuid4()).split("-")[-1]
        self.using_table_name = table_name
        self.flag = flag

    @property
    def client(self) -> Any:
        """Get client."""
        return self.vearch

    def _get_matadata_field(self, metadatas: Optional[List[dict]] = None) -> None:
        field_list = []
        if metadatas:
            for key, value in metadatas[0].items():
                if isinstance(value, int):
                    field_list.append({"field": key, "type": "int"})
                    continue
                if isinstance(value, str):
                    field_list.append({"field": key, "type": "str"})
                    continue
                if isinstance(value, float):
                    field_list.append({"field": key, "type": "float"})
                    continue
                else:
                    raise ValueError("Please check data type,support int, str, float")
        self.field_list = field_list

    def _add_texts(
        self,
        ids: Iterable[str],
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if embeddings is None:
            raise ValueError("embeddings is None")
        self._get_matadata_field(metadatas)
        if self.flag:
            dbs_list = self.vearch.list_dbs()
            if self.using_db_name not in dbs_list:
                create_db_code = self.vearch.create_db(self.using_db_name)
                if not create_db_code:
                    raise ValueError("create db failed!!!")
            space_list = self.vearch.list_spaces(self.using_db_name)
            if self.using_table_name not in space_list:
                create_space_code = self._create_space(len(embeddings[0]))
                if not create_space_code:
                    raise ValueError("create space failed!!!")
            docid = []
            if embeddings is not None and metadatas is not None:
                meta_field_list = [i["field"] for i in self.field_list]
                for text, metadata, embed, id_d in zip(
                    texts, metadatas, embeddings, ids
                ):
                    profiles: typing.Dict[str, Any] = {}
                    profiles["text"] = text
                    for f in meta_field_list:
                        profiles[f] = metadata[f]
                    embed_np = np.array(embed)
                    profiles["text_embedding"] = {
                        "feature": (embed_np / np.linalg.norm(embed_np)).tolist()
                    }
                    insert_res = self.vearch.insert_one(
                        self.using_db_name, self.using_table_name, profiles, id_d
                    )
                    if insert_res["status"] == 200:
                        docid.append(insert_res["_id"])
                        continue
                    else:
                        retry_insert = self.vearch.insert_one(
                            self.using_db_name, self.using_table_name, profiles
                        )
                        docid.append(retry_insert["_id"])
                        continue
        else:
            table_path = os.path.join(
                self.using_metapath, self.using_table_name + ".schema"
            )
            if not os.path.exists(table_path):
                dim = len(embeddings[0])
                response_code = self._create_table(dim)
                if response_code:
                    raise ValueError("create table failed!!!")
            if embeddings is not None and metadatas is not None:
                doc_items = []
                meta_field_list = [i["field"] for i in self.field_list]
                for text, metadata, embed, id_d in zip(
                    texts, metadatas, embeddings, ids
                ):
                    profiles_v: typing.Dict[str, Any] = {}
                    profiles_v["text"] = text
                    profiles_v["_id"] = id_d
                    for f in meta_field_list:
                        profiles_v[f] = metadata[f]
                    embed_np = np.array(embed)
                    profiles_v["text_embedding"] = embed_np / np.linalg.norm(embed_np)
                    doc_items.append(profiles_v)
                docid = self.vearch.add(doc_items)
                t_time = 0
                while len(docid) != len(embeddings):
                    time.sleep(0.5)
                    if t_time > 6:
                        break
                    t_time += 1
                self.vearch.dump()
        return docid

    def _create_table(
        self,
        dim: int = 1024,
    ) -> int:
        """
        Create Standalone VectorStore Table.

        Args:
            dim:dimension of vector.
            fields_list: the field you want to store.

        Return:
            code,0 for success,1 for failed.
        """
        type_dict = {
            "int": vearch.dataType.INT,
            "str": vearch.dataType.STRING,
            "float": vearch.dataType.FLOAT,
        }
        engine_info = {
            "index_size": 1,
            "retrieval_type": "HNSW",
            "retrieval_param": {
                "metric_type": "InnerProduct",
                "nlinks": -1,
                "efConstruction": -1,
            },
        }
        filed_list_add = self.field_list.append({"field": "text", "type": "str"})
        fields = [
            vearch.GammaFieldInfo(fi["field"], type_dict[fi["type"]])
            for fi in filed_list_add
        ]
        vector_field = vearch.GammaVectorInfo(
            name="text_embedding",
            type=vearch.dataType.VECTOR,
            is_index=True,
            dimension=dim,
            model_id="",
            store_type="MemoryOnly",
            store_param={"cache_size": 10000},
        )

        return self.vearch.create_table(
            engine_info,
            name=self.using_table_name,
            fields=fields,
            vector_field=vector_field,
        )

    def _create_space(
        self,
        dim: int = 1024,
    ) -> int:
        """
        Create Cluster VectorStore space.

        Args:
            dim:dimension of vector.

        Return:
            code,0 failed for ,1 for success.
        """
        type_dict = {"int": "integer", "str": "string", "float": "float"}
        space_config = {
            "name": self.using_table_name,
            "partition_num": 1,
            "replica_num": 1,
            "engine": {
                "index_size": 1,
                "retrieval_type": "HNSW",
                "retrieval_param": {
                    "metric_type": "InnerProduct",
                    "nlinks": -1,
                    "efConstruction": -1,
                },
            },
        }
        tmp_proer = {
            "text": {"type": "string"},
            "text_embedding": {
                "type": "vector",
                "index": True,
                "dimension": dim,
                "store_type": "MemoryOnly",
            },
        }
        for item in self.field_list:
            tmp_proer[item["field"]] = {"type": type_dict[item["type"]]}
        space_config["properties"] = tmp_proer

        return self.vearch.create_space(self.using_db_name, space_config)

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        if not self.vearch:
            raise ValueError("Vearch Engine is not initialized")

        embeddings = []
        metadatas = []
        ids = []
        texts = []
        for node in nodes:
            embeddings.append(node.get_embedding())
            metadatas.append(
                node_to_metadata_dict(
                    node, remove_text=True, flat_metadata=self.flat_metadata
                )
            )
            ids.append(node.node_id)
            texts.append(node.get_content(metadata_mode=MetadataMode.NONE) or "")

        return self._add_texts(
            ids=ids,
            texts=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.

        Args:
            query : vector store query.

        Returns:
            VectorStoreQueryResult: Query results.
        """
        meta_filters = {}
        if query.filters is not None:
            for filter_ in query.filters.legacy_filters():
                meta_filters[filter_.key] = filter_.value
        if self.flag:
            meta_field_list = self.vearch.get_space(
                self.using_db_name, self.using_table_name
            )
            meta_field_list.remove("text_embedding")
        embed = query.query_embedding
        if embed is None:
            raise ValueError("query.query_embedding is None")
        k = query.similarity_top_k
        if self.flag:
            query_data = {
                "query": {
                    "sum": [
                        {
                            "field": "text_embedding",
                            "feature": (embed / np.linalg.norm(embed)).tolist(),
                        }
                    ],
                },
                "retrieval_param": {"metric_type": "InnerProduct", "efSearch": 64},
                "size": k,
                "fields": meta_field_list,
            }
            query_result = self.vearch.search(
                self.using_db_name, self.using_table_name, query_data
            )
            res = query_result["hits"]["hits"]
        else:
            query_data = {
                "vector": [
                    {
                        "field": "text_embedding",
                        "feature": embed / np.linalg.norm(embed),
                    }
                ],
                "fields": [],
                "retrieval_param": {"metric_type": "InnerProduct", "efSearch": 64},
                "topn": k,
            }
            query_result = self.vearch.search(query_data)
            res = query_result[0]["result_items"]
        nodes = []
        similarities = []
        ids = []
        for item in res:
            content = ""
            meta_data = {}
            node_id = ""
            if self.flag:
                score = item["_score"]
                item = item["_source"]
            for item_key in item:
                if item_key == "text":
                    content = item[item_key]
                    continue
                elif item_key == "_id":
                    node_id = item[item_key]
                    ids.append(node_id)
                    continue
                if self.flag != 1 and item_key == "score":
                    score = item[item_key]
                    continue
                meta_data[item_key] = item[item_key]
            similarities.append(score)
            try:
                node = metadata_dict_to_node(meta_data)
                node.set_content(content)
            except Exception:
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    meta_data
                )
                node = TextNode(
                    text=content,
                    id_=node_id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )
            nodes.append(node)
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def _delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """
        Delete the documents which have the specified ids.

        Args:
            ids: The ids of the embedding vectors.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful.
            False otherwise, None if not implemented.
        """
        ret: Optional[bool] = None
        tmp_res = []
        if ids is None or len(ids) == 0:
            return ret
        for _id in ids:
            if self.flag:
                ret = self.vearch.delete(self.using_db_name, self.using_table_name, _id)
            else:
                ret = self.vearch.del_doc(_id)
            tmp_res.append(ret)
        return all(i == 0 for i in tmp_res)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        Returns:
            None
        """
        if len(ref_doc_id) == 0:
            return
        ids: List[str] = []
        ids.append(ref_doc_id)
        self._delete(ids)
