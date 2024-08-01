""" Jaguar Vector Store.

. A distributed vector database
. The ZeroMove feature enables instant horizontal scalability
. Multimodal: embeddings, text, images, videos, PDFs, audio, time series, and geospatial
. All-masters: allows both parallel reads and writes
. Anomaly detection capabilities: anomaly and anomamous
. RAG support: combines LLMs with proprietary and real-time data
. Shared metadata: sharing of metadata across multiple vector indexes
. Distance metrics: Euclidean, Cosine, InnerProduct, Manhatten, Chebyshev, Hamming, Jeccard, Minkowski

"""

import datetime
import json
import logging
from typing import Any, List, Optional, Tuple, Union, cast

from llama_index.legacy.schema import BaseNode, Document, TextNode
from llama_index.legacy.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

logger = logging.getLogger(__name__)


class JaguarVectorStore(VectorStore):
    """Jaguar vector store.

    See http://www.jaguardb.com
    See http://github.com/fserv/jaguar-sdk

    Example:
       .. code-block:: python

           vectorstore = JaguarVectorStore(
               pod = 'vdb',
               store = 'mystore',
               vector_index = 'v',
               vector_type = 'cosine_fraction_float',
               vector_dimension = 1536,
               url='http://192.168.8.88:8080/fwww/',
           )
    """

    stores_text: bool = True

    def __init__(
        self,
        pod: str,
        store: str,
        vector_index: str,
        vector_type: str,
        vector_dimension: int,
        url: str,
    ):
        """Constructor of JaguarVectorStore.

        Args:
            pod: str:  name of the pod (database)
            store: str:  name of vector store in the pod
            vector_index: str:  name of vector index of the store
            vector_type: str:  type of the vector index
            vector_dimension: int:  dimension of the vector index
            url: str:  URL end point of jaguar http server
        """
        self._pod = pod
        self._store = store
        self._vector_index = vector_index
        self._vector_type = vector_type
        self._vector_dimension = vector_dimension

        try:
            from jaguardb_http_client.JaguarHttpClient import JaguarHttpClient
        except ImportError:
            logger.error("E0001 error import JaguarHttpClient")
            raise ValueError(
                "Could not import jaguardb-http-client python package. "
                "Please install it with `pip install -U jaguardb-http-client`"
            )

        self._jag = JaguarHttpClient(url)
        self._token = ""

    def __del__(self) -> None:
        pass

    @classmethod
    def class_name(cls) -> str:
        return "JaguarVectorStore"

    @property
    def client(self) -> Any:
        """Get client."""
        return self._jag

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings
        """
        use_node_metadata = add_kwargs.get("use_node_metadata", False)
        ids = []
        for node in nodes:
            text = node.get_text()
            embedding = node.get_embedding()
            if use_node_metadata is True:
                metadata = node.metadata
            else:
                metadata = None
            zid = self.add_text(text, embedding, metadata, **add_kwargs)
            ids.append(zid)

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.
        """
        podstore = self._pod + "." + self._store
        q = "delete from " + podstore + " where zid='" + ref_doc_id + "'"
        self.run(q)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query: VectorStoreQuery object
            kwargs:  may contain 'where', 'metadata_fields', 'args', 'fetch_k'
        """
        embedding = query.query_embedding
        k = query.similarity_top_k
        (nodes, ids, simscores) = self.similarity_search_with_score(
            embedding, k=k, form="node", **kwargs
        )
        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=simscores)

    def load_documents(
        self, embedding: List[float], k: int, **kwargs: Any
    ) -> List[Document]:
        """Query index to load top k most similar documents.

        Args:
            embedding: a list of floats
            k: topK number
            kwargs:  may contain 'where', 'metadata_fields', 'args', 'fetch_k'
        """
        return cast(
            List[Document],
            self.similarity_search_with_score(embedding, k=k, form="doc", **kwargs),
        )

    def create(
        self,
        metadata_fields: str,
        text_size: int,
    ) -> None:
        """
        create the vector store on the backend database.

        Args:
            metadata_fields (str):  exrta metadata columns and types
        Returns:
            True if successful; False if not successful
        """
        podstore = self._pod + "." + self._store

        """
        v:text column is required.
        """
        q = "create store "
        q += podstore
        q += f" ({self._vector_index} vector({self._vector_dimension},"
        q += f" '{self._vector_type}'),"
        q += f"  v:text char({text_size}),"
        q += metadata_fields + ")"
        self.run(q)

    def add_text(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> str:
        """
        Add  texts through the embeddings and add to the vectorstore.

        Args:
          texts: text string to add to the jaguar vector store.
          embedding: embedding vector of the text, list of floats
          metadata: {'file_path': '../data/paul_graham/paul_graham_essay.txt',
                          'file_name': 'paul_graham_essay.txt',
                          'file_type': 'text/plain',
                          'file_size': 75042,
                          'creation_date': '2023-12-24',
                          'last_modified_date': '2023-12-24',
                          'last_accessed_date': '2023-12-28'}
          kwargs: vector_index=name_of_vector_index
                  file_column=name_of_file_column
                  metadata={...}

        Returns:
            id from adding the text into the vectorstore
        """
        text = text.replace("'", "\\'")
        vcol = self._vector_index
        filecol = kwargs.get("file_column", "")
        text_tag = kwargs.get("text_tag", "")

        if text_tag != "":
            text = text_tag + " " + text

        podstorevcol = self._pod + "." + self._store + "." + vcol
        q = "textcol " + podstorevcol
        js = self.run(q)
        if js == "":
            return ""
        textcol = js["data"]

        zid = ""
        if metadata is None:
            ### no metadata and no files to upload
            str_vec = [str(x) for x in embedding]
            values_comma = ",".join(str_vec)
            podstore = self._pod + "." + self._store
            q = "insert into " + podstore + " ("
            q += vcol + "," + textcol + ") values ('" + values_comma
            q += "','" + text + "')"
            js = self.run(q, False)
            zid = js["zid"]
        else:
            str_vec = [str(x) for x in embedding]
            nvec, vvec, filepath = self._parseMeta(metadata, filecol)
            if filecol != "":
                rc = self._jag.postFile(self._token, filepath, 1)
                if not rc:
                    return ""
            names_comma = ",".join(nvec)
            names_comma += "," + vcol
            ## col1,col2,col3,vecl

            if vvec is not None and len(vvec) > 0:
                values_comma = "'" + "','".join(vvec) + "'"
            else:
                values_comma = "'" + "','".join(vvec) + "'"

            ### 'va1','val2','val3'
            values_comma += ",'" + ",".join(str_vec) + "'"
            ### 'v1,v2,v3'
            podstore = self._pod + "." + self._store
            q = "insert into " + podstore + " ("
            q += names_comma + "," + textcol + ") values (" + values_comma
            q += ",'" + text + "')"
            if filecol != "":
                js = self.run(q, True)
            else:
                js = self.run(q, False)
            zid = js["zid"]

        return zid

    def similarity_search_with_score(
        self,
        embedding: Optional[List[float]],
        k: int = 3,
        form: str = "node",
        **kwargs: Any,
    ) -> Union[Tuple[List[TextNode], List[str], List[float]], List[Document]]:
        """Return nodes most similar to query embedding, along with ids and scores.

        Args:
            embedding: embedding of text to look up.
            k: Number of nodes to return. Defaults to 3.
            form: if "node", return Tuple[List[TextNode], List[str], List[float]]
                  if "doc", return List[Document]
            kwargs: may have where, metadata_fields, args, fetch_k
        Returns:
            Tuple(list of nodes, list of ids, list of similaity scores)
        """
        where = kwargs.get("where", None)
        metadata_fields = kwargs.get("metadata_fields", None)

        args = kwargs.get("args", None)
        fetch_k = kwargs.get("fetch_k", -1)

        vcol = self._vector_index
        vtype = self._vector_type
        if embedding is None:
            return ([], [], [])
        str_embeddings = [str(f) for f in embedding]
        qv_comma = ",".join(str_embeddings)
        podstore = self._pod + "." + self._store
        q = (
            "select similarity("
            + vcol
            + ",'"
            + qv_comma
            + "','topk="
            + str(k)
            + ",fetch_k="
            + str(fetch_k)
            + ",type="
            + vtype
        )
        q += ",with_score=yes,with_text=yes"
        if args is not None:
            q += "," + args

        if metadata_fields is not None:
            x = "&".join(metadata_fields)
            q += ",metadata=" + x

        q += "') from " + podstore

        if where is not None:
            q += " where " + where

        jarr = self.run(q)

        if jarr is None:
            return ([], [], [])

        nodes = []
        ids = []
        simscores = []
        docs = []
        for js in jarr:
            score = js["score"]
            text = js["text"]
            zid = js["zid"]

            md = {}
            md["zid"] = zid
            if metadata_fields is not None:
                for m in metadata_fields:
                    mv = js[m]
                    md[m] = mv

            if form == "node":
                node = TextNode(
                    id_=zid,
                    text=text,
                    metadata=md,
                )
                nodes.append(node)
                ids.append(zid)
                simscores.append(float(score))
            else:
                doc = Document(
                    id_=zid,
                    text=text,
                    metadata=md,
                )
                docs.append(doc)

        if form == "node":
            return (nodes, ids, simscores)
        else:
            return docs

    def is_anomalous(
        self,
        node: BaseNode,
        **kwargs: Any,
    ) -> bool:
        """Detect if given text is anomalous from the dataset.

        Args:
            query: Text to detect if it is anomaly
        Returns:
            True or False
        """
        vcol = self._vector_index
        vtype = self._vector_type
        str_embeddings = [str(f) for f in node.get_embedding()]
        qv_comma = ",".join(str_embeddings)
        podstore = self._pod + "." + self._store
        q = "select anomalous(" + vcol + ", '" + qv_comma + "', 'type=" + vtype + "')"
        q += " from " + podstore

        js = self.run(q)
        if isinstance(js, list) and len(js) == 0:
            return False
        jd = json.loads(js[0])
        if jd["anomalous"] == "YES":
            return True
        return False

    def run(self, query: str, withFile: bool = False) -> dict:
        """Run any query statement in jaguardb.

        Args:
            query (str): query statement to jaguardb
        Returns:
            None for invalid token, or
            json result string
        """
        if self._token == "":
            logger.error(f"E0005 error run({query})")
            return {}

        resp = self._jag.post(query, self._token, withFile)
        txt = resp.text
        try:
            return json.loads(txt)
        except Exception:
            return {}

    def count(self) -> int:
        """Count records of a store in jaguardb.

        Args: no args
        Returns: (int) number of records in pod store
        """
        podstore = self._pod + "." + self._store
        q = "select count() from " + podstore
        js = self.run(q)
        if isinstance(js, list) and len(js) == 0:
            return 0
        jd = json.loads(js[0])
        return int(jd["data"])

    def clear(self) -> None:
        """Delete all records in jaguardb.

        Args: No args
        Returns: None
        """
        podstore = self._pod + "." + self._store
        q = "truncate store " + podstore
        self.run(q)

    def drop(self) -> None:
        """Drop or remove a store in jaguardb.

        Args: no args
        Returns: None
        """
        podstore = self._pod + "." + self._store
        q = "drop store " + podstore
        self.run(q)

    def prt(self, msg: str) -> None:
        nows = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("/tmp/debugjaguar.log", "a") as file:
            print(f"{nows} msg={msg}", file=file, flush=True)

    def login(
        self,
        jaguar_api_key: Optional[str] = "",
    ) -> bool:
        """Login to jaguar server with a jaguar_api_key or let self._jag find a key.

        Args:
            optional jaguar_api_key (str): API key of user to jaguardb server
        Returns:
            True if successful; False if not successful
        """
        if jaguar_api_key == "":
            jaguar_api_key = self._jag.getApiKey()
        self._jaguar_api_key = jaguar_api_key
        self._token = self._jag.login(jaguar_api_key)
        if self._token == "":
            logger.error("E0001 error init(): invalid jaguar_api_key")
            return False
        return True

    def logout(self) -> None:
        """Logout to cleanup resources.

        Args: no args
        Returns: None
        """
        self._jag.logout(self._token)

    def _parseMeta(self, nvmap: dict, filecol: str) -> Tuple[List[str], List[str], str]:
        filepath = ""
        if filecol == "":
            nvec = list(nvmap.keys())
            vvec = list(nvmap.values())
        else:
            nvec = []
            vvec = []
            if filecol in nvmap:
                nvec.append(filecol)
                vvec.append(nvmap[filecol])
                filepath = nvmap[filecol]

            for k, v in nvmap.items():
                if k != filecol:
                    nvec.append(k)
                    vvec.append(v)

        return nvec, vvec, filepath
