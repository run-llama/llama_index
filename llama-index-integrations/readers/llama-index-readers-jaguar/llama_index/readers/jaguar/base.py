"""Jaguar Reader."""

import datetime
import json
from typing import Any, List, Optional

from jaguardb_http_client.JaguarHttpClient import JaguarHttpClient
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class JaguarReader(BaseReader):
    """
    Jaguar reader.
    Retrieve documents from existing persisted Jaguar store.
    """

    def __init__(
        self,
        pod: str,
        store: str,
        vector_index: str,
        vector_type: str,
        vector_dimension: int,
        url: str,
    ):
        """
        Constructor of JaguarReader.

        Args:
            pod: name of the pod (database)
            store: name of vector store in the pod
            vector_index: name of vector index of the store
            vector_type: type of the vector index
            vector_dimension: dimension of the vector index
            url: end point URL of jaguar http server

        """
        self._pod = pod
        self._store = store
        self._vector_index = vector_index
        self._vector_type = vector_type
        self._vector_dimension = vector_dimension
        self._jag = JaguarHttpClient(url)
        self._token = ""

    def login(
        self,
        jaguar_api_key: Optional[str] = "",
    ) -> bool:
        """
        Login to jaguar server with a jaguar_api_key or let self._jag find a key.

        Args:
            optional jaguar_api_key (str): API key of user to jaguardb server.
            If not provided, jaguar api key is read from environment variable
            JAGUAR_API_KEY or from file $HOME/.jagrc
        Returns:
            True if successful; False if not successful

        """
        if jaguar_api_key == "":
            jaguar_api_key = self._jag.getApiKey()
        self._jaguar_api_key = jaguar_api_key
        self._token = self._jag.login(jaguar_api_key)
        return self._token != ""

    def logout(self) -> None:
        """
        Logout from jaguar server to cleanup resources.

        Args: no args
        Returns: None
        """
        self._jag.logout(self._token)

    def load_data(
        self,
        embedding: Optional[List[float]] = None,
        k: int = 10,
        metadata_fields: Optional[List[str]] = None,
        where: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Load data from the jaguar vector store.

        Args:
            embedding: list of float number for vector. If this
                       is given, it returns topk similar documents.
            k: Number of results to return.
            where: "a = '100' or ( b > 100 and c < 200 )"
                   If embedding is not given, it finds values
                   of columns in metadata_fields, and the text value.
            metadata_fields: Optional[List[str]] a list of metadata fields to load
                       in addition to the text document

        Returns:
            List of documents

        """
        if embedding is not None:
            return self._load_similar_data(
                embedding=embedding,
                k=k,
                metadata_fields=metadata_fields,
                where=where,
                **kwargs,
            )
        else:
            return self._load_store_data(
                k=k, metadata_fields=metadata_fields, where=where, **kwargs
            )

    def _load_similar_data(
        self,
        embedding: List[float],
        k: int = 10,
        metadata_fields: Optional[List[str]] = None,
        where: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Load data by similarity search from the jaguar store."""
        ### args is additional search conditions, such as time decay
        args = kwargs.get("args")
        fetch_k = kwargs.get("fetch_k", -1)

        vcol = self._vector_index
        vtype = self._vector_type
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
        q += ",with_score,with_text"
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
            return []

        docs = []
        for js in jarr:
            score = js["score"]
            text = js["text"]
            zid = js["zid"]

            md = {}
            md["zid"] = zid
            md["score"] = score
            if metadata_fields is not None:
                for m in metadata_fields:
                    md[m] = js[m]

            doc = Document(
                id_=zid,
                text=text,
                metadata=md,
            )
            docs.append(doc)

        return docs

    def _load_store_data(
        self,
        k: int = 10,
        metadata_fields: Optional[List[str]] = None,
        where: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Load a number of document from the jaguar store."""
        vcol = self._vector_index
        podstore = self._pod + "." + self._store
        txtcol = vcol + ":text"

        sel_str = "zid," + txtcol
        if metadata_fields is not None:
            sel_str += "," + ",".join(metadata_fields)

        q = "select " + sel_str
        q += " from " + podstore

        if where is not None:
            q += " where " + where
        q += " limit " + str(k)

        jarr = self.run(q)
        if jarr is None:
            return []

        docs = []
        for ds in jarr:
            js = json.loads(ds)
            text = js[txtcol]
            zid = js["zid"]

            md = {}
            md["zid"] = zid
            if metadata_fields is not None:
                for m in metadata_fields:
                    md[m] = js[m]

            doc = Document(
                id_=zid,
                text=text,
                metadata=md,
            )
            docs.append(doc)

        return docs

    def run(self, query: str) -> dict:
        """
        Run any query statement in jaguardb.

        Args:
            query (str): query statement to jaguardb
        Returns:
            None for invalid token, or
            json result string

        """
        if self._token == "":
            return {}

        resp = self._jag.post(query, self._token, False)
        txt = resp.text
        try:
            return json.loads(txt)
        except Exception:
            return {}

    def prt(self, msg: str) -> None:
        nows = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("/tmp/debugjaguarrdr.log", "a") as file:
            print(f"{nows} msg={msg}", file=file, flush=True)
