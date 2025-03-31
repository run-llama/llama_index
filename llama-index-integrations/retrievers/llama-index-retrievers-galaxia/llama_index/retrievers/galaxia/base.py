import time
import http.client
import json

from typing import List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


class GalaxiaClient:
    def __init__(self, api_url, api_key, knowledge_base_id, n_retries, wait_time):
        self.api_url = api_url
        self.api_key = api_key
        self.knowledge_base_id = knowledge_base_id
        self.n_retries = n_retries
        self.wait_time = wait_time

        self.headers = {
            'X-Api-Key': api_key,
            'Content-Type': "application/json"
        }


    def initalize(self, conn, question):
        payload = "{\n  \"algorithmVersion\":\"%s\",\n  \"text\":\"%s\" \n}" % (
            self.knowledge_base_id,
            question.replace('"', '\\"')
        )
        
        conn.request("POST", "/analyze/initialize", payload, self.headers)
        
        res = conn.getresponse()
        data = res.read()
        result = json.loads(data.decode("utf-8"))
        return result
    
    
    def check_status(self, conn, init_res):
        payload = "{\n  \"operationId\": \"%s\"\n}" % init_res['operationId']
        
        conn.request("POST", "/analyze/status", payload, self.headers)
        
        res = conn.getresponse()
        data = res.read()
        result = json.loads(data.decode("utf-8"))
        return result
    
    
    def get_result(self, conn, init_res):
        payload = "{\n  \"operationId\": \"%s\"\n}" % init_res['operationId']
        
        conn.request("POST", "/analyze/result", payload, self.headers)
        
        res = conn.getresponse()
        data = res.read()
        result = json.loads(data.decode("utf-8"))
        return result


    def retrieve(self, query):
        conn = http.client.HTTPSConnection(self.api_url)

        flag_init = False
        for i in range(self.n_retries):
            init_res = self.initalize(conn, query)

            if 'operationId' in init_res.keys():
                flag_init = True
                break

            time.sleep(self.wait_time*i)

        if not flag_init:
            # failed to init
            return None

        flag_proc = False
        for i in range(1, self.n_retries+1):
            time.sleep(self.wait_time*i)
            status = self.check_status(conn, init_res)

            if status['status'] == 'processed':
                flag_proc = True
                break

        if flag_proc:
            res = self.get_result(conn, init_res)
            return res['result']['resultItems']

        else:
            # failed to process
            return None



class GalaxiaRetriever(BaseRetriever):
    """Galaxia knowledge retriever

    before using the API create your knowledge base here:
    beta.cloud.smabbler.com/

    learn more here:
    https://smabbler.gitbook.io/smabbler/api-rag/smabblers-api-rag

    Args:
        api_url : url of galaxia API, e.g. "https://beta.api.smabbler.com"
        api_key : API key
        knowledge_base_id : ID of the knowledge base (galaxia model)

    Example:
        .. code-block:: python

            from llama_index.retrievers.galaxia import GalaxiaRetriever
            from llama_index.core.schema import QueryBundle

            retriever = GalaxiaRetriever(
                api_url="https://beta.api.smabbler.com",
                api_key="<key>",
                knowledge_base_id="<knowledge_base_id>",
            )

            result = retriever._retrieve(QueryBundle(
                "<test question>"
            ))

            print(result)
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        knowledge_base_id: str,
        n_retries: int = 20,
        wait_time: int = 5,
        callback_manager: Optional[CallbackManager] = None,
    ):
        self._client = GalaxiaClient(
            api_url, api_key, knowledge_base_id, n_retries, wait_time
        )

        super().__init__(callback_manager)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """ """
        query = query_bundle.query_str
        response = self._client.retrieve(query)

        if response is None:
            return []

        score = 1
        node_with_score = []

        for res in response:
            node_with_score.append(
                NodeWithScore(
                    node=TextNode(
                        text=res["category"],
                        metadata={
                            "model":res['model'],
                            "file":res['group'],
                        },
                    ),
                    score=res["rank"],
                )
            )

        return node_with_score
