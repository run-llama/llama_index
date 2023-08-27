
from typing import Any, Dict, List, Optional
from sparqlwrapper import SPARQLWrapper, JSON, INSERT, DELETE, SELECT
from llama_index.graph_stores.types import GraphStore

class SparqlGraphStore(GraphStore):
    def __init__(self, endpoint_url: str) -> None:
        self._client = SPARQLWrapper(endpoint_url)
        self._client.setReturnFormat(JSON)

    @property
    def client(self) -> Any:
        return self._client

    def get(self, subj: str) -> List[List[str]]:
        query = f'''
        SELECT ?rel ?obj WHERE {{
            <{subj}> ?rel ?obj .
        }}
        '''
        self._client.setQuery(query)
        results = self._client.query().convert()
        return [[subj, result['rel']['value'], result['obj']['value']] for result in results['results']['bindings']]

    def get_rel_map(self, subjs: Optional[List[str]] = None, depth: int = 2) -> Dict[str, List[List[str]]]:
        if subjs is None:
            return {}

        rel_map = {}
        for subj in subjs:
            rel_map[subj] = self.get(subj)

        return rel_map

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        query = f'''
        INSERT DATA {{
            <{subj}> <{rel}> <{obj}> .
        }}
        '''
        self._client.setQuery(query)
        self._client.setMethod(INSERT)
        self._client.query()

    def delete(self, subj: str, rel: str, obj: str) -> None:
        query = f'''
        DELETE WHERE {{
            <{subj}> <{rel}> <{obj}> .
        }}
        '''
        self._client.setQuery(query)
        self._client.setMethod(DELETE)
        self._client.query()

    def persist(self, persist_path: str) -> None:
        pass

    def get_schema(self, refresh: bool = False) -> str:
        query = '''
        SELECT DISTINCT ?pred WHERE {
            ?s ?pred ?o .
        }
        '''
        self._client.setQuery(query)
        results = self._client.query().convert()
        return [result['pred']['value'] for result in results['results']['bindings']]

    def query(self, query: str, param_map: Optional[Dict[str, Any]] = {}) -> Any:
        self._client.setQuery(query)
        self._client.setMethod(SELECT)
        return self._client.query().convert()
