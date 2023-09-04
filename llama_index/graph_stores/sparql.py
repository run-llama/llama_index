import re
import random
import string
from SPARQLWrapper import SPARQLWrapper, GET, POST, DIGEST
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import logging
import fsspec
from llama_index.graph_stores.types import GraphStore

DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_FNAME = "graph_store.json"

logging.basicConfig(filename='loggy.log', filemode='w', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info('sparql HERE')


class SparqlGraphStore(GraphStore):
    """SPARQL graph store

    #### TODO This protocol defines the interface for a graph store, which is responsible
    for storing and retrieving knowledge graph data.

    Attributes:
        client: Any: The client used to connect to the graph store.
        get: Callable[[str], List[List[str]]]: Get triplets for a given subject.
        get_rel_map: Callable[[Optional[List[str]], int], Dict[str, List[List[str]]]]:
            Get subjects' rel map in max depth.
        upsert_triplet: Callable[[str, str, str], None]: Upsert a triplet.
        delete: Callable[[str, str, str], None]: Delete a triplet.
        persist: Callable[[str, Optional[fsspec.AbstractFileSystem]], None]:
            Persist the graph store to a file.
        get_schema: Callable[[bool], str]: Get the schema of the graph store.
    """

    def __init__(  # - BATCH SIZE?
        self,
        sparql_endpoint: str,
        sparql_graph: str,
        sparql_base_uri: str,
        # **kwargs: Keyword arguments. (might be useful later)
        **kwargs: Any,
    ):
        self.sparql_endpoint = sparql_endpoint
        self.sparql_graph = sparql_graph
        self.create_graph(sparql_graph)

        self.sparql_prefixes = f"""
            PREFIX er:  <http://purl.org/stuff/er#>
            BASE <{sparql_base_uri}>
        """

# SPARQL comms --------------------------------------------

    def create_graph(self, uri):
        self.sparql_update('CREATE GRAPH <'+uri+'>')

    def sparql_query(self, query_string):
        sparql_client = SPARQLWrapper(self.sparql_endpoint)
        sparql_client.setMethod(GET)
        sparql_client.setQuery(query_string)
        results = sparql_client.query()
        message = results.response.read().decode('utf-8')
        logger.info('Endpoint says : ' + message)
        return message

    def sparql_update(self, query_string):
        sparql_client = SPARQLWrapper(self.sparql_endpoint)
        sparql_client.setMethod(POST)
        sparql_client.setQuery(query_string)
        results = sparql_client.query()
        message = results.response.read().decode('utf-8')
        print('Endpoint says : ' + message)
        logger.info('Endpoint says : ' + message)

    def insert_data(self, data):  # data is 'floating' string, without prefixes
        insert_query = self.sparql_prefixes + \
            'INSERT DATA { GRAPH <'+self.sparql_graph+'> { '+data+' }}'
        self.sparql_update(insert_query)

# Helpers --------------------------------------------------
    def make_id(self):
        """
        Generate a random 4-character string using only numeric characters and capital letters.
        """
        characters = string.ascii_uppercase + string.digits  # available characters
        return ''.join(random.choice(characters) for _ in range(4))

    def escape_for_rdf(self, str):
        # control characters
        str = str.encode("unicode_escape").decode("utf-8")
        # single and double quotes
        str = re.sub(r'(["\'])', r'\\\1', input_str)
        return str

    # TODO unescape from RDF?

    def select_triplets(self, subj):
        logger.info('#### sparql get_triplets called')
        query_string = self.sparql_prefixes + """
            SELECT DISTINCT ?rel ?obj WHERE {
                GRAPH <http://purl.org/stuff/guardians> {
                    ?triplet a er:Triplet ;
                    er:subject ?subject ;
                    er:property ?property ;
                er:object ?object .

                ?subject er:value {subj} .
                ?property er:value ?rel .
                ?object er:value ?obj .
                }
            }
        """
        triplets = self.sparql_query(query_string)
        logger.info('triplets = \n'+str(triplets))

# From interface types.py ----------------------------------
# DOES IT ONE-BY-ONE!!!!!!!!!!!!!!!!!!!!!!

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""
        logger.info('#### sparql upsert_triplet called')

        subj = self.escape_for_rdf(subj)
        rel = self.escape_for_rdf(rel)
        obj = self.escape_for_rdf(obj)

        triplet_fragment = '#T'+self.make_id()
        s_fragment = '#E_'+self.make_id()
        p_fragment = '#R_'+self.make_id()
        o_fragment = '#E_'+self.make_id()

        triple = f"""
            <{triplet_fragment}> a er:Triplet ;
                        er:subject <{s_fragment}> ;
                        er:property <{p_fragment}> ;
                        er:object <{o_fragment}> .

            <{s_fragment}> a er:Entity ;
                        er:value "{subj}" .

            <{p_fragment}> a er:Relationship ;
                        er:value "{rel}" .

            <{o_fragment}> a er:Entity ;
                        er:value "{obj}" .
                """
        self.insert_data(triple)

    def client(self) -> Any:
        """Get client."""
        # logger.info('#### sparql client(self) called')
        ...

    def get(self, subj: str) -> List[List[str]]:
        """Get triplets."""
        logger.info('#### sparql get called')
        triplets = get_triplets(str)
        logger.info('triplets = ' + triplets)
        return triplets

    def get_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2
    ) -> Dict[str, List[List[str]]]:
        """Get depth-aware rel map."""
        logger.info('#### sparql get_rel_map called')
        ...

    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete triplet."""
        logger.info('#### sparql delete called')
        ...

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        """Persist the graph store to a file."""
        logger.info('#### sparql persist called')
        return None

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the graph store."""
        logger.info('#### get_schema called')
        ...

    def query(self, query: str, param_map: Optional[Dict[str, Any]] = {}) -> Any:
        """Query the graph store with statement and parameters."""
        logger.info('#### sparql query called')
        ...
