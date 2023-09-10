import re
import random
import string
from string import Template
from SPARQLWrapper import SPARQLWrapper, GET, POST, JSON, DIGEST
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import logging
import fsspec
from llama_index.graph_stores.types import GraphStore

DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_FNAME = "graph_store.json"

logging.basicConfig(filename='sparqy.log', filemode='w', level=logging.INFO)
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
        self.prior_subjs = []
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
        sparql_client.setReturnFormat(JSON)
        results = []
        try:
            returned = sparql_client.queryAndConvert()
            for r in returned["results"]["bindings"]:
                results.append(r)
        except Exception as e:
            logger.info(e)
        return results

    def sparql_update(self, query_string):
        print(query_string)
        sparql_client = SPARQLWrapper(self.sparql_endpoint)
        sparql_client.setMethod(POST)
        sparql_client.setQuery(query_string)
        results = sparql_client.query()
        message = results.response.read().decode('utf-8')
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
        str = str.encode("unicode_escape").decode(
            "utf-8").replace('\\x', '\\u00')
        # single and double quotes
        str = re.sub(r'(["\'])', r'\\\1', str)
        return str

    def unescape_from_rdf(self, str):
        # Unescape single and double quotes
        str = re.sub(r'\\(["\'])', r'\1', str)
        # Replace SPARQL's unicode escape sequences with the actual characters
        str = bytes(str, "utf-8").decode("unicode_escape")
        return str

    def select_triplets(self, subj, limit=-1):
        print('select triplets')
        logger.info('sparql get_triplets called')
        subj = self.escape_for_rdf(subj)
        template = Template("""
            $prefixes
            SELECT DISTINCT ?rel ?obj WHERE {
                GRAPH <http://purl.org/stuff/guardians> {
                    ?triplet a er:Triplet ;
                    er:subject ?subject ;
                    er:property ?property ;
                er:object ?object .

                ?subject er:value "$subject" .
                ?property er:value ?rel .
                ?object er:value ?obj .
                }
            }
            $limit_str
        """)

        limit_str = ''
        if (limit > 0):
            limit_str = '\nLIMIT ' + str(limit)

        query_string = template.substitute(
            {'prefixes': self.sparql_prefixes, 'subject': subj,  'limit_str': limit_str})

        triplets = self.sparql_query(query_string)
        return triplets

    def rels(self, subj, limit=-1):
        logger.info('#### sparql get_triplets called')
        subj = self.escape_for_rdf(subj)
        template = Template("""
    $prefixes
    SELECT DISTINCT ?rel1 ?obj1 ?rel2 ?obj2 WHERE {

    GRAPH <http://purl.org/stuff/guardians> {
        ?triplet a er:Triplet ;
            er:subject ?subject ;
            er:property ?property ;
            er:object ?object .

        ?subject er:value "$subject"  .
        ?property er:value ?rel1 .
        ?object er:value ?obj1 .
                            
    OPTIONAL {
        ?triplet2 a er:Triplet ;
            er:subject ?object ;
            er:property ?property2 ;
            er:object ?object2 .

        ?property2 er:value ?rel2 .
        ?object2 er:value ?obj2 .
    }}}
    $limit_str
        """)

        limit_str = ''
        if (limit > 0):
            limit_str = '\nLIMIT ' + str(limit)

        query_string = template.substitute(
            {'prefixes': self.sparql_prefixes, 'subject': subj,  'limit_str': limit_str})

        triplets_json = self.sparql_query(query_string)

        return self.to_arrows(subj, triplets_json)

    def to_arrows(self, subj, rels):
        """
        Convert subject and relations to a string in the desired format.
        """
        arrows = []
        for r in rels:
            rel1 = r['rel1']['value']
            obj1 = r['obj1']['value']
            rel1 = self.unescape_from_rdf(rel1)  # probably unnecessary
            obj1 = self.unescape_from_rdf(obj1)

            try:
                rel2 = r['rel2']['value']
                obj2 = r['obj2']['value']
                rel2 = self.unescape_from_rdf(rel2)
                obj2 = self.unescape_from_rdf(obj2)
                arrows.append(
                    f"""{subj}, -[{rel1}]->, {obj1}',<-[{rel2}]-, {obj2}""")
            except:
                arrows.append(f"""{subj}, -[{rel1}]->, {obj1}""")
        return {subj: arrows}

# From interface types.py ----------------------------------

# DOES IT ONE-BY-ONE!!!!!!!!!!!!!!!!!!!!!!
    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""
        logger.info('sparql upsert_triplet called')

        # crude loop prevention
        # if obj in self.prior_subjs:
        #    return
        # self.prior_subjs.append(subj)

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
        logger.info('sparql client(self) called')
        ...

    def get(self, subj: str) -> List[List[str]]:
        """Get triplets."""
        logger.info('#### sparql get called')
        triplets = self.get_triplets(str)
        logger.info('triplets = ' + triplets)
        return triplets

    def get_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2
    ) -> Dict[str, List[List[str]]]:
        """Get depth-aware rel map."""
        logger.info('get_rel_map called')
        return self.rels(subjs[0])  # only the first for now
        ...

    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete triplet."""
        logger.info('delete called')
        ...

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        """Persist the graph store to a file."""
        logger.info('persist called')
        return None

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the graph store."""
        logger.info('get_schema called')
        ...

    def query(self, query: str, param_map: Optional[Dict[str, Any]] = {}) -> Any:
        """Query the graph store with statement and parameters."""
        logger.info('query called')
        ...
