from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import fsspec

from SPARQLWrapper import SPARQLWrapper

DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_FNAME = "graph_store.json"

"""
@prefix schema: <http://schema.org/>.
@prefix stuff: <http://hyperdata.it/stuff>.

<https://en.wikipedia.org/wiki/Etruscan_civilization> a schema:TextDigitalDocument .
stuff:triplet1234 a stuff:Triplet ;
stuff:source <https://en.wikipedia.org/wiki/Etruscan_civilization>
stuff:A "first" ;
stuff:B "second" ;
stuff:C "third" .
"""

@runtime_checkable
class SparqlStore(GraphStore):
    """SPARQL graph store connector.

    This protocol defines the interface for a graph store, which is responsible
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
    

    # Initialize SPARQLWrapper instance
    sparql = SPARQLWrapper("https://fuseki.hyperdata.it/llama_index-test/")


    schema: str = ""

    @property
    def client(self) -> Any:
        """Get client."""
        return self

    def get(self, subj: str) -> List[List[str]]:
        """Get triplets using the SPARQLWrapper instance."""
        # Define the SPARQL query
        query = """
            SELECT ?s ?p ?o
            WHERE {
                ?s ?p ?o .
                FILTER (?s = <{}>)
            }
        """.format(subj)

        # Set the query to the SPARQLWrapper instance
        self.sparql.setQuery(query)

        # Execute the query and get the results
        results = self.sparql.query().convert()

        # Extract the triplets from the results
        triplets = [[result["s"]["value"], result["p"]["value"], result["o"]["value"]] for result in results["results"]["bindings"]]

        return triplets

    def get_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2
    ) -> Dict[str, List[List[str]]]:
        """Get depth-aware rel map."""
        # Define the SPARQL query
        query = """
            SELECT ?s ?p ?o
            WHERE {
                ?s ?p ?o .
                FILTER (?s IN (<{}>))
                FILTER NOT EXISTS {{
                    ?s ?p ?o2 .
                    ?o2 ?p2 ?o3 .
                }}
            }
        """.format(", ".join(["<{}>".format(subj) for subj in subjs]))

        # Set the query to the SPARQLWrapper instance
        self.sparql.setQuery(query)

        # Execute the query and get the results
        results = self.sparql.query().convert()

        # Extract the triplets from the results
        triplets = [[result["s"]["value"], result["p"]["value"], result["o"]["value"]] for result in results["results"]["bindings"]]

        # Create a dictionary to store the relationship map
        rel_map = {}

        # Populate the relationship map with the triplets
        for triplet in triplets:
            subj, rel, obj = triplet
            if subj not in rel_map:
                rel_map[subj] = []
            rel_map[subj].append([rel, obj])

        return rel_map


    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""
        # Define the SPARQL query for upserting a triplet
        query = """
            INSERT DATA {{
                <{}> <{}> "{}" .
            }}
        """.format(subj, rel, obj)

        # Set the query to the SPARQLWrapper instance
        self.sparql.setQuery(query)

        # Execute the query
        self.sparql.query()



        

    def delete(self, subj: str, rel: str, obj: str) -> None:
        """
        Delete a triplet using the SPARQLWrapper instance.
        """
        # Define the SPARQL query for deleting a triplet
        query = """
            DELETE DATA {{
                <{}> <{}> "{}" .
            }}
        """.format(subj, rel, obj)

        # Set the query to the SPARQLWrapper instance
        self.sparql.setQuery(query)

        # Execute the query
        self.sparql.query()



    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        """Persist the graph store to a file."""
        # Define the SPARQL CONSTRUCT query to get all the data from the graph store
        query = """
            CONSTRUCT {
                ?s ?p ?o .
            }
            WHERE {
                ?s ?p ?o .
            }
        """

        # Set the query to the SPARQLWrapper instance
        self.sparql.setQuery(query)

        # Execute the query and get the results
        results = self.sparql.query().convert()

        # Convert the results to JSON
        data = results.serialize(format='json')

        # Check if a filesystem was provided
        if fs is None:
            # Use the local filesystem if none was provided
            with open(persist_path, 'w') as f:
                f.write(data)
        else:
            # Use the provided filesystem to save the data
            with fs.open(persist_path, 'w') as f:
                f.write(data)
 

    def get_schema(self, refresh: bool = False) -> str:
        """
        Get the schema of the graph store using the SPARQLWrapper instance.
        """
        # Define the SPARQL query to get the schema
        query = """
            SELECT DISTINCT ?p
            WHERE {
                ?s ?p ?o .
            }
        """

        # Set the query to the SPARQLWrapper instance
        self.sparql.setQuery(query)

        # Execute the query and get the results
        results = self.sparql.query().convert()

        # Extract the schema from the results
        schema = [result["p"]["value"] for result in results["results"]["bindings"]]

        return schema

    def query(self, query: str, param_map: Optional[Dict[str, Any]] = {}) -> Any:
        """
        This method uses the SPARQLWrapper instance to execute a SPARQL query on the remote store.
        The query and parameters are provided as arguments.
        """
        # Set the query to the SPARQLWrapper instance
        self.sparql.setQuery(query)

        # If there are parameters, bind them to the query
        if param_map:
            for key, value in param_map.items():
                self.sparql.addParameter(key, value)

        # Execute the query and get the results
        results = self.sparql.query().convert()

        return results
        
        """Query the graph store with statement and parameters."""
        ...

