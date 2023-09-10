# TODO

from llama_index.graph_stores import SparqlGraphStore

# SPARQL Config

ENDPOINT = 'https://fuseki.hyperdata.it/llama_index_sparql-test/'  # ok to clean

GRAPH = 'http://purl.org/stuff/guardians'
BASE_URI = 'http://purl.org/stuff/data'

graph_store = SparqlGraphStore(
    sparql_endpoint=ENDPOINT,
    sparql_graph=GRAPH,
    sparql_base_uri=BASE_URI,
)


def test_simple_query():
    print('start test')
    # results = graph_store.select_triplets('Peter Quill', 10)
    results = graph_store.rels('Peter Quill', 10)
    print(results)


# test_simple_query()
