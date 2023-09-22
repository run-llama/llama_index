
import pytest
from llama_index.graph_stores.sparql import SparqlGraphStore

ENDPOINT = 'https://fuseki.hyperdata.it/llama_index_sparql-test/'
GRAPH = 'http://purl.org/stuff/llama_index_sparql-test/'
BASE_URI = 'http://purl.org/stuff/data'

# Fixture to initialize a SparqlGraphStore instance before each test
@pytest.fixture
def sparql_graph_store():
    store = SparqlGraphStore(sparql_endpoint=ENDPOINT, sparql_graph=GRAPH, sparql_base_uri=BASE_URI)
    store.upsert_triplet(subj='subject', rel='relation', obj='object') # sample data
    return store

# Tests for SparqlGraphStore class methods

def test_init(sparql_graph_store):
    assert isinstance(sparql_graph_store, SparqlGraphStore)

def test_create_graph(sparql_graph_store):
    # Assuming the create_graph method does not return anything
    result = sparql_graph_store.create_graph(uri=GRAPH)
    assert result is None # TODO test existence

def test_client(sparql_graph_store):
    # Assuming the client method returns an object (specific type will be determined later)
    result = sparql_graph_store.client()
    assert result is not None # TODO test type?

def test_get(sparql_graph_store):
    # Assuming it doesn't get more than this...
    result = sparql_graph_store.get(subj='subject')
    result_str = result['subject'][0]
    print('****')
    print(result_str)

    print('****')
    assert result_str == 'subject, -[relation]->, object'

def test_get_rel_map(sparql_graph_store):
    # Assuming the get_rel_map method returns a dictionary with string keys and list of lists of strings as values
    result = sparql_graph_store.get_rel_map(subjs=["http://example.com/subject1", "http://example.com/subject2"])
    assert isinstance(result, dict)
    assert all(isinstance(key, str) and isinstance(value, list) and all(isinstance(sub_item, list) and all(isinstance(sub_sub_item, str) for sub_sub_item in sub_item) for sub_item in value) for key, value in result.items())

def test_delete(sparql_graph_store):
    # Assuming the delete method does not return anything
    result = sparql_graph_store.delete(subj="http://example.com/subject", rel="http://example.com/relation", obj="http://example.com/object")
    assert result is None

def test_persist(sparql_graph_store):
    # Assuming the persist method does not return anything
    result = sparql_graph_store.persist(persist_path="http://example.com/persist_path")
    assert result is None

def test_get_schema(sparql_graph_store):
    # Assuming the get_schema method returns a string
    result = sparql_graph_store.get_schema()
    assert isinstance(result, str)

def test_query(sparql_graph_store):
    # Assuming the query method returns an object of any type (specific type will be determined later)
    result = sparql_graph_store.query(query="SELECT ?x WHERE { ?x a <http://example.com/Type> }")
    assert result is not None