from string import Template
import pytest
from llama_index.graph_stores.sparql import SparqlGraphStore

ENDPOINT = 'https://fuseki.hyperdata.it/llama_index_sparql-test/'

GRAPH = 'http://purl.org/stuff/llama_index_sparql-test/'
BASE_URI = 'http://purl.org/stuff/data'
TEST_SUBJECT, TEST_RELATION, TEST_OBJECT = 'subject', 'relation','object'

# Fixture to initialize a SparqlGraphStore instance before each test
@pytest.fixture
def sparql_graph_store():
    # clear from dataset
    store = SparqlGraphStore(sparql_endpoint=ENDPOINT, sparql_graph=GRAPH, sparql_base_uri=BASE_URI)
    store.drop_graph(GRAPH)

    # recreate graph
    store = SparqlGraphStore(sparql_endpoint=ENDPOINT, sparql_graph=GRAPH, sparql_base_uri=BASE_URI)
    store.upsert_triplet(subj=TEST_SUBJECT, rel=TEST_RELATION, obj=TEST_OBJECT) # sample data
    return store

# Tests for SparqlGraphStore class methods

def test_init(sparql_graph_store):
    assert isinstance(sparql_graph_store, SparqlGraphStore)

def test_create_graph(sparql_graph_store):
    result = sparql_graph_store.create_graph(uri=GRAPH)
    assert result is None # TODO test existence

def test_client(sparql_graph_store):
    # Assuming the client method returns an object (specific type will be determined later)
    result = sparql_graph_store.client()
    assert result is not None # TODO test type?

def test_get(sparql_graph_store):
    # Assuming one will do...
    result = sparql_graph_store.get(subj=TEST_SUBJECT)
    result_str = result[TEST_SUBJECT][0]
    assert result_str == TEST_SUBJECT+', -['+TEST_RELATION+']->, '+TEST_OBJECT

def test_get_rel_map(sparql_graph_store): # TODO improve
    result = sparql_graph_store.get_rel_map(subjs=[TEST_SUBJECT])
    print('reresult = ')
    print(result)
    assert result == {'subject': ['subject, -[relation]->, object']}
 
def test_delete(sparql_graph_store):
    template = Template("""
        $prefixes
                        
        SELECT ?t WHERE {
            GRAPH <$graph> {
        
            ?t a er:Triplet ;
                        er:subject ?sname ;
                        er:property ?pname ;
                        er:object ?oname .

            ?sname a er:Entity ;
                        er:value "$subj" .

            ?pname a er:Relationship ;
                        er:value "$rel" .

            ?oname a er:Entity ;
                        er:value "$obj" .
            }
        }
    """)
    query = template.substitute({'prefixes': sparql_graph_store.sparql_prefixes, 'graph':sparql_graph_store.sparql_graph, 'subj': TEST_SUBJECT, 'rel': TEST_RELATION, 'obj': TEST_OBJECT})

    print("query = ")
    print(query)
    # check the triple is there
    results = sparql_graph_store.sparql_query(query)
    print("resultsA = ")
    print(str(results))
    # test the triple is gone
    sparql_graph_store.delete(subj=TEST_SUBJECT, rel=TEST_RELATION, obj=TEST_OBJECT)
    results = sparql_graph_store.sparql_query(query)
    print("resultsB = ")
    print(str(results))
    assert results == []

def test_persist(sparql_graph_store):
    # TODO make more useful
    result = sparql_graph_store.persist(persist_path="./test-data")
    assert result is None

def test_get_schema(sparql_graph_store):
    # Assuming the get_schema method returns a string
    result = sparql_graph_store.get_schema()
    assert isinstance(result, str)

def test_query(sparql_graph_store):
    # 
    result = sparql_graph_store.query(query="SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
    print('****')
    print(result)
    print('****')
    assert result is not None